# val_sharegpt.py
import os
import argparse
import torch
import time
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
import random

class SpecValidationDataset(Dataset):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.spec_token_ids = tokenizer.convert_tokens_to_ids(
            [f"<|spec_{i}|>" for i in range(1, args.spec_depth + 1)]
        )
        self.dataset = load_dataset(
            args.dataset_name,
            split=args.dataset_split,
            streaming=True,
            trust_remote_code=True
        ).shuffle(seed=42, buffer_size=10000)
        self.iter = iter(self.dataset)

    def __len__(self):
        return 1000  # evaluate on 1000 samples

    def __getitem__(self, idx):
        while True:
            try:
                example = next(self.iter)
                if "conversations" in example:
                    conversations = example["conversations"]
                    # 至少需要一轮问答（user + assistant）
                    if len(conversations) < 2:
                        continue
                    # 只取第一轮
                    first_turn = conversations[0]
                    second_turn = conversations[1]
                    
                    # 验证角色顺序：user -> assistant
                    if not (first_turn.get("from", "").lower() in ("human", "user") and
                            second_turn.get("from", "").lower() in ("gpt", "assistant")):
                        continue  # skip malformed

                    messages_user = [
                        {"role": "user", "content": first_turn["value"].strip()},
                    ]

                    text_user = self.tokenizer.apply_chat_template(
                        messages_user,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    continue

                tokens_user = self.tokenizer(text_user, add_special_tokens=False)["input_ids"][:-3]
                return {
                    "input_ids": torch.tensor(tokens_user, dtype=torch.long),
                }

            except Exception:
                continue

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--draft_model_path", type=str, default=None,
                        help="Path to draft model checkpoint (e.g., ./new-model/checkpoint_latest). "
                             "If not provided, uses --output_dir/checkpoint_latest")
    parser.add_argument("--output_dir", type=str, default="./qwen3-spec")
    parser.add_argument("--dataset_name", type=str, default="cerebras/SlimPajama-627B")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--max_context", type=int, default=2048)
    parser.add_argument("--spec_depth", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    return parser.parse_args()

@torch.no_grad()
def speculative_decode(draft_model, input_ids, max_new_tokens, spec_depth, tokenizer, device, base_model):
    """
    Simulate speculative decoding with ground-truth future for acceptance rate calculation.
    """
    input_ids = input_ids.to(device)  # [C]
    print("seq", tokenizer.decode(input_ids[0]))

    base_tokens = base_model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=draft_model.config.pad_token_id,
        eos_token_id=draft_model.config.eos_token_id,
    )  # [1, C + K]

    # draft_gen_tokens = draft_model.generate(
        # input_ids,
        # do_sample=False,
        # max_new_tokens=max_new_tokens,
        # pad_token_id=draft_model.config.pad_token_id,
        # eos_token_id=draft_model.config.eos_token_id,
    # )  # [1, C + K]

    draft_gen_tokens = input_ids.clone()
    context_out = draft_model(input_ids=draft_gen_tokens, use_cache=True, logits_to_keep=[-1])
    past_key_values = context_out.past_key_values
    new_token = torch.argmax(context_out.logits, dim=-1)
    draft_gen_tokens = torch.cat((draft_gen_tokens, new_token), dim=-1)
    while draft_gen_tokens.shape[1] - input_ids.shape[1] < max_new_tokens:
        out = draft_model(
            input_ids=new_token,
            past_key_values=past_key_values,
            use_cache=True
        )
        past_key_values = out.past_key_values
        new_token = torch.argmax(out.logits, dim=-1)
        draft_gen_tokens = torch.cat((draft_gen_tokens, new_token), dim=-1)
        if draft_model.config.eos_token_id == new_token:
            break

    spec_token_ids = tokenizer.convert_tokens_to_ids(
        [f"<|spec_{i}|>" for i in range(1, spec_depth + 1)]
    )
    spec_token_ids = torch.tensor(spec_token_ids,dtype=input_ids.dtype, device=input_ids.device).view(1, spec_depth)
    logits_to_keep = [-i for i in range(spec_depth+1, 0, -1)]
    start_time = time.time()
    step_count = 0
    cur_ids = input_ids.clone()

    draft_input_ids = torch.cat((cur_ids, spec_token_ids), dim=-1)
    context_out = draft_model(input_ids=draft_input_ids, use_cache=True, logits_to_keep=logits_to_keep)
    kv_cache = context_out.past_key_values
    draft_tokens = torch.argmax(context_out.logits, dim=-1)
    kv_cache.crop(cur_ids.shape[1])

    while cur_ids.shape[1] - input_ids.shape[1] < max_new_tokens:
        assert(draft_tokens.shape[1] == spec_depth+1)
        verify_out = draft_model(
            input_ids=draft_tokens,
            past_key_values=kv_cache,
            use_cache=True
        )
        kv_cache = verify_out.past_key_values
        verify_tokens = torch.argmax(verify_out.logits, dim=-1)

        posterior_mask = (draft_tokens[:, 1:] == verify_tokens[:, :spec_depth]).int()
        accept_length = (torch.cumprod(posterior_mask, dim=-1)).sum(dim=1)+1
        cur_ids = torch.cat((cur_ids, draft_tokens[:, :accept_length]), dim=-1)
        kv_cache.crop(cur_ids.shape[1])
        new_token = verify_tokens[:, accept_length-1]

        # print("draft_tokens", draft_tokens)
        # print("verify_tokens", verify_tokens)
        # print("draft_str", tokenizer.decode(draft_tokens[0]))
        # print("verify_str", tokenizer.decode(verify_tokens[0]))
        # print("accept_length", accept_length)
        # print("seqlen", cur_ids.shape[1])
        # print("seq", tokenizer.decode(cur_ids[0]))

        draft_input_ids = torch.cat((new_token, spec_token_ids), dim=-1)
        draft_out = draft_model(input_ids=draft_input_ids, past_key_values=kv_cache, use_cache=True, logits_to_keep=logits_to_keep)
        kv_cache = draft_out.past_key_values
        draft_tokens = torch.argmax(draft_out.logits, dim=-1)
        cur_ids = torch.cat((cur_ids, new_token), dim=-1)
        kv_cache.crop(cur_ids.shape[1])
        accept_length += 1

        step_count += 1

        if draft_model.config.eos_token_id in cur_ids.cpu().tolist() or draft_model.config.eos_token_id == new_token:
            break

    end_time = time.time()
    new_token_count = cur_ids.shape[1] - input_ids.shape[1]
    if((draft_gen_tokens != cur_ids[:, :draft_gen_tokens.shape[1]]).sum()):
        print("base_tokens", base_tokens[:, -128:])
        print("draft_gen_tokens", draft_gen_tokens[:, -128:])
        print("cur_ids", cur_ids[:, -new_token_count:])

    return step_count, new_token_count

def main():
    args = parse_args()
    device = args.device

    # === Load tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(args.draft_model_path, trust_remote_code=True)

    # === Load base model (frozen) ===
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    # === Load draft model ===
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()


    # === Load dataset ===
    dataset = SpecValidationDataset(tokenizer, args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    count = 0
    total_step_count = 0
    total_new_token_count = 0

    print(f"Evaluating {args.num_samples} samples...")
    for batch in dataloader:
        if count >= args.num_samples:
            break

        input_ids = batch["input_ids"]
        if torch.isinf(input_ids).sum().item() or torch.isnan(input_ids).sum().item():
            print("input_ids", input_ids, flush=True)
            breakpoint()

        step_count, new_token_count = speculative_decode(
            draft_model,
            input_ids,
            args.max_new_tokens,
            args.spec_depth,
            tokenizer, device,
            base_model,
        )
        total_new_token_count += new_token_count
        total_step_count += step_count
        total_tps = total_new_token_count / total_step_count / 2
        # total_aligned += aligned_base
        count += 1

        print(f"Processed {count} samples, new tokens: {new_token_count}, tps: {new_token_count / step_count / 2}, avg tps: {total_tps}")


    print("\n" + "="*50)
    print(f"Speculative Decoding Evaluation Results")
    print(f"Spec depth: {args.spec_depth}")
    print(f"Samples: {count}")
    print(f"TPS: {total_tps}")
    print("="*50)

if __name__ == "__main__":
    main()