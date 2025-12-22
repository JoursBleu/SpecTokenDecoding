# val_sharegpt.py
import os
import argparse
import torch
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
def speculative_decode(base_model, draft_model, input_ids, spec_depth, tokenizer, device):
    """
    Simulate speculative decoding with ground-truth future for acceptance rate calculation.
    """
    input_ids = input_ids.to(device)  # [C]

    base_tokens = base_model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=128,
        pad_token_id=draft_model.config.pad_token_id,
        eos_token_id=draft_model.config.eos_token_id,
    )  # [1, C + K]

    spec_token_ids = tokenizer.convert_tokens_to_ids(
        [f"<|spec_{i}|>" for i in range(1, spec_depth + 1)]
    )
    start_time = time.time()
    step_count = 0
    cur_ids = input_ids.clone()

    draft_input_ids = torch.cat((cur_ids, spec_token_ids), dim=-1)
    context_out = draft_model(input_ids=draft_input_ids, use_cache=True)
    kv_cache = context_out.past_key_values
    draft_logits = context_out.logits
    draft_probs = torch.softmax(draft_logits[:,-(spec_depth+1),:], dim=-1)
    draft_tokens = torch.multinomial(draft_probs.view(-1, draft_probs.size(-1)), 1).view(draft_probs.shape[0], -1)
    kv_cache.crop(cur_ids.shape[1])

    while cur_ids.shape[1] - input_ids.shape[1] < max_new_tokens:
        breakpoint()
        assert(draft_tokens.shape[1] == spec_depth+1)
        assert(kv_cache.seq_length() == cur_ids.shape[1])
        verify_out = draft_model(
            input_ids=draft_tokens,
            do_sample=False,
            past_key_values=kv_cache,
            use_cache=True
        )
        verify_logits = verify_out.logits
        kv_cache = verify_out.past_key_values
        verify_probs = torch.softmax(verify_logits, dim=-1)
        verify_tokens = torch.multinomial(verify_probs.view(-1, verify_probs.size(-1)), 1).view(verify_probs.shape[0], -1)

        posterior_mask = (draft_tokens[:, 1:] == verify_tokens[:, :draft_len]).int()
        accept_length = (torch.cumprod(posterior_mask, dim=-1)).sum(dim=1)+1
        cur_ids = torch.cat((cur_ids, draft_tokens[:, :accept_length]), dim=-1)
        kv_cache.crop(cur_ids.shape[1])
        new_token = verify_tokens[:, accept_length-1]

        draft_input_ids = torch.cat((new_token, spec_token_ids), dim=-1)
        draft_out = draft_model(input_ids=draft_input_ids, past_key_values=kv_cache, use_cache=True)
        draft_logits = draft_out.logits
        draft_prob = torch.softmax(draft_logits, dim=-1)
        draft_tokens = torch.multinomial(draft_prob.view(-1, draft_prob.size(-1)), 1).view(draft_prob.shape[0], -1)

        # print("draft_tokens", draft_tokens)
        # print("verify_tokens", verify_tokens)
        # print("draft_str", tokenizer.decode(draft_tokens[0]))
        # print("verify_str", tokenizer.decode(verify_tokens[0]))
        # print("accept_length", accept_length)
        # print("seqlen", cur_ids.shape[1])
        # print("seq", tokenizer.decode(cur_ids[0]))

        step_count += 1

        if base_model.config.eos_token_id in cur_ids.cpu().tolist() or base_model.config.eos_token_id == new_token:
            break

    end_time = time.time()

    return accepted, aligned_base

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
    draft_base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if args.use_lora:
        draft_model = PeftModel.from_pretrained(draft_base, args.draft_model_path)
    else:
        state_dict = torch.load(os.path.join(args.draft_model_path, "pytorch_model.bin"), map_location="cpu")
        draft_base.load_state_dict(state_dict)
        draft_model = draft_base
    draft_model = draft_model

    # Load base model's trainable params (e.g., spec_embed_tokens)
    print("load trainable params")
    trainable_path = os.path.join(args.draft_model_path, "trainable_base_params.bin")
    if os.path.exists(trainable_path):
        trainable_state = torch.load(trainable_path, map_location="cpu")
        draft_model.load_state_dict(trainable_state, strict=False)  # only load matching keys
    draft_model.to(device).eval()


    # === Load dataset ===
    dataset = SpecValidationDataset(tokenizer, args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_accepted = 0
    total_aligned = 0
    total_drafts = 0
    count = 0

    print(f"Evaluating {args.num_samples} samples...")
    for batch in dataloader:
        if count >= args.num_samples:
            break

        accepted, aligned_base = speculative_decode(
            base_model, draft_model, batch["input_ids"], args.spec_depth, tokenizer, device
        )
        total_accepted += accepted
        total_drafts += args.spec_depth
        total_aligned += aligned_base
        count += 1

        print(f"Processed {count} samples, current acceptance rate: {total_accepted / total_drafts:.2%}, current aligned rate: {total_aligned / total_drafts:.2%}")

    acceptance_rate = total_accepted / total_drafts
    avg_accepted_per_step = total_accepted / count

    print("\n" + "="*50)
    print(f"Speculative Decoding Evaluation Results")
    print(f"Spec depth: {args.spec_depth}")
    print(f"Samples: {count}")
    print(f"Average accepted tokens per draft: {avg_accepted_per_step:.2f} / {args.spec_depth}")
    print(f"Acceptance rate: {acceptance_rate:.2%}")
    print(f"Average aligned tokens: {total_aligned / count:.3}")
    print("="*50)

if __name__ == "__main__":
    main()