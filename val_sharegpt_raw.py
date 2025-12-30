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
                    
                    messages = [
                        {"role": "user", "content": first_turn["value"].strip()},
                        {"role": "assistant", "content": second_turn["value"].strip()}
                    ]
                    messages_user = [
                        {"role": "user", "content": first_turn["value"].strip()},
                    ]
                    
                    if not messages[0]["content"] or not messages[1]["content"]:
                        continue
                    
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    text_user = self.tokenizer.apply_chat_template(
                        messages_user,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    continue

                if not text.strip():
                    continue

                tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
                tokens_user = self.tokenizer(text_user, add_special_tokens=False)["input_ids"][:-3]
                len_user = len(tokens_user)
                if len(tokens) - len_user < self.args.spec_depth:
                    continue
                start = random.randint(0, len(tokens) - len_user - self.args.spec_depth) + len_user
                context = tokens[:start]
                input_ids = context + self.spec_token_ids
                base_input_ids = tokens[:start+self.args.spec_depth]
                if len(input_ids) > 2048:
                    continue
                input_ids = torch.tensor(input_ids, dtype=torch.long)
                # if any(torch.isinf(input_ids).item() or any(torch.isnan(input_ids)).item()):
                    # print("input_ids", input_ids, flush=True)
                    # continue
                return {
                    "input_ids": input_ids,
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
    return parser.parse_args()

def color_accepted_tokens(target_tokens, spec_tokens, tokenizer):
    target_tokens = target_tokens[0].tolist()  # 假设 batch_size=1
    spec_tokens = spec_tokens[0].tolist()
    min_len = min(len(target_tokens), len(spec_tokens))
    
    colored_parts = []
    for i in range(len(target_tokens)):
        tok_id = target_tokens[i]
        tok_str = tokenizer.convert_ids_to_tokens(tok_id)
        # 解码单个 token 并处理特殊符号（如开头有 Ġ 表示空格）
        decoded = tokenizer.decode([tok_id], skip_special_tokens=False)
        if i < min_len and target_tokens[i] == spec_tokens[i]:
            # 红色
            colored_parts.append(f"\033[31m{decoded}\033[0m")
        else:
            colored_parts.append(decoded)
    return "".join(colored_parts)

@torch.no_grad()
def speculative_decode(base_model, draft_model, input_ids, spec_depth, tokenizer, device):
    """
    Simulate speculative decoding with ground-truth future for acceptance rate calculation.
    """
    input_ids = input_ids.to(device)  # [C]

    # Step 1: Base model predicts K tokens
    base_tokens = base_model.generate(
        input_ids[:, :-spec_depth],
        do_sample=False,
        max_new_tokens=spec_depth+1,
        pad_token_id=draft_model.config.pad_token_id,
        eos_token_id=draft_model.config.eos_token_id,
    )[:, -spec_depth-1:]  # [1, C + K]

    # Step 2: Target model predicts K tokens
    target_tokens = draft_model.generate(
        input_ids[:, :-spec_depth],
        do_sample=False,
        max_new_tokens=spec_depth+1,
        pad_token_id=draft_model.config.pad_token_id,
        eos_token_id=draft_model.config.eos_token_id,
    )[:, -spec_depth-1:]  # [1, C + K]

    # Step 2: Draft model predicts K tokens
    logits = draft_model(input_ids=input_ids).logits[:, -spec_depth-1:]
    score = torch.softmax(logits, dim=-1) # [bs, seqlen]
    spec_tokens = torch.argmax(logits, dim=-1) # [bs, seqlen, voc_size]
    spec_scores = torch.gather(score, dim=-1, index=spec_tokens.unsqueeze(-1)).squeeze(-1)

    aligned_base = (target_tokens == base_tokens).sum().item()
    accepted = (target_tokens == spec_tokens).sum().item()

    high_spec_scores = spec_scores > 0.7
    high_score = (torch.cumprod(high_spec_scores[:, 1:], dim=-1)).sum(dim=1).item()
    high_score_accepted = min(high_score, accepted-1)

    print("base", base_tokens)
    print("base text:", tokenizer.decode(base_tokens[0]))
    print("target", target_tokens)
    print("target text:", tokenizer.decode(target_tokens[0]))
    print("draft", spec_tokens)
    print("draft score", spec_scores)
    print("draft text:", tokenizer.decode(spec_tokens[0]))
    
    print("aligned_base", aligned_base)
    print("accepted", accepted)
    print("high_score", high_score)
    print("high_score_accepted", high_score_accepted)

    return accepted, aligned_base, high_score, high_score_accepted

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
        draft_model = draft_model.merge_and_unload()
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
        draft_model.model.spec_embed_tokens.data = trainable_state['base_model.model.model.spec_embed_tokens']
    draft_model.to(device).eval()


    # === Load dataset ===
    dataset = SpecValidationDataset(tokenizer, args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_accepted = 0
    total_aligned = 0
    total_drafts = 0
    count = 0
    total_high_score = 0.0001
    total_high_score_accepted = 0.0001

    print(f"Evaluating {args.num_samples} samples...")
    for batch in dataloader:
        if count >= args.num_samples:
            break

        accepted, aligned_base, high_score, high_score_accepted = speculative_decode(
            base_model, draft_model, batch["input_ids"], args.spec_depth, tokenizer, device
        )
        total_accepted += accepted
        total_drafts += args.spec_depth + 1
        total_aligned += aligned_base
        total_high_score += high_score
        total_high_score_accepted += high_score_accepted
        count += 1

        acceptance_rate = total_accepted / total_drafts
        avg_accepted_per_step = total_accepted / count
        high_score_accept_rate = (total_high_score_accepted / total_high_score)

        print(f"Processed {count} samples, current acceptance rate: {acceptance_rate:.2%}, current aligned rate: {total_aligned / total_drafts:.2%}, high score avg count: {total_high_score / count:.2}, high score accept rate: {high_score_accept_rate:.2%}")


    print("\n" + "="*50)
    print(f"Speculative Decoding Evaluation Results")
    print(f"Spec depth: {args.spec_depth}")
    print(f"Samples: {count}")
    print(f"Average accepted tokens per draft: {avg_accepted_per_step:.2f} / {args.spec_depth+1}")
    print(f"Average accepted tokens high score: {high_score_accept_rate:.2}")
    print(f"Acceptance rate: {acceptance_rate:.2%}")
    print(f"Average aligned tokens: {total_aligned / count:.3}")
    print("="*50)

if __name__ == "__main__":
    main()