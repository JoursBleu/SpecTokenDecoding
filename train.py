# train.py

import argparse
import copy
import json
import random
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser()

    # ===== model / io =====
    parser.add_argument("--model_name_or_path", type=str,
                        default="/lpai/volumes/lpai-yharnam-lx-my/lt/models/Llama-3.1-8B-Instruct",
                        help="Base model path or HF repo")
    parser.add_argument("--output_dir", type=str, default="./ckpts_tmp",
                        help="Checkpoint output directory")

    # ===== training =====
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    # ===== spec objective =====
    parser.add_argument("--spec_len", type=int, default=8,
                        help="Length of spec-token span")
    parser.add_argument("--spec_prob", type=float, default=0.5,
                        help="Probability of applying span corruption")

    # ===== misc =====
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deepspeed", type=str, default="ds_configs/zero2_bf16.json")

    return parser.parse_args()


def load_ds_config(args):
    with open(args.deepspeed, "r") as f:
        ds_config = json.load(f)

    ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps

    return ds_config

def causal_lm_collator(features, pad_token_id, max_length):
    input_ids, attention_mask, labels = [], [], []
    for f in features:
        seq_len = len(f["input_ids"])
        if seq_len > max_length:
            # 截断
            input_ids.append(f["input_ids"][:max_length])
            attention_mask.append(f["attention_mask"][:max_length])
            labels.append(f["labels"][:max_length])
        else:
            # padding
            pad_len = max_length - seq_len
            input_ids.append(f["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
            
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def preprocess(example, tokenizer, args, spec_token_ids):
    conv = example.get("conversations", [])

    # 只取第一轮 user + assistant
    if len(conv) < 2:
        return {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
        }

    if conv[0]["from"] not in ["human", "user"] or conv[1]["from"] not in ["gpt", "assistant"]:
        return {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
        }

    messages_user = [{"role": "user", "content": conv[0]["value"].strip()}]
    messages = [
        {"role": "user", "content": conv[0]["value"].strip()},
        {"role": "assistant", "content": conv[1]["value"].strip()},
    ]

    # 用 apply_chat_template() 生成完整 prompt
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    formatted_user = tokenizer.apply_chat_template(
        messages_user,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Tokenize
    tokens = tokenizer(
        formatted,
        truncation=True,
        max_length=args.max_length,
        padding=False,
    )
    tokens_user = tokenizer(
        formatted_user,
        truncation=True,
        max_length=args.max_length,
        padding=False,
    )
    user_len = len(tokens_user["input_ids"])
    seq_len = len(tokens["input_ids"])

    if random.random() > args.spec_prob:
        # 正常 causal LM：input = full tokens, labels = input shifted
        input_ids = tokens["input_ids"]
        labels = [-100] * len(input_ids)
        labels[user_len:] = input_ids[user_len:]
        return {
            "input_ids": input_ids,
            "attention_mask": tokens["attention_mask"],
            "labels": labels,
        }

    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # spec span masking 逻辑
    # 如果剩余长度不足以放下 spec span，则 skip
    if seq_len - user_len < args.spec_len + 1:
        return {
            "input_ids": None,
            "attention_mask": None,
            "labels": None,
        }

    start = random.randint(0, seq_len - user_len - args.spec_len - 1) + user_len
    input_ids = tokens["input_ids"]
    labels = [-100] * len(input_ids)
    labels[start+1:start + args.spec_len+1] = input_ids[start+1:start + args.spec_len+1]
    input_ids[start:start + args.spec_len] = spec_token_ids
    attention_mask = tokens["attention_mask"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
    )
    tokenizer.add_special_tokens(
        json.load(open("tokenizer/special_tokens.json"))
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 填充 spec span
    spec_token_ids = [
        tokenizer.convert_tokens_to_ids(f"<|spec_{i}|>") for i in range(1, args.spec_len + 1)
    ]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    model.resize_token_embeddings(len(tokenizer))

    dataset = load_dataset(
        "json", 
        data_files="/lpai/volumes/lpai-yharnam-lx-my/lt/data/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json",
        split="train",
    ).shuffle(seed=42)
    # dataset = dataset.map(convert_openhermes, remove_columns=dataset.column_names, num_proc=8)
    dataset = dataset.filter(
        lambda x: (
            isinstance(x.get("conversations"), list)
            and len(x["conversations"]) >= 2
            and len(x["conversations"][0].get("value","").strip()) > 0
            and len(x["conversations"][1].get("value","").strip()) > 0
        )
    )
    dataset = dataset.map(
        lambda x: preprocess(x, tokenizer, args, spec_token_ids),
        remove_columns=dataset.column_names,
        num_proc=8,
    )
    dataset = dataset.filter(lambda x: x is not None and "input_ids" in x and x["input_ids"] is not None)
    # for item in dataset:
        # breakpoint()

    ds_config = load_ds_config(args)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.01,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_steps=5000,
        bf16=True,
        deepspeed=ds_config,
        report_to="tensorboard",
        logging_dir=args.output_dir+"/logs",
        save_total_limit=10,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_drop_last=True,
    )

    data_collator = lambda features: causal_lm_collator(
        features, tokenizer.pad_token_id, args.max_length
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()


