#!/bin/bash

set -v

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH=/lpai/volumes/lpai-yharnam-lx-my/lt/transformers-qwen3/src

export SPEC_LEN=8

torchrun \
  --nproc_per_node=4 \
  train.py \
  --model_name /lpai/volumes/lpai-yharnam-lx-my/lt/models/Llama-3.1-8B-Instruct \
  --dataset_name /lpai/volumes/lpai-yharnam-lx-my/lt/data/ShareGPT_Vicuna_unfiltered \
  --use_lora \
  --lora_r 128 \
  --lora_layer_ratio 1.0 \
  --learning_rate 5e-5 \
  --vloss_weight 100.0 \
  --ploss_weight 1.0 \
  --hidden_layers -1 -2 \
  --max_context 2048 \
  --spec_depth ${SPEC_LEN} \
  --gradient_accumulation_steps 4 \
  --batch_size 1 \
  --num_steps 100000 \
  --save_steps 1000 \
  --logging_steps 10 \
  --output_dir ./test_ckpt

