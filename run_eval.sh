#!/bin/bash

set -v

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=/lpai/volumes/lpai-yharnam-lx-my/lt/transformers-qwen3/src

export SPEC_LEN=$2
export STEP=$3
export CKPT_PATH=$4


python val_sharegpt_raw.py \
  --model_name /lpai/volumes/lpai-yharnam-lx-my/lt/models/Llama-3.1-8B-Instruct \
  --draft_model_path ./${CKPT_PATH}/step_${STEP} \
  --dataset_name ../data/ShareGPT_Vicuna_unfiltered \
  --dataset_split "train" \
  --use_lora \
  --spec_depth ${SPEC_LEN}


