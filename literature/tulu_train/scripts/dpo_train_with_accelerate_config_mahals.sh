#!/bin/bash

# MAHALS: DPO training for Llama 3.1 8B (10% Tulu preference dataset)
#
# example usage (ORIGINAL):
# sh scripts/dpo_train_with_accelerate_config.sh 8 configs/train_configs/dpo/default.yaml
#
# MAHALS usage:
# sh scripts/dpo_train_with_accelerate_config_mahals.sh 1 configs/train_configs/mahals/llama31_8b_dpo_10pct.yaml  # sanity test (1 GPU)
# sh scripts/dpo_train_with_accelerate_config_mahals.sh 8 configs/train_configs/mahals/llama31_8b_dpo_10pct.yaml  # full training (8 GPUs)

# Check if exactly two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <num_gpus> <config_file>"
    echo "Example: $0 2 path/to/config.yaml"
    exit 1
fi

NUM_GPUS="$1"
CONFIG_FILE="$2"

# Generate CUDA_VISIBLE_DEVICES as a range from 0 to NUM_GPUS-1
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export CUDA_VISIBLE_DEVICES

echo "Number of GPUs: $NUM_GPUS"
echo "Using config file: $CONFIG_FILE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory,
# but it will trade off speed.
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes "$NUM_GPUS" \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/dpo.py \
    "$CONFIG_FILE"
