#!/bin/bash

# VFM baseline training script
# Based on paper specs: frozen backbone, linear head, AdamW lr=1e-3, bs=128, 2 epochs

cd /home/adrian/synthetic-image-detection

# Activate conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate synthetic-generation

# HuggingFace authentication (uncomment and set your token if needed)
# export HF_TOKEN="your_hf_token_here"

echo "Using GPU: $GPU_ID (PCI_BUS_ID order)"
echo "==========================================="

export HF_HOME="/opt/huggingface/cache"

# Run training
python scripts/vfm/train.py \
    --backbone facebook/dinov3-vit7b16-pretrain-lvd1689m \
    --train_csv data/dataset_v5/train.csv \
    --val_csv data/benchmarks/chameleon/annotations.csv \
    --data_root /home/adrian/synthetic-image-detection \
    --batch_size 64 \
    --accumulate_grad_batches 2 \
    --epochs 2 \
    --learning_rate 1e-3 \
    --num_workers 8 \
    --output_dir runs/vfm/dinov3_vit7b16 \
    --precision 16-mixed \
    --accelerator gpu \
    --device 5 \
    --seed 42 \
    --wandb_project synthetic-detection-vfm \
    --wandb_name dinov3_qlora_baseline \
    --use_qlora
