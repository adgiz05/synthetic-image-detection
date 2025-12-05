#!/usr/bin/env bash
# ============================================================
# Script to train the FullImage ViT model
# ============================================================

# --- CONFIGURABLE VARIABLES ---

# Paths to your CSVs
TRAIN_CSV="data/dataset_v3/train.csv"
VAL_CSV="data/dataset_v3/eval_robustness_subset.csv"

# Output folder where checkpoints, logs, args.yaml, etc. will be saved
OUTPUT_DIR="out/lawwwing-full-img-cls/baseline/"

# Model and training settings
MODEL_ID="google/vit-base-patch16-224-in21k"
PATCH_SIZE=128
MAX_PATCHES=16
BATCH_SIZE=32
EPOCHS=150
LR=5e-4
ATTN_DIM=128
NUM_WORKERS=8
DEVICE=2

CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HOME=/opt/huggingface/cache python scripts/train_full_img.py \
    --train_path ${TRAIN_CSV} \
    --validation_path ${VAL_CSV} \
    --output_dir ${OUTPUT_DIR} \
    --device ${DEVICE} \
    --model_id ${MODEL_ID} \
    --patch_size ${PATCH_SIZE} \
    --max_patches ${MAX_PATCHES} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --max_epochs ${EPOCHS} \
    --attn_dim ${ATTN_DIM} \
    --num_workers ${NUM_WORKERS} \
    --accumulate_grad_batches 2 
