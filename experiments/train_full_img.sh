#!/usr/bin/env bash
# ============================================================
# Script to train the FullImage ViT model
# ============================================================

# --- CONFIGURABLE VARIABLES ---

# Paths to your CSVs
TRAIN_CSV="data/dataset_v5/train.csv"
VAL_CSV="data/dataset_v5/val.csv"

# Output folder where checkpoints, logs, args.yaml, etc. will be saved
OUTPUT_DIR="out/lawwwing-full-img-cls/"

# Model and training settings
MODEL_ID="google/vit-base-patch16-224-in21k"
PATCH_SIZE=96
MAX_PATCHES=24
BATCH_SIZE=64
EPOCHS=100
LR=5e-4
ATTN_DIM=128
NUM_WORKERS=8
MONITOR="val/auc"
MODE="max"
DEVICE=5

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
    --monitor ${MONITOR} \
    --mode ${MODE} \
    --accumulate_grad_batches 1 \
    --predict_model \
    --predict_transform
