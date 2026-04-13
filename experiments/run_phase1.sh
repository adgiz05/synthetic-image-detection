#!/usr/bin/env bash
set -eo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────────
TRAIN_PATH="data/dataset_v5/train.csv"
VAL_PATH="data/dataset_v5/val.csv"
ROOT_DIR=""          # prepended to relative image paths (leave empty if absolute)
OUTPUT_DIR="runs/phase1"
DEVICE=4

# ─── Adaptive tube configuration ─────────────────────────────────────────────
MAX_TUBES=8         # Maximum tubes per image (adaptive)
MIN_TUBES=2          # Minimum tubes per image
OVERLAP_RATIO=0   # Target overlap ratio (0=no overlap, 0.5=half overlap)
SCALES="64 96 128"
TARGET_SIZE=96
NUM_VIEWS=2
MIN_IMAGE_SIZE=256
MAX_IMAGE_SIZE=2048

# ─── Model architecture ───────────────────────────────────────────────────────
ENCODER_DIM=256
FUSED_DIM=256
Z_AUTH_DIM=128
Z_SRC_DIM=128
ATTN_DIM=128

# ─── Phase 1 loss weights ─────────────────────────────────────────────────────
LAMBDA_AUTH=1.0
LAMBDA_SRC_CON=0.5
LAMBDA_DECOUPLE=0.01
TEMP_AUTH=0.07
TEMP_SRC=0.07

# ─── Training ─────────────────────────────────────────────────────────────────
LR=5e-4
WARMUP_STEPS=200
MAX_EPOCHS=50
BATCH_SIZE=64
ACCUM=1          # effective batch = BATCH_SIZE × ACCUM
NUM_WORKERS=8

# ─── Launch ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate synthetic-generation

python scripts/train_phase1.py \
    --train_path        "$TRAIN_PATH" \
    --val_path          "$VAL_PATH" \
    --root_dir          "$ROOT_DIR" \
    --output_dir        "$OUTPUT_DIR" \
    --device            "$DEVICE" \
    --max_tubes         "$MAX_TUBES" \
    --min_tubes         "$MIN_TUBES" \
    --overlap_ratio     "$OVERLAP_RATIO" \
    --scales            $SCALES \
    --target_size       "$TARGET_SIZE" \
    --num_views         "$NUM_VIEWS" \
    --min_image_size    "$MIN_IMAGE_SIZE" \
    --max_image_size    "$MAX_IMAGE_SIZE" \
    --encoder_dim       "$ENCODER_DIM" \
    --fused_dim         "$FUSED_DIM" \
    --z_auth_dim        "$Z_AUTH_DIM" \
    --z_src_dim         "$Z_SRC_DIM" \
    --attn_dim          "$ATTN_DIM" \
    --lambda_auth       "$LAMBDA_AUTH" \
    --lambda_src_con    "$LAMBDA_SRC_CON" \
    --lambda_decouple   "$LAMBDA_DECOUPLE" \
    --temp_auth         "$TEMP_AUTH" \
    --temp_src          "$TEMP_SRC" \
    --lr                "$LR" \
    --warmup_steps      "$WARMUP_STEPS" \
    --max_epochs        "$MAX_EPOCHS" \
    --batch_size        "$BATCH_SIZE" \
    --accumulate_grad_batches "$ACCUM" \
    --num_workers       "$NUM_WORKERS" \
    --predict_model \
    --resume_if_possible
