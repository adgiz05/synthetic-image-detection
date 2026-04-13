#!/usr/bin/env bash
set -eo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────────
PHASE1_CKPT="out/phase1/run_20260313_114452/last.ckpt"  # UPDATE THIS!
TRAIN_PATH="data/dataset_v5/train.csv"
VAL_PATH="data/dataset_v5/val.csv"
ROOT_DIR=""          # prepended to relative image paths (leave empty if absolute)
OUTPUT_DIR="out/phase2"
DEVICE=5

# ─── NOTE: Architecture & tube config are loaded from phase1 args.yaml ────────
# No need to specify: num_tubes, scales, target_size, encoder_dim, etc.
# They are automatically inherited from the phase 1 checkpoint directory.

# ─── Fine-tuning strategy ─────────────────────────────────────────────────────
# Option 2 (default): frozen encoders + trainable fusion
FREEZE_ENCODERS=""          # empty = use default (True)
UNFREEZE_FUSION="--unfreeze_fusion"  # train fusion MLP

# Option 1 (more conservative): uncomment to freeze fusion too
# UNFREEZE_FUSION=""

# Option 3 (full fine-tuning): uncomment to unfreeze encoders
# FREEZE_ENCODERS="--unfreeze_encoders"

# ─── Training ─────────────────────────────────────────────────────────────────
LR=1e-4              # Lower than phase 1 (5e-4)
WARMUP_STEPS=100
MAX_EPOCHS=50
BATCH_SIZE=128        # Can be larger than phase 1 (no multi-view)
ACCUM=2
NUM_WORKERS=8

# ─── Launch ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate synthetic-generation

python scripts/train_phase2.py \
    --phase1_ckpt       "$PHASE1_CKPT" \
    --train_path        "$TRAIN_PATH" \
    --val_path          "$VAL_PATH" \
    --root_dir          "$ROOT_DIR" \
    --output_dir        "$OUTPUT_DIR" \
    --device            "$DEVICE" \
    $FREEZE_ENCODERS \
    $UNFREEZE_FUSION \
    --lr                "$LR" \
    --warmup_steps      "$WARMUP_STEPS" \
    --max_epochs        "$MAX_EPOCHS" \
    --batch_size        "$BATCH_SIZE" \
    --accumulate_grad_batches "$ACCUM" \
    --num_workers       "$NUM_WORKERS" \
    --monitor           "val/auc" \
    --mode              "max" \
    --resume_if_possible
