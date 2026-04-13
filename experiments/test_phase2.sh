#!/usr/bin/env bash
set -eo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────────
TEST_PATH="data/dataset_v5/test.csv"
MODEL_PATH="out/phase2/run_20260319_160756/last.ckpt" # Update with your phase 2 run
OUTPUT_DIR="out/phase2/run_20260319_160756/"
ROOT_DIR=""          # prepended to relative image paths (leave empty if absolute)
DEVICE=5

# ─── Inference settings ───────────────────────────────────────────────────────
BATCH_SIZE=256
NUM_WORKERS=8

# ─── Launch ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate synthetic-generation

CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HOME=/opt/huggingface/cache python scripts/test_phase2.py \
    --test_path     "$TEST_PATH" \
    --model         "$MODEL_PATH" \
    --root_dir      "$ROOT_DIR" \
    --output_dir    "$OUTPUT_DIR" \
    --device        "$DEVICE" \
    --batch_size    "$BATCH_SIZE" \
    --num_workers   "$NUM_WORKERS"
