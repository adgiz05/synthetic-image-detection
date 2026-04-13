#!/bin/bash

# VFM baseline testing script
# Computes: acc, balanced_acc, auc, f1, recall, precision

# set -euo pipefail

cd /home/adrian/synthetic-image-detection

# Activate conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate synthetic-generation

# -----------------------------
# Config (override via env vars)
# -----------------------------
GPU_ID=${GPU_ID:-5}
CHECKPOINT=${CHECKPOINT:-runs/vfm/dinov3_vit7b16/last.ckpt}
TEST_CSV=${TEST_CSV:-data/benchmarks/lawwwing/annotations.csv}
DATA_ROOT=${DATA_ROOT:-/home/adrian/synthetic-image-detection}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-8}
OUTPUT_PREDICTIONS_CSV=${OUTPUT_PREDICTIONS_CSV:-runs/vfm/dinov3_vit7b16/lawwwing_benchmark/predictions.csv}
OUTPUT_METRICS_CSV=${OUTPUT_METRICS_CSV:-runs/vfm/dinov3_vit7b16/lawwwing_benchmark/metrics.csv}

export HF_HOME="/opt/huggingface/cache"

echo "Using GPU: ${GPU_ID} (PCI_BUS_ID order)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Test CSV: ${TEST_CSV}"
echo "==========================================="

CMD=(
  python scripts/vfm/test.py
  --checkpoint "${CHECKPOINT}"
  --test_csv "${TEST_CSV}"
  --data_root "${DATA_ROOT}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --device "${GPU_ID}"
  --output_metrics_csv "${OUTPUT_METRICS_CSV}"
)

if [[ -n "${OUTPUT_PREDICTIONS_CSV}" ]]; then
  CMD+=(--output_predictions_csv "${OUTPUT_PREDICTIONS_CSV}")
fi

"${CMD[@]}"
