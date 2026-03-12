#!/bin/bash

TEST_PATH="data/dataset_v5/test.csv"
MODEL_PATH="out/lawwwing-full-img-cls/run_20260302_093551/best.ckpt"
OUTPUT_DIR="out/lawwwing-full-img-cls/run_20260302_093551/"
DEVICE=5

CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HOME=/opt/huggingface/cache python scripts/test_full_img.py \
  --test_path ${TEST_PATH} \
  --model_path ${MODEL_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --device ${DEVICE} \
  --batch_size 128 \