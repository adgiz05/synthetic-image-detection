#!/bin/bash

HF_HOME=/opt/huggingface/cache CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/full_img_train.py \
    --train_path data/dataset_v3/train.csv \
    --val_path data/dataset_v3/eval_robustness_subset.csv \
    --batch_size 8 \
    --num_workers 8 \
    --max_epochs 10 \
    --device 5 \
    --experiment dataset_v3_baseline \
    --project imaginet-full-img-cls \
