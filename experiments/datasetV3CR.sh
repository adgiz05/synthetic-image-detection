#!/bin/bash

HF_HOME=/opt/huggingface/cache CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/train_patched.py \
    --train_path data/dataset_v3/train.csv \
    --val_path data/dataset_v3/val.csv \
    --batch_size 256 \
    --num_workers 16 \
    --max_epochs 150 \
    --device 5 \
    --patch_size 224 \
    --experiment dataset_v3_baseline \
    --project imaginet-cls \
    --check_val_every_n_epoch 10 \
    --accumulate_grad_batches 1 \
