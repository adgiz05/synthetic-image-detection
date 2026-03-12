#!/bin/bash
# Script to test the MultiScaleTubeDataModule
#
# This tests the data loading pipeline with multi-scale tubes and augmentations.
# Make sure you have the CSV files with columns: image_path, label, [model]

cd "$(dirname "$0")/.."

python scripts/contrastive_train.py \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --num_tubes 8 \
    --scales 64 128 256 \
    --target_size 128 \
    --num_views 2 \
    --batch_size 4 \
    --num_workers 4
