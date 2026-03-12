#!/bin/bash

python scripts/benchmark_full_img.py \
  --pred_path out/lawwwing-full-img-cls/run_20251211_115344/best-predictions.csv \
  --test_path data/dataset_v3/test.csv \
  --out_metrics_csv out/lawwwing-full-img-cls/run_20251211_115344/metrics_by_benchmark.csv \
  --out_plot_png out/lawwwing-full-img-cls/run_20251211_115344/bench_plot.png