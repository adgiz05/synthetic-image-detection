"""
VFM baseline testing script.

Evaluates a trained VFM checkpoint on a CSV split and reports:
- acc
- balanced_acc
- auc
- f1
- recall
- precision
"""
import os
os.environ["HF_HOME"] = "/opt/huggingface/cache"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from datamodule import VFMDataModule
from train import VFMClassifier


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute binary classification metrics requested by the user."""
    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
    }

    unique_labels = np.unique(y_true)
    if unique_labels.size < 2:
        warnings.warn(
            "AUC is undefined because the evaluation CSV has a single class in y_true. Returning NaN.",
            RuntimeWarning,
        )
        metrics["auc"] = float("nan")
    else:
        metrics["auc"] = roc_auc_score(y_true, y_prob)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Test VFM baseline checkpoint")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Lightning .ckpt file")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test CSV")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/adrian/synthetic-image-detection",
        help="Root directory for image paths",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Physical GPU id to use with CUDA_VISIBLE_DEVICES (e.g. 5). If omitted, uses default device.",
    )
    parser.add_argument(
        "--output_predictions_csv",
        type=str,
        default=None,
        help="Optional path to save per-sample predictions (y_true, y_pred, y_prob)",
    )
    parser.add_argument(
        "--output_metrics_csv",
        type=str,
        default="metrics.csv",
        help="Path to save global and per-benchmark metrics CSV",
    )
    parser.add_argument(
        "--log_every_n_batches",
        type=int,
        default=20,
        help="tqdm update frequency in batches (miniters)",
    )

    args = parser.parse_args()

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        print(f"Setting CUDA_VISIBLE_DEVICES={args.device}")

    pl.seed_everything(42)

    # Load model from checkpoint; hyperparameters are restored automatically.
    print(f"Loading checkpoint: {args.checkpoint}")
    # QLoRA checkpoints can include bitsandbytes quantization state entries that
    # may differ across library versions/devices. Non-strict loading keeps the
    # trained head weights while safely ignoring non-critical extra keys.
    model = VFMClassifier.load_from_checkpoint(
        args.checkpoint,
        map_location="cpu",
        strict=False,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Reuse the same preprocessing pipeline as training.
    datamodule = VFMDataModule(
        train_csv=args.test_csv,
        test_csv=args.test_csv,
        data_root=args.data_root,
        backbone_name=model.hparams.backbone_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        return_metadata=False,
    )
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    if test_loader is None:
        raise ValueError("No test dataloader available. Check --test_csv.")

    y_true = []
    y_pred = []
    y_prob = []

    print("Running inference...")
    total_batches = len(test_loader)
    log_every_n_batches = max(1, args.log_every_n_batches)
    progress_bar = tqdm(
        test_loader,
        total=total_batches,
        desc="Inference",
        dynamic_ncols=True,
        miniters=log_every_n_batches,
    )
    with torch.no_grad():
        for batch in progress_bar:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(images)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            y_true.append(labels.detach().cpu().numpy())
            y_pred.append(preds.detach().cpu().numpy())
            y_prob.append(probs.detach().cpu().numpy())

    y_true_np = np.concatenate(y_true)
    y_pred_np = np.concatenate(y_pred)
    y_prob_np = np.concatenate(y_prob)

    test_df = pd.read_csv(args.test_csv).reset_index(drop=True)
    if "benchmark" not in test_df.columns:
        raise ValueError("Missing required column in test CSV: benchmark")

    if len(test_df) != len(y_true_np):
        raise ValueError(
            "Prediction count does not match CSV rows. "
            f"Got {len(y_true_np)} predictions for {len(test_df)} rows."
        )

    eval_df = test_df[["benchmark"]].copy()
    eval_df["y_true"] = y_true_np
    eval_df["y_pred"] = y_pred_np
    eval_df["y_prob"] = y_prob_np

    metrics = compute_binary_metrics(y_true_np, y_pred_np, y_prob_np)

    metrics_rows = [
        {
            "benchmark": "global",
            "acc": metrics["acc"],
            "balanced_acc": metrics["balanced_acc"],
            "auc": metrics["auc"],
            "f1": metrics["f1"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
        }
    ]

    for benchmark_name in sorted(eval_df["benchmark"].dropna().unique()):
        bench_df = eval_df[eval_df["benchmark"] == benchmark_name]
        bench_metrics = compute_binary_metrics(
            bench_df["y_true"].to_numpy(),
            bench_df["y_pred"].to_numpy(),
            bench_df["y_prob"].to_numpy(),
        )
        metrics_rows.append(
            {
                "benchmark": str(benchmark_name),
                "acc": bench_metrics["acc"],
                "balanced_acc": bench_metrics["balanced_acc"],
                "auc": bench_metrics["auc"],
                "f1": bench_metrics["f1"],
                "recall": bench_metrics["recall"],
                "precision": bench_metrics["precision"],
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = Path(args.output_metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)

    print("\n" + "=" * 40)
    print("Evaluation Metrics")
    print("=" * 40)
    print(f"acc          : {metrics['acc']:.6f}")
    print(f"balanced_acc : {metrics['balanced_acc']:.6f}")
    auc_value = metrics["auc"]
    if np.isnan(auc_value):
        print("auc          : nan")
    else:
        print(f"auc          : {auc_value:.6f}")
    print(f"f1           : {metrics['f1']:.6f}")
    print(f"recall       : {metrics['recall']:.6f}")
    print(f"precision    : {metrics['precision']:.6f}")
    print("=" * 40)
    print(f"Saved metrics to: {metrics_path}")

    if args.output_predictions_csv is not None:
        pred_df = eval_df.copy()
        output_predictions_path = Path(args.output_predictions_csv)
        output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(output_predictions_path, index=False)
        print(f"Saved predictions to: {args.output_predictions_csv}")


if __name__ == "__main__":
    main()
