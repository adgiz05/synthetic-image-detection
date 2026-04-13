"""
Test script for Phase 2 tube-based binary classification.

Loads a Phase 2 checkpoint and evaluates on a test set, computing metrics
per benchmark and generating comparison plots.

Usage:
    python scripts/test_phase2.py \
        --test_path   data/test.csv \
        --model       runs/phase2/run_XXX/best.ckpt \
        --output_dir  eval_outputs/phase2 \
        --device      0
"""

import os
import sys
import glob
import yaml
import argparse
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

# -------------------------------------------------------------------------
# Paths / env
# -------------------------------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------------------------------------------------------------
# Import library modules
# -------------------------------------------------------------------------

from src.datasets import MultiScaleTubeDataset
from src.collators import FastMultiScaleTubeCollator, MultiScaleTubeCollator
from src.tube_model import TubeContrastiveModule
from src.metrics import compute_binary_auc, safe_prf

# -------------------------------------------------------------------------
# DataModule
# -------------------------------------------------------------------------

class TestDataModule(pl.LightningDataModule):
    """DataModule for testing Phase 2 tube model."""

    def __init__(
        self,
        test_path: str,
        batch_size: int,
        num_workers: int,
        # Tube config (loaded from checkpoint args.yaml)
        max_tubes: int = 16,
        min_tubes: int = 4,
        overlap_ratio: float = 0.25,
        scales: List[int] = None,
        target_size: int = 128,
        min_image_size: int = 256,
        max_image_size: int = 2048,
        use_fast_collator: bool = True,
        root_dir: str = "",
    ):
        super().__init__()
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir

        scales = scales or [64, 128, 256]

        # For testing: only 1 view (no augmentations)
        if use_fast_collator:
            self.collator = FastMultiScaleTubeCollator(
                max_tubes=max_tubes,
                min_tubes=min_tubes,
                overlap_ratio=overlap_ratio,
                scales=scales,
                target_size=target_size,
                num_views=1,
                min_image_size=min_image_size,
                max_image_size=max_image_size,
            )
        else:
            self.collator = MultiScaleTubeCollator(
                max_tubes=max_tubes,
                min_tubes=min_tubes,
                overlap_ratio=overlap_ratio,
                scales=scales,
                target_size=target_size,
                num_views=1,
                normalize=True,
                min_image_size=min_image_size,
                max_image_size=max_image_size,
            )

    def setup(self, stage: str = None):
        self.test_dataset = MultiScaleTubeDataset(
            data_path=self.test_path,
            predict_model=False,
            return_benchmark=True,
            root_dir=self.root_dir,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collator,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

# -------------------------------------------------------------------------
# Config loading helpers
# -------------------------------------------------------------------------

def resolve_run_dir_and_ckpt(model_arg: str) -> Tuple[str, str]:
    """Resolve run directory and checkpoint path."""
    if os.path.isdir(model_arg):
        run_dir = model_arg

        best_ckpts = glob.glob(os.path.join(run_dir, "*best*.ckpt")) + glob.glob(os.path.join(run_dir, "best-*.ckpt"))
        if best_ckpts:
            ckpt_path = max(best_ckpts, key=os.path.getmtime)
            return run_dir, ckpt_path

        last_ckpt = os.path.join(run_dir, "last.ckpt")
        if os.path.isfile(last_ckpt):
            return run_dir, last_ckpt

        any_ckpts = glob.glob(os.path.join(run_dir, "*.ckpt"))
        if any_ckpts:
            ckpt_path = max(any_ckpts, key=os.path.getmtime)
            return run_dir, ckpt_path

        raise FileNotFoundError(f"No .ckpt found inside run directory: {run_dir}")

    if not os.path.isfile(model_arg) or not model_arg.endswith(".ckpt"):
        raise FileNotFoundError(f"--model must be a .ckpt file or a directory. Got: {model_arg}")

    ckpt_path = model_arg
    run_dir = os.path.dirname(ckpt_path)
    return run_dir, ckpt_path


def load_args_yaml(run_dir: str) -> Dict[str, Any]:
    """Load args.yaml if present."""
    config_path = os.path.join(run_dir, "args.yaml")
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}

# -------------------------------------------------------------------------
# Reporting: metrics per benchmark + groups + plotting
# -------------------------------------------------------------------------

def compute_metrics_by_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy/precision/recall/f1/auc per benchmark."""
    if "benchmark" not in df.columns:
        df = df.copy()
        df["benchmark"] = "ALL"

    rows = []
    for bm, g in df.groupby("benchmark", dropna=False):
        bm_name = "NaN" if pd.isna(bm) else str(bm)
        n = len(g)

        y_true = g["label"].to_numpy()
        y_pred = g["pred"].to_numpy()
        scores = g["prob_1"].to_numpy() if "prob_1" in g.columns else None

        mask_labeled = (y_true != -1) & (~pd.isna(y_true))
        y_true_l = y_true[mask_labeled]
        y_pred_l = y_pred[mask_labeled]
        scores_l = scores[mask_labeled] if scores is not None else None

        if y_true_l.size == 0:
            acc = prec = rec = f1 = auc = float("nan")
        else:
            acc = float((y_pred_l == y_true_l).mean())
            prec, rec, f1 = safe_prf(y_true_l, y_pred_l)
            auc = compute_binary_auc(y_true_l, scores_l) if scores_l is not None else float("nan")

        rows.append({
            "benchmark": bm_name,
            "n": int(n),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
        })

    out = pd.DataFrame(rows)
    return out.sort_values("benchmark").reset_index(drop=True)


def print_metrics_table(df: pd.DataFrame, title: str):
    """Pretty-print a metrics table."""
    if df.empty:
        print(f"\n[EMPTY] No data found for {title}")
        return

    cols = ["benchmark", "n", "accuracy", "precision", "recall", "f1", "auc"]
    df_show = df[cols].copy()

    for c in ["accuracy", "precision", "recall", "f1", "auc"]:
        df_show[c] = df_show[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")

    print(f"\n{'=' * 90}")
    print(f"{title:^90}")
    print(f"{'=' * 90}")
    print(df_show.to_string(index=False))
    print("=" * 90)


def build_groups(bench_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build groups like ID/OOD + C/R/RC splits."""
    groups: Dict[str, pd.DataFrame] = {}
    groups["ID"] = bench_df[bench_df["benchmark"] == "ID"]
    groups["OOD"] = bench_df[bench_df["benchmark"] == "OOD"]

    c_id = [f"C{i}_ID" for i in range(6)]
    c_ood = [f"C{i}_OOD" for i in range(6)]
    r_id = [f"R{i}_ID" for i in range(6)]
    r_ood = [f"R{i}_OOD" for i in range(6)]
    rc_id = [f"RC{i}_ID" for i in range(6)]
    rc_ood = [f"RC{i}_OOD" for i in range(6)]

    groups["C_ID"] = bench_df[bench_df["benchmark"].isin(c_id)]
    groups["C_OOD"] = bench_df[bench_df["benchmark"].isin(c_ood)]
    groups["R_ID"] = bench_df[bench_df["benchmark"].isin(r_id)]
    groups["R_OOD"] = bench_df[bench_df["benchmark"].isin(r_ood)]
    groups["RC_ID"] = bench_df[bench_df["benchmark"].isin(rc_id)]
    groups["RC_OOD"] = bench_df[bench_df["benchmark"].isin(rc_ood)]

    known = set(["ID", "OOD"] + c_id + c_ood + r_id + r_ood + rc_id + rc_ood)
    groups["OTHER"] = bench_df[~bench_df["benchmark"].isin(known)]

    return groups


def plot_group(ax, df_group: pd.DataFrame, title: str):
    """Plot accuracy/f1/auc across benchmarks for a group."""
    if df_group.empty:
        ax.set_title(f"{title} (empty)")
        ax.axis("off")
        return

    metrics = ["accuracy", "f1", "auc"]
    df_plot = df_group.set_index("benchmark")[metrics]

    for metric in metrics:
        ax.plot(df_plot.index, df_plot[metric], marker="o", label=metric)

    ax.set_title(title)
    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", rotation=45)

# -------------------------------------------------------------------------
# CLI + main
# -------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Test Phase 2 tube-based classifier")

    p.add_argument("--test_path", type=str, required=True, help="Path to test CSV")
    p.add_argument("--model", "--model_path", dest="model_path", type=str, required=True,
                   help="Path to ckpt OR run directory")
    p.add_argument("--root_dir", type=str, default="", help="Optional prefix for relative image paths")

    p.add_argument("--device", type=int, default=0, help="GPU id (set -1 for CPU)")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--num_workers", type=int, default=8, help="Num DataLoader workers")

    p.add_argument("--output_dir", type=str, default="eval_outputs/phase2", help="Where to write outputs")
    p.add_argument("--pred_csv", type=str, default="predictions.csv", help="Predictions CSV filename")
    p.add_argument("--metrics_csv", type=str, default="metrics_by_benchmark.csv", help="Metrics CSV filename")
    p.add_argument("--plot_png", type=str, default="benchmark_comparison.png", help="Plot filename")

    p.add_argument("--no_report", action="store_true", help="Skip metrics/tables/plots")
    p.add_argument("--show", action="store_true", help="Show plot window")

    return p.parse_args()


@torch.no_grad()
def run_inference(model, dataloader, device):
    """Run inference and collect predictions."""
    model.eval()
    model.to(device)

    all_results = []

    for batch in tqdm(dataloader, desc="Inference"):
        # Get input tensors
        if "tubes_rgb" in batch:
            tubes = batch["tubes_rgb"].to(device)
            wav = None
        else:
            tubes = batch["tubes"].to(device)
            wav = batch["tubes_wavelet"].to(device)

        tube_mask = batch.get("tube_mask", None)
        if tube_mask is not None:
            tube_mask = tube_mask.to(device)

        labels = batch["labels"]
        benchmarks = batch.get("benchmarks", [None] * len(labels))
        image_paths = batch.get("image_paths", [""] * len(labels))

        # Forward pass
        out = model(tubes, wav, tube_mask=tube_mask, view_idx=0)
        logits = out["logits_auth"]  # [B, 2]
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        # Collect results
        for i in range(len(labels)):
            all_results.append({
                "image_path": image_paths[i] if i < len(image_paths) else "",
                "benchmark": benchmarks[i] if i < len(benchmarks) and benchmarks[i] is not None else "",
                "label": int(labels[i].item()),
                "pred": int(preds[i].item()),
                "prob_0": float(probs[i, 0].item()),
                "prob_1": float(probs[i, 1].item()),
                "confidence": float(probs[i, preds[i]].item()),
            })

    return pd.DataFrame(all_results)


def main():
    args = parse_args()
    pl.seed_everything(42)

    run_dir, ckpt_path = resolve_run_dir_and_ckpt(args.model_path)
    config = load_args_yaml(run_dir)

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    print(f"[INFO] Config: {run_dir}/args.yaml")

    # Load checkpoint to get saved hyperparameters (more reliable than args.yaml)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_hparams = checkpoint.get("hyper_parameters", {})

    # Extract tube configuration: prefer checkpoint hparams, fallback to args.yaml
    max_tubes = config.get("max_tubes", config.get("num_tubes", 16))
    min_tubes = config.get("min_tubes", 4)
    overlap_ratio = config.get("overlap_ratio", 0.25)
    scales = config.get("scales", [64, 128, 256])
    # IMPORTANT: Use target_size from checkpoint (model's actual value), not args.yaml
    target_size = ckpt_hparams.get("target_size", config.get("target_size", 96))
    min_image_size = config.get("min_image_size", 256)
    max_image_size = config.get("max_image_size", 2048)
    use_fast_collator = config.get("use_fast_collator", True)

    print(f"[INFO] Tube config: max={max_tubes}, min={min_tubes}, scales={scales}, target={target_size}")
    print(f"[INFO] Fast collator: {use_fast_collator}")

    os.makedirs(args.output_dir, exist_ok=True)
    pred_csv_path = os.path.join(args.output_dir, args.pred_csv)
    metrics_csv_path = os.path.join(args.output_dir, args.metrics_csv)
    plot_png_path = os.path.join(args.output_dir, args.plot_png)

    # Load model - handle torch.compile checkpoints with "_orig_mod." prefix
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip "_orig_mod." prefix if present (from torch.compile)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("._orig_mod.", ".")
        cleaned_state_dict[new_key] = v

    if any("_orig_mod" in k for k in state_dict.keys()):
        print("[INFO] Detected torch.compile checkpoint, stripping _orig_mod prefix")

    model = TubeContrastiveModule.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        strict=False,
    )

    # Manually load the cleaned state_dict
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        # Filter out expected missing keys (buffers that get re-created)
        real_missing = [k for k in missing if not any(x in k for x in ['num_batches_tracked'])]
        if real_missing:
            print(f"[WARN] Missing keys: {real_missing[:5]}...")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:5]}...")

    model.eval()

    # Setup data
    dm = TestDataModule(
        test_path=args.test_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_tubes=max_tubes,
        min_tubes=min_tubes,
        overlap_ratio=overlap_ratio,
        scales=scales,
        target_size=target_size,
        min_image_size=min_image_size,
        max_image_size=max_image_size,
        use_fast_collator=use_fast_collator,
        root_dir=args.root_dir,
    )
    dm.setup("test")

    # Run inference
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running inference on {device}")

    pred_df = run_inference(model, dm.test_dataloader(), device)
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"[OK] Saved predictions CSV: {os.path.abspath(pred_csv_path)}")

    # Quick summary
    n_total = len(pred_df)
    n_correct = (pred_df["pred"] == pred_df["label"]).sum()
    acc = n_correct / n_total if n_total > 0 else 0
    print(f"[INFO] Overall accuracy: {acc:.4f} ({n_correct}/{n_total})")

    if args.no_report:
        return

    # Metrics + plots
    if "benchmark" not in pred_df.columns or pred_df["benchmark"].astype(str).str.len().eq(0).all():
        print("[INFO] No benchmark column present/filled. Computing overall metrics only.")
        bench_df = compute_metrics_by_benchmark(pred_df)
    else:
        bench_df = compute_metrics_by_benchmark(pred_df)

    bench_df.to_csv(metrics_csv_path, index=False)
    print(f"[OK] Saved metrics CSV: {os.path.abspath(metrics_csv_path)}")

    groups = build_groups(bench_df)

    print_metrics_table(groups["ID"], "GLOBAL - ID")
    print_metrics_table(groups["OOD"], "GLOBAL - OOD")
    print_metrics_table(groups["C_ID"], "C_ID")
    print_metrics_table(groups["C_OOD"], "C_OOD")
    print_metrics_table(groups["R_ID"], "R_ID")
    print_metrics_table(groups["R_OOD"], "R_OOD")
    print_metrics_table(groups["RC_ID"], "RC_ID")
    print_metrics_table(groups["RC_OOD"], "RC_OOD")
    print_metrics_table(groups["OTHER"], "OTHER (including new benchmarks)")

    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharey=True)

    plot_group(axes[0, 0], groups["C_ID"], "C_ID Benchmarks")
    plot_group(axes[0, 1], groups["C_OOD"], "C_OOD Benchmarks")
    plot_group(axes[1, 0], groups["R_ID"], "R_ID Benchmarks")
    plot_group(axes[1, 1], groups["R_OOD"], "R_OOD Benchmarks")
    plot_group(axes[2, 0], groups["RC_ID"], "RC_ID Benchmarks")
    plot_group(axes[2, 1], groups["RC_OOD"], "RC_OOD Benchmarks")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Metric", loc="upper center", ncol=3)
    fig.suptitle("Phase 2 Benchmark Comparison (Accuracy, F1, AUC)", fontsize=16, y=1.02)

    plt.tight_layout()
    plt.savefig(plot_png_path, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved plot PNG: {os.path.abspath(plot_png_path)}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
