import os
import sys
import glob
import yaml
import argparse
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFile, features

import torch
import pytorch_lightning as pl

# -------------------------------------------------------------------------
# Paths / env
# -------------------------------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

# Allows PIL to load truncated JPEGs instead of erroring out
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------------------------------------------------------------
# Import library modules
# -------------------------------------------------------------------------

from src.constants import IMAGENET_MEAN, IMAGENET_STD
from src.models import FullImageModule
from src.datasets import FullImageDataset
from src.collators import ValFullImagePatchCollator
from src.metrics import compute_binary_auc, safe_prf

# -------------------------------------------------------------------------
# DataModule
# -------------------------------------------------------------------------

class TestDataModule(pl.LightningDataModule):
    """DataModule for testing/inference."""
    
    def __init__(
        self,
        test_path: str,
        batch_size: int,
        num_workers: int,
        patch_size: int,
        max_patches: int,
        root_dir: str = ""
    ):
        super().__init__()
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.collate_fn = ValFullImagePatchCollator(
            patch_size=patch_size,
            max_patches=max_patches,
            normalize=True,
            return_benchmark=True
        )

    def setup(self, stage: str = None):
        self.test_dataset = FullImageDataset(
            self.test_path,
            predict_model=False,
            return_benchmark=True,
            root_dir=self.root_dir
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

# -------------------------------------------------------------------------
# Config loading helpers
# -------------------------------------------------------------------------

def resolve_run_dir_and_ckpt(model_arg: str) -> Tuple[str, str]:
    """Resolve run directory and checkpoint path from either a ckpt or a run directory."""
    if os.path.isdir(model_arg):
        run_dir = model_arg

        # For evaluation, prefer best checkpoint over last (opposite of resuming training)
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
    """Load args.yaml if present. Falls back to empty dict if missing."""
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

        # Ignore fallback rows for metrics by default (still reported via counts)
        g_eval = g[g["is_fallback"] == False].copy()  # noqa: E712

        n_eval = len(g_eval)
        if n_eval == 0:
            rows.append({
                "benchmark": bm_name,
                "n": int(n),
                "n_eval": 0,
                "fallback_count": int((g["is_fallback"] == True).sum()),  # noqa: E712
                "accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "auc": float("nan"),
            })
            continue

        y_true = g_eval["label"].to_numpy()
        y_pred = g_eval["pred"].to_numpy()
        scores = g_eval["prob_1"].to_numpy() if "prob_1" in g_eval.columns else None

        # If labels are missing (-1), metrics become NaN
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
            "n_eval": int(n_eval),
            "fallback_count": int((g["is_fallback"] == True).sum()),  # noqa: E712
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
        })

    out = pd.DataFrame(rows)
    return out.sort_values("benchmark").reset_index(drop=True)


def print_metrics_table(df: pd.DataFrame, title: str):
    """Pretty-print a metrics table for a subset."""
    if df.empty:
        print(f"\n[EMPTY] No data found for {title}")
        return

    cols = ["benchmark", "n", "n_eval", "fallback_count", "accuracy", "precision", "recall", "f1", "auc"]
    df_show = df[cols].copy()

    for c in ["accuracy", "precision", "recall", "f1", "auc"]:
        df_show[c] = df_show[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")

    print(f"\n{'=' * 100}")
    print(f"{title:^100}")
    print(f"{'=' * 100}")
    print(df_show.to_string(index=False))
    print("=" * 100)


def build_groups(bench_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build groups like ID/OOD + C/R/RC splits (keeps unknown benchmarks out of those groups)."""
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

    known = set(
        ["ID", "OOD"] + c_id + c_ood + r_id + r_ood + rc_id + rc_ood
    )
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
    p = argparse.ArgumentParser()

    p.add_argument("--test_path", type=str, required=True, help="Path to test CSV (must have image_path; label/benchmark optional)")
    p.add_argument("--model", "--model_path", dest="model_path", type=str, required=True, help="Path to ckpt OR run directory")
    p.add_argument("--root_dir", type=str, default="", help="Optional prefix for relative image paths")

    p.add_argument("--device", type=int, default=0, help="GPU id (set -1 for CPU)")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size")
    p.add_argument("--num_workers", type=int, default=8, help="Num DataLoader workers")

    p.add_argument("--output_dir", type=str, default="eval_outputs", help="Where to write outputs")
    p.add_argument("--pred_csv", type=str, default="predictions.csv", help="Predictions CSV filename inside output_dir")
    p.add_argument("--metrics_csv", type=str, default="metrics_by_benchmark.csv", help="Metrics CSV filename inside output_dir")
    p.add_argument("--plot_png", type=str, default="benchmark_comparison.png", help="Plot filename inside output_dir")

    p.add_argument("--no_report", action="store_true", help="Skip metrics/tables/plots")
    p.add_argument("--show", action="store_true", help="Show plot window")
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(42)

    print("[INFO] Pillow WEBP support:", features.check("webp"))

    run_dir, ckpt_path = resolve_run_dir_and_ckpt(args.model_path)
    config = load_args_yaml(run_dir)

    patch_size = int(config.get("patch_size", 224))
    max_patches = int(config.get("max_patches", 32))

    os.makedirs(args.output_dir, exist_ok=True)
    pred_csv_path = os.path.join(args.output_dir, args.pred_csv)
    metrics_csv_path = os.path.join(args.output_dir, args.metrics_csv)
    plot_png_path = os.path.join(args.output_dir, args.plot_png)

    # Load model from checkpoint
    model = FullImageModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()

    dm = TestDataModule(
        test_path=args.test_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=patch_size,
        max_patches=max_patches,
        root_dir=args.root_dir,
    )
    dm.setup("test")

    use_gpu = (args.device >= 0) and torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    devices = [args.device] if use_gpu else 1

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    pred_batches = trainer.predict(model, dataloaders=dm.test_dataloader())

    # Collect predictions
    rows = []
    for pb in pred_batches:
        paths = pb["image_paths"]
        abs_paths = pb["abs_paths"]
        bms = pb["benchmarks"]
        labels = pb["labels"].numpy()

        # Keys returned by your predict_step()
        preds = pb["preds"].numpy()
        conf = pb["conf"].numpy()
        p0 = pb["prob0"].numpy()
        p1 = pb["prob1"].numpy()

        fb = pb["is_fallback"]
        errs = pb["load_errors"]

        for i in range(len(paths)):
            rows.append({
                "image_path": paths[i],
                "abs_path": abs_paths[i],
                "benchmark": bms[i] if bms[i] is not None else "",
                "label": int(labels[i]),

                "pred": int(preds[i]),
                "confidence": float(conf[i]),
                "prob_0": float(p0[i]),
                "prob_1": float(p1[i]),

                "is_fallback": bool(fb[i]),
                "load_error": str(errs[i]),
            })


    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(pred_csv_path, index=False)

    print(f"[OK] Saved predictions CSV: {os.path.abspath(pred_csv_path)}")

    # Fallback diagnostics (this is what will expose your newtral/free-viral issue quickly)
    if "benchmark" in pred_df.columns and pred_df["benchmark"].astype(str).str.len().gt(0).any():
        fb_stats = (
            pred_df.groupby("benchmark")["is_fallback"]
            .agg(["count", "sum"])
            .rename(columns={"count": "n", "sum": "fallback_count"})
            .reset_index()
        )
        fb_stats["fallback_rate"] = fb_stats["fallback_count"] / fb_stats["n"]
        worst = fb_stats.sort_values("fallback_rate", ascending=False).head(10)
        print("\n[DIAG] Worst benchmarks by fallback_rate:")
        print(worst.to_string(index=False))

        # Extra warning for benchmarks that are basically all fallback
        nearly_all = fb_stats[fb_stats["fallback_rate"] > 0.95]
        if not nearly_all.empty:
            print("\n[WARN] These benchmarks are ~all fallback (likely path/format issue):")
            print(nearly_all.sort_values("fallback_rate", ascending=False).to_string(index=False))

    if args.no_report:
        return

    # Metrics + plots
    if "benchmark" not in pred_df.columns or pred_df["benchmark"].astype(str).str.len().eq(0).all():
        print("[INFO] No benchmark column present/filled. Skipping report.")
        return

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

    # Show new benchmarks / any extra ones
    print_metrics_table(groups["OTHER"], "OTHER (including new benchmarks)")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharey=True)

    plot_group(axes[0, 0], groups["C_ID"], "C_ID Benchmarks")
    plot_group(axes[0, 1], groups["C_OOD"], "C_OOD Benchmarks")

    plot_group(axes[1, 0], groups["R_ID"], "R_ID Benchmarks")
    plot_group(axes[1, 1], groups["R_OOD"], "R_OOD Benchmarks")

    plot_group(axes[2, 0], groups["RC_ID"], "RC_ID Benchmarks")
    plot_group(axes[2, 1], groups["RC_OOD"], "RC_OOD Benchmarks")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Metric", loc="upper center", ncol=3)
    fig.suptitle("Benchmark Comparison (Accuracy, F1, AUC)", fontsize=16, y=1.02)

    plt.tight_layout()
    plt.savefig(plot_png_path, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved plot PNG: {os.path.abspath(plot_png_path)}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
