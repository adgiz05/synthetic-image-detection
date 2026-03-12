import os
import argparse
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Metrics helpers (no sklearn)
# ----------------------------

def compute_binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC for binary labels using only numpy. Returns NaN if one class is missing."""
    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores).astype(float).ravel()

    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(-scores)  # descending
    sorted_labels = labels[order]

    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)

    tpr = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)

    # Prepend (0,0)
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))

    return float(np.trapz(tpr, fpr))


def safe_prf(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Compute precision/recall/f1 for positive class=1 with safe zero-division handling."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (pd.notna(precision) and pd.notna(recall) and (precision + recall) > 0) else float("nan")
    return precision, recall, f1


# ----------------------------
# IO + merging
# ----------------------------

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Pick the first existing column from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_and_merge(pred_path: str, test_path: str, key: str) -> pd.DataFrame:
    """Load prediction CSV and test CSV, merge to bring benchmark (and label if needed)."""
    pred_df = pd.read_csv(pred_path)
    test_df = pd.read_csv(test_path)

    if key not in pred_df.columns:
        raise ValueError(f"Key column '{key}' not found in predictions CSV.")
    if key not in test_df.columns:
        raise ValueError(f"Key column '{key}' not found in test CSV.")

    if "benchmark" not in test_df.columns:
        raise ValueError("Test CSV must contain a 'benchmark' column to rescue it.")

    # Reduce test to only necessary columns (avoid duplicate columns after merge)
    keep_cols = [key, "benchmark"]
    if "label" in test_df.columns:
        keep_cols.append("label")
    test_small = test_df[keep_cols].copy()

    # If test has duplicates in key, keep the first occurrence
    if test_small[key].duplicated().any():
        test_small = test_small.drop_duplicates(subset=[key], keep="first")

    merged = pred_df.merge(test_small, on=key, how="left", suffixes=("", "_test"))

    if merged["benchmark"].isna().any():
        missing = int(merged["benchmark"].isna().sum())
        print(f"[WARN] {missing} rows have missing benchmark after merge (no match in test CSV).")

    return merged


# ----------------------------
# Metrics per benchmark
# ----------------------------

def compute_metrics_by_benchmark(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy/precision/recall/f1/auc for each benchmark."""
    # Resolve columns from common naming variants
    col_label_pred = pick_col(merged, ["label"])  # from preds or merged
    col_label_test = pick_col(merged, ["label_test"])  # if suffix happened (rare here)
    col_pred = pick_col(merged, ["pred", "preds"])
    col_prob1 = pick_col(merged, ["prob_1", "prob1", "p1", "score", "probs_1"])

    if col_pred is None:
        raise ValueError("Predictions CSV must contain 'pred' (or 'preds') column.")

    # Choose label source:
    # - if merged has label and it's not all -1, use it
    # - else if test label exists, use it
    y_col = None
    if col_label_pred is not None:
        lab = merged[col_label_pred].values
        if not np.all(pd.isna(lab)) and not np.all(lab == -1):
            y_col = col_label_pred
    if y_col is None and col_label_test is not None:
        y_col = col_label_test

    # If no labels, we can still produce counts per benchmark, but metrics will be NaN
    has_labels = y_col is not None

    rows = []
    for bm, g in merged.groupby("benchmark", dropna=False):
        bm_name = "NaN" if pd.isna(bm) else str(bm)

        n = len(g)
        y_pred = g[col_pred].to_numpy()

        if has_labels:
            y_true = g[y_col].to_numpy()
            # Ignore unlabeled rows if they use -1
            mask_labeled = (y_true != -1) & (~pd.isna(y_true))
            y_true = y_true[mask_labeled]
            y_pred_l = y_pred[mask_labeled]

            if y_true.size == 0:
                acc = prec = rec = f1 = auc = float("nan")
                n_lab = 0
            else:
                acc = float((y_pred_l == y_true).mean())
                prec, rec, f1 = safe_prf(y_true, y_pred_l)

                if col_prob1 is not None:
                    scores = g.loc[mask_labeled, col_prob1].to_numpy()
                    auc = compute_binary_auc(y_true, scores)
                else:
                    auc = float("nan")
                n_lab = int(y_true.size)
        else:
            acc = prec = rec = f1 = auc = float("nan")
            n_lab = 0

        rows.append({
            "benchmark": bm_name,
            "n": int(n),
            "n_labeled": int(n_lab),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
        })

    out = pd.DataFrame(rows)

    # Sort benchmarks in a human-friendly way: ID/OOD first, then others alphabetically
    def sort_key(b: str) -> Tuple[int, str]:
        if b == "ID":
            return (0, b)
        if b == "OOD":
            return (1, b)
        return (2, b)

    out = out.sort_values(by="benchmark", key=lambda s: s.map(lambda x: sort_key(str(x)))).reset_index(drop=True)
    return out


# ----------------------------
# Reporting (tables + plots)
# ----------------------------

def print_metrics_table(df: pd.DataFrame, title: str):
    """Pretty-print a metrics table for a subset of benchmarks."""
    if df.empty:
        print(f"\n❌ No data found for {title}")
        return

    cols = ["benchmark", "n", "n_labeled", "accuracy", "precision", "recall", "f1", "auc"]
    df_show = df[cols].copy()

    for c in ["accuracy", "precision", "recall", "f1", "auc"]:
        df_show[c] = df_show[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")

    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print(f"{'=' * 80}")
    print(df_show.to_string(index=False))
    print("=" * 80)


def plot_group(ax, df_group: pd.DataFrame, title: str):
    """Plot accuracy/f1/auc across benchmarks for a given group."""
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


def build_groups(bench_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build the benchmark groups exactly like your snippet."""
    groups = {}
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
    return groups


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_path", type=str, required=True, help="Predictions CSV (must include image_path + pred + prob_1/prob1 for AUC)")
    p.add_argument("--test_path", type=str, required=True, help="Test CSV (must include image_path + benchmark; label optional)")
    p.add_argument("--key", type=str, default="image_path", help="Merge key column (default: image_path)")

    p.add_argument("--out_metrics_csv", type=str, default="metrics_by_benchmark.csv", help="Output metrics CSV")
    p.add_argument("--out_plot_png", type=str, default="benchmark_comparison.png", help="Output plot PNG")
    p.add_argument("--show", action="store_true", help="Show the plot window")
    return p.parse_args()


def main():
    args = parse_args()

    merged = load_and_merge(args.pred_path, args.test_path, key=args.key)

    bench_df = compute_metrics_by_benchmark(merged)
    bench_df.to_csv(args.out_metrics_csv, index=False)
    print(f"[OK] Saved metrics CSV: {os.path.abspath(args.out_metrics_csv)}")

    groups = build_groups(bench_df)

    # Print tables like your snippet
    print_metrics_table(groups["ID"], "GLOBAL - ID")
    print_metrics_table(groups["OOD"], "GLOBAL - OOD")

    print_metrics_table(groups["C_ID"], "C_ID")
    print_metrics_table(groups["C_OOD"], "C_OOD")

    print_metrics_table(groups["R_ID"], "R_ID")
    print_metrics_table(groups["R_OOD"], "R_OOD")

    print_metrics_table(groups["RC_ID"], "RC_ID")
    print_metrics_table(groups["RC_OOD"], "RC_OOD")

    # Plot 3x2 like your snippet
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
    plt.savefig(args.out_plot_png, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved plot PNG: {os.path.abspath(args.out_plot_png)}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
