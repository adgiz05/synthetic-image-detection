"""Metrics computation utilities (no sklearn dependency)."""

from typing import Tuple

import numpy as np
import pandas as pd


def compute_binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute binary ROC AUC using only numpy.

    Args:
        labels: array of shape [N], values 0/1
        scores: array of shape [N], predicted scores (probability for class 1)

    Returns:
        auc: float in [0,1] or NaN if only one class present.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    if labels.ndim != 1:
        labels = labels.ravel()
    if scores.ndim != 1:
        scores = scores.ravel()

    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Sort by descending score
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    # Cumulative TP/FP
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)

    tpr = tp / n_pos
    fpr = fp / n_neg

    # Prepend (0,0)
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))

    auc = np.trapz(tpr, fpr)
    return float(auc)


def safe_prf(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute precision/recall/f1 for positive class=1 with safe zero-division handling.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        
    Returns:
        (precision, recall, f1) tuple, with NaN for undefined values
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (
        pd.notna(precision) and pd.notna(recall) and (precision + recall) > 0
    ) else float("nan")
    
    return precision, recall, f1
