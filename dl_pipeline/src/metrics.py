"""
Evaluation metrics required by the COMP9517 group-project spec.

Computes top-1 / top-5 accuracy, overall accuracy, and macro-averaged
precision / recall / F1, plus helpers for confusion matrices.

Macro averaging: treat every species equally (important for fine-grained
tasks where overall accuracy can hide weak classes).
See Opitz & Burst, "Macro F1 and Macro F1", and scikit-learn docs.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool = True,
    topk: int = 5,
) -> Dict[str, Any]:
    """
    Run the model over ``loader`` and gather labels + predictions.

    Returns
    -------
    dict with:
        y_true      : (N,) int64
        y_pred      : (N,) int64          top-1 class indices
        y_topk      : (N, K) int64        top-k class indices
        logits_time : float               pure forward-pass seconds (approx.)
        wall_time   : float               full loop wall-clock seconds
        n_samples   : int
    """
    model.eval()
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []
    y_topk: List[np.ndarray] = []

    forward_s = 0.0
    wall_t0 = time.perf_counter()

    for images, targets in tqdm(loader, desc="predict", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if use_amp and device.type == "cuda":
            with autocast("cuda", dtype=torch.float16):
                logits = model(images)
        else:
            logits = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_s += time.perf_counter() - t0

        top1 = logits.argmax(dim=1)
        k = min(topk, logits.size(1))
        topk_idx = logits.topk(k, dim=1).indices

        y_true.append(targets.cpu().numpy())
        y_pred.append(top1.cpu().numpy())
        y_topk.append(topk_idx.cpu().numpy())

    wall_s = time.perf_counter() - wall_t0
    y_true_a = np.concatenate(y_true)
    y_pred_a = np.concatenate(y_pred)
    y_topk_a = np.concatenate(y_topk)

    return {
        "y_true": y_true_a,
        "y_pred": y_pred_a,
        "y_topk": y_topk_a,
        "logits_time": forward_s,
        "wall_time": wall_s,
        "n_samples": int(y_true_a.shape[0]),
    }


def topk_accuracy(y_true: np.ndarray, y_topk: np.ndarray) -> float:
    """Fraction of samples whose true label appears in the top-k predictions."""
    # y_topk: (N, K)
    hits = (y_topk == y_true[:, None]).any(axis=1)
    return float(hits.mean())


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_topk: Optional[np.ndarray] = None,
    *,
    labels: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """
    Spec metrics: top-1, top-5 (if y_topk given), overall accuracy,
    macro precision / recall / F1.
    """
    if labels is None:
        labels = list(range(int(max(y_true.max(), y_pred.max()) + 1)))

    top1 = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    out: Dict[str, float] = {
        "top1_acc": top1,
        "overall_acc": top1,  # same for single-label classification
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }
    if y_topk is not None:
        out["top5_acc"] = topk_accuracy(y_true, y_topk)
    return out


def make_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Return the full (C x C) confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=labels)


def most_confused_pairs(
    cm: np.ndarray,
    idx_to_name: Dict[int, str],
    *,
    top_n: int = 15,
) -> List[Tuple[str, str, int]]:
    """
    Largest off-diagonal confusion counts: (true_name, pred_name, count).

    Useful for the report's error analysis (spec asks for hardest pairs).
    """
    cm = cm.copy()
    np.fill_diagonal(cm, 0)
    flat = cm.ravel()
    order = np.argsort(flat)[::-1]
    pairs: List[Tuple[str, str, int]] = []
    n_classes = cm.shape[0]
    for idx in order:
        if len(pairs) >= top_n:
            break
        count = int(flat[idx])
        if count <= 0:
            break
        true_i, pred_i = divmod(int(idx), n_classes)
        pairs.append((idx_to_name[true_i], idx_to_name[pred_i], count))
    return pairs
