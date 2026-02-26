#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvalResult:
    threshold: float
    cm: np.ndarray
    acc: float
    prec: float
    rec: float
    f1: float
    auroc: float


def choose_threshold(
    errors_on: np.ndarray,
    errors_off: np.ndarray,
    method: str = "percentile",
    q: float = 95.0,
) -> float:
    """
    Choose threshold on recon error.

    Default: percentile of ON errors (e.g., 95th percentile).
    """
    if method == "percentile":
        return float(np.percentile(errors_on, q))
    if method == "midpoint":
        return float(0.5 * (np.median(errors_on) + np.median(errors_off)))
    raise ValueError(f"Unknown threshold method: {method}")


def estimate_threshold(
    project_fn: Callable[[torch.Tensor], torch.Tensor],
    errors_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_val_on: np.ndarray,
    threshold_method: str = "percentile",
    threshold_q: float = 95.0,
    device: torch.device = "cpu",
) -> float:
    """Use ONLY on-manifold validation data to estimate threshold."""
    xt = torch.tensor(x_val_on.astype(np.float32), dtype=torch.float32, device=device)
    with torch.no_grad():
        x_proj = project_fn(xt)
        err = errors_fn(xt, x_proj).cpu().numpy()

    if threshold_method == "percentile":
        thr = float(np.percentile(err, threshold_q))
    else:
        raise ValueError(f"Unknown threshold_method={threshold_method}")
    return thr


def eval_with_threshold(
    model_name: str,
    project_fn: Callable[[torch.Tensor], torch.Tensor],
    errors_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_test_on: np.ndarray,
    x_test_off: np.ndarray,
    thr: float,
    device: torch.device = "cpu",
) -> Tuple[EvalResult, Dict[str, np.ndarray]]:
    """Evaluate on test set using a FIXED threshold."""
    x = np.concatenate([x_test_on, x_test_off], axis=0).astype(np.float32)
    y_true = np.concatenate([np.ones(len(x_test_on)), np.zeros(len(x_test_off))], axis=0).astype(int)

    xt = torch.tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        x_proj = project_fn(xt)
        err = errors_fn(xt, x_proj).cpu().numpy()

    y_pred = (err <= thr).astype(int)  # 1=ON, 0=OFF

    # ---- metrics (replace with your numpy versions if you removed sklearn) ----
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    scores = -err
    try:
        auroc = roc_auc_score(y_true, scores)
    except ValueError:
        auroc = float("nan")

    res = EvalResult(
        threshold=thr,
        cm=cm,
        acc=float(acc),
        prec=float(prec),
        rec=float(rec),
        f1=float(f1),
        auroc=float(auroc),
    )

    print(f"\n[{model_name}] threshold(from val_on)={thr:.6f}")
    print(f"[{model_name}] CM (rows true [ON,OFF], cols pred [ON,OFF])\n{cm}")
    print(
        f"[{model_name}] acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} "
        f"f1={f1:.4f} auroc={auroc:.4f}"
    )

    cache = {"x": x, "y_true": y_true, "err": err, "x_proj": x_proj.cpu().numpy()}
    return res, cache
