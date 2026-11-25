"""Probability metrics + day-level bootstrap.

WHY bootstrap by day: connections on the same day share weather, disruptions and train-level delays.
Resampling individual rows treats them as independent and produces CIs that are far too narrow.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error, equal-width bins: sum_b |B|/N * |mean(y_b) - mean(p_b)|."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    n = len(y)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(total)


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    mp, my, cnt = [], [], []
    for b in range(n_bins):
        m = bins == b
        if m.sum() >= 50:
            mp.append(p[m].mean())
            my.append(y[m].mean())
            cnt.append(m.sum())
    return np.array(mp), np.array(my), np.array(cnt)


def suite(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece": ece(y, p),
    }
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    return out


def bootstrap_by_day(
    y: np.ndarray, p: np.ndarray, day: np.ndarray, metric, n_boot: int = 500, seed: int = 0
) -> tuple[float, float, float]:
    """(point, lo, hi) 95% percentile CI, resampling whole days with replacement."""
    rng = np.random.default_rng(seed)
    days, inv = np.unique(day, return_inverse=True)
    idx_by_day = [np.flatnonzero(inv == i) for i in range(len(days))]
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(days), len(days))
        idx = np.concatenate([idx_by_day[i] for i in pick])
        stats.append(metric(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(metric(y, p)), float(lo), float(hi)
