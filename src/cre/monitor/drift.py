"""Population Stability Index per feature: reference (training) vs current window.

PSI = sum_b (cur_b - ref_b) * ln(cur_b / ref_b). Conventional reading: < 0.1 stable, 0.1-0.25 moderate shift,
> 0.25 major shift. Numeric features use reference deciles as bin edges (so each reference bin holds ~10%);
missing values get their own bin. That matters here: "A not started" shows up as missing trends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4


def _psi(ref: np.ndarray, cur: np.ndarray) -> float:
    ref, cur = np.clip(ref, EPS, None), np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    r, c = ref.to_numpy(dtype=float), cur.to_numpy(dtype=float)
    edges = (
        np.unique(np.nanquantile(r[~np.isnan(r)], np.linspace(0, 1, bins + 1)[1:-1])) if (~np.isnan(r)).any() else []
    )

    def hist(x):
        nan = np.isnan(x)
        idx = np.searchsorted(edges, x[~nan], side="right")
        counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
        return np.append(counts, nan.sum()) / max(len(x), 1)

    return _psi(hist(r), hist(c))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    r, c = ref.astype(str).value_counts(normalize=True), cur.astype(str).value_counts(normalize=True)
    levels = r.index.union(c.index)
    return _psi(r.reindex(levels, fill_value=0).to_numpy(), c.reindex(levels, fill_value=0).to_numpy())


def psi_table(ref: pd.DataFrame, cur: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.Series:
    out = {f: psi_numeric(ref[f], cur[f]) for f in numeric}
    out.update({f: psi_categorical(ref[f], cur[f]) for f in categorical})
    return pd.Series(out).sort_values(ascending=False)
