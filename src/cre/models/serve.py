"""Load model artifacts and score rows, with the stale-data fallback built in (one code path for app + monitor)."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd

from cre.config import path
from cre.models.data import apply_categories


@dataclass
class Scorer:
    boosters: list
    calibrator: object
    features: list[str]
    categories: dict
    fb_booster: object | None
    fb_calibrator: object | None
    fb_features: list[str] | None

    def score(self, df: pd.DataFrame, freshness: str = "FRESH") -> tuple[np.ndarray, str]:
        """Return (P(miss), confidence). Non-FRESH live data -> static-feature fallback model."""
        x = apply_categories(df, self.categories)
        if freshness != "FRESH" and self.fb_booster is not None:
            p = self.fb_booster.predict(x[self.fb_features])
            return self.fb_calibrator.predict(p), "reduced (fallback model)"
        raw = np.mean([b.predict(x[self.features]) for b in self.boosters], axis=0)
        return self.calibrator.predict(raw), "full"


@lru_cache
def load_scorer() -> Scorer:
    store = path("models_store/x").parent
    meta = json.loads((store / "meta.json").read_text())
    boosters = [lgb.Booster(model_file=str(store / f"lgb_seed{s}.txt")) for s in meta["seeds"]]
    with open(store / "calibrator.pkl", "rb") as f:
        cal = pickle.load(f)  # noqa: S301 - our own artifact, written by cre.models.train
    fb_b = fb_c = fb_f = None
    if (store / "fallback_lgb.txt").exists():
        fb_b = lgb.Booster(model_file=str(store / "fallback_lgb.txt"))
        with open(store / "fallback_calibrator.pkl", "rb") as f:
            fb_c = pickle.load(f)  # noqa: S301
        fb_f = json.loads(path("reports/fallback_metrics.json").read_text())["features"]
    return Scorer(boosters, cal, meta["features"], meta["categories"], fb_b, fb_c, fb_f)
