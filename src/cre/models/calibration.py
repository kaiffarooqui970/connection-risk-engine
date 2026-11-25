"""Post-hoc calibration: none vs Platt vs isotonic, chosen by day-grouped CV on validation.

WHY CV inside validation: fitting a calibrator on validation and then scoring it on the same rows rewards
the most flexible method (isotonic). Two folds by day give an honest comparison; the winner is refit on all
of validation; the test month is never touched.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from cre.evaluate.metrics import suite

EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Identity:
    name = "uncalibrated"

    def fit(self, p, y):
        return self

    def predict(self, p):
        return np.asarray(p)


class Platt:
    name = "platt"

    def fit(self, p, y):
        self.lr = LogisticRegression(C=1e6).fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return self.lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class Isotonic:
    name = "isotonic"

    def fit(self, p, y):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p, y)
        return self

    def predict(self, p):
        return self.iso.predict(p)


METHODS = [Identity, Platt, Isotonic]


def compare(p: np.ndarray, y: np.ndarray, day: np.ndarray) -> dict[str, dict]:
    """2-fold CV by alternating days. Returns pooled out-of-fold metrics per method."""
    days = np.unique(day)
    fold = np.isin(day, days[::2])
    out = {}
    for M in METHODS:
        oof = np.empty_like(p, dtype=float)
        for f in (True, False):
            tr, te = fold == f, fold != f
            oof[te] = M().fit(p[tr], y[tr]).predict(p[te])
        out[M.name] = suite(y, oof)
    return out


def choose(results: dict[str, dict]) -> str:
    """Lowest log loss; ECE is reported alongside (a guardrail, not the selector)."""
    return min(results, key=lambda k: results[k]["log_loss"])


def fit(name: str, p: np.ndarray, y: np.ndarray):
    return {M.name: M for M in METHODS}[name]().fit(p, y)
