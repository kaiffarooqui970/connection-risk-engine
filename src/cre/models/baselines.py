"""Baselines the model must beat. Each returns P(miss) so they plug into the same cost model.

- rule_transfer: the static DB-style rule "planned transfer < X min = risky"; P = training miss rate of each side.
- rule_slack: delay propagation, our proxy for "DB's forecast": risky iff
  planned transfer - A's known delay + B's known delay < MCT; P = training miss rate of each side.
- hist: the smoothed out-of-fold historical miss rate for (station, window bucket, hour), used directly.
- logreg: logistic regression on the same features as the main model (a linear, well-understood reference).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cre.config import mct_by_station
from cre.features.build import CATEGORICAL, NUMERIC

RULE_TRANSFER_X = 10  # minutes; a typical "tight connection" threshold. Re-tuned for the *decision* in Phase 7


class BinRule:
    """Binary rule with each side's probability = training miss rate (so it has a log loss)."""

    def __init__(self, flag_fn):
        self.flag_fn, self.p = flag_fn, {}

    def fit(self, df: pd.DataFrame) -> BinRule:
        f = self.flag_fn(df)
        self.p = {True: float(df.loc[f, "y"].mean()), False: float(df.loc[~f, "y"].mean())}
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        f = self.flag_fn(df)
        return np.where(f, self.p[True], self.p[False])


def rule_transfer(x: int = RULE_TRANSFER_X) -> BinRule:
    return BinRule(lambda d: (d["planned_transfer_min"] < x).to_numpy())


def rule_slack() -> BinRule:
    mct = mct_by_station()
    return BinRule(lambda d: (d["naive_slack"] < d["station_key"].astype(str).map(mct)).to_numpy())


def hist_predict(df: pd.DataFrame) -> np.ndarray:
    return df["hist_miss_rate"].to_numpy(dtype=float)


def logreg() -> Pipeline:
    num = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200, sparse_output=True)
    pre = ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=1.0, max_iter=1000))])


def to_lr_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[NUMERIC + CATEGORICAL].copy()
    for c in CATEGORICAL:
        out[c] = out[c].astype(str)
    return out
