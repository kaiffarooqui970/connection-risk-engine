"""Load the feature table into pandas, with dtypes each model family needs."""

from __future__ import annotations

import duckdb
import pandas as pd

from cre.config import path, settings
from cre.features.build import CATEGORICAL, FEATURES, NUMERIC

META = [
    "conn_id",
    "lead_min",
    "service_date",
    "split",
    "missed",
    "outcome",
    "lost_min_if_missed",
    "lost_censored",
    "a_train_number",
    "b_train_number",
]


def load(con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    con = con or duckdb.connect(str(path(settings()["data"]["duckdb_path"])), read_only=True)
    cols = ", ".join(dict.fromkeys(META + FEATURES))
    df = con.sql(f"SELECT {cols} FROM ml.features ORDER BY conn_id, lead_min").df()
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    for c in NUMERIC:
        df[c] = df[c].astype("float32")
    df["y"] = df["missed"].astype(int)
    df["service_date"] = pd.to_datetime(df["service_date"])
    return df


def categories(train: pd.DataFrame) -> dict[str, list]:
    """Freeze category levels from TRAIN so val/test/live encode identically (unseen -> NaN)."""
    return {c: list(train[c].cat.categories) for c in CATEGORICAL}


def apply_categories(df: pd.DataFrame, cats: dict[str, list]) -> pd.DataFrame:
    df = df.copy()
    for c, levels in cats.items():
        df[c] = pd.Categorical(df[c].astype(str), categories=levels)
    return df
