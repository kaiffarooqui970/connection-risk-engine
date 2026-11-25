"""Build `app/bundle/`: the small, derived data the Streamlit app needs (so it deploys without the warehouse).

Contents: test-month connections with features needed for display, main + fallback model probabilities,
Ê, actual outcomes; plus the metrics JSONs. Derived from DB data (CC BY 4.0), with attribution in README.
"""

from __future__ import annotations

import json
import pickle
import shutil

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from cre.config import path, settings
from cre.evaluate.policy_sim import load_frame
from cre.models.data import apply_categories, load


def main() -> None:
    out = path("app/bundle/x").parent
    con = duckdb.connect(str(path(settings()["data"]["duckdb_path"])), read_only=True)
    sim = load_frame(con)
    sim = sim[sim.split == "test"]
    feats = load(con)
    feats = feats[feats.split == "test"]
    meta = json.loads(path("models_store/meta.json").read_text())
    fb_feats = json.loads(path("reports/fallback_metrics.json").read_text())["features"]
    x = apply_categories(feats, meta["categories"])
    fb = lgb.Booster(model_file=str(path("models_store/fallback_lgb.txt")))
    with open(path("models_store/fallback_calibrator.pkl"), "rb") as f:
        fb_cal = pickle.load(f)  # noqa: S301
    feats = feats.assign(p_fallback=fb_cal.predict(fb.predict(x[fb_feats])))
    info = con.sql("""SELECT conn_id, a_train_type || ' ' || a_train_number AS train_a,
                             b_train_type || ' ' || b_train_number AS train_b, t_ident AS next_stop_b,
                             a_arr_planned, a_arr_actual, b_dep_planned, b_dep_actual
                      FROM ml.connections""").df()
    b = sim.merge(
        feats[["conn_id", "lead_min", "p_fallback", "a_known_delay", "b_known_delay", "a_started"]],
        on=["conn_id", "lead_min"],
    ).merge(info, on="conn_id")
    keep = [
        "conn_id",
        "lead_min",
        "service_date",
        "station_key",
        "a_category",
        "b_category",
        "train_a",
        "train_b",
        "next_stop_b",
        "a_arr_planned",
        "a_arr_actual",
        "b_dep_planned",
        "b_dep_actual",
        "planned_transfer_min",
        "a_known_delay",
        "b_known_delay",
        "a_started",
        "naive_slack",
        "e_hat",
        "p_lgb_cal",
        "p_logreg",
        "p_rule_slack",
        "p_fallback",
        "missed",
        "outcome",
        "lost",
    ]
    b = b[keep].copy()
    from cre.evaluate.metrics import ece

    daily = (
        b.groupby(b.service_date.astype(str))
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "predicted": g.p_lgb_cal.mean(),
                    "observed": g.missed.mean(),
                    "ece": ece(g.missed.to_numpy(float), g.p_lgb_cal.to_numpy()),
                }
            )
        )
        .reset_index()
        .rename(columns={"service_date": "day"})
    )
    daily.to_parquet(out / "daily.parquet", index=False)
    # row-level explorer: last 7 days of the test month keeps the repo small
    last = pd.to_datetime(b.service_date).max()
    b = b[pd.to_datetime(b.service_date) > last - pd.Timedelta(days=7)]
    for c in ["p_lgb_cal", "p_logreg", "p_rule_slack", "p_fallback"]:
        b[c] = b[c].astype(np.float32)
    b.to_parquet(out / "test_connections.parquet", index=False, compression="zstd")
    for f in ["model_metrics.json", "policy_metrics.json", "fallback_metrics.json"]:
        shutil.copy(path(f"reports/{f}"), out / f)
    print(f"bundle: {len(b)} rows, {sum(p.stat().st_size for p in out.iterdir()) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
