"""Live scoring path: DB Timetables plan + change feed for the hubs -> connections -> features -> P(miss).

Train/serve skew (documented in DESIGN §5.1): in training, "known delay of A" is A's realized delay at its
last departed upstream stop. Live, we only poll the hubs, so we use DB's *forecast* delay for A at S
(`ct - pt` from the latest change snapshot at or before now). A has started iff it has a forecast.
Trend features are unavailable live (NaN; the model has seen NaN for not-yet-started trains). PSI on
`a_known_delay` and the calibration tracker are what tell us if this skew hurts.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import duckdb
import numpy as np
import pandas as pd

from cre.config import eva_to_station, mct_by_station, settings
from cre.contracts.validate import check_freshness
from cre.decision.policy import CAP
from cre.monitor.rules import LIVE_MAX_AGE_MIN

LD = {"ICE", "IC", "EC", "ECE", "EN", "NJ", "FLX", "RJ", "RJX", "D", "TGV", "EST"}


def category(t: str | None) -> str:
    if t is None:
        return "unknown"
    if t in LD:
        return "long_distance"
    if t == "S":
        return "sbahn"
    if t in {"Bus", "SEV", "BSV"}:
        return "bus"
    return "regional"


def freshness(con: duckdb.DuckDBPyConnection, now: datetime) -> str:
    latest = con.execute(
        "SELECT max(fetched_at) FROM live.fetch_log WHERE endpoint = 'fchg' AND status = 'ok'"
    ).fetchall()[0][0]
    return check_freshness("db_fchg", latest, now, timedelta(minutes=LIVE_MAX_AGE_MIN)).status


def live_connections(con: duckdb.DuckDBPyConnection, now: datetime, lead_min: int = 30) -> pd.DataFrame:
    """Connections whose A arrives in [now + lead - 5, now + lead + 5]: what a passenger would ask about now."""
    stations = eva_to_station()
    con.register("_evas", pd.DataFrame({"eva": list(stations), "station_key": list(stations.values())}))
    lo, hi = now + timedelta(minutes=lead_min - 5), now + timedelta(minutes=lead_min + 5)
    df = con.execute(
        """
    WITH latest AS (
        SELECT * FROM live.changes WHERE fetched_at <= ?
        QUALIFY row_number() OVER (PARTITION BY id ORDER BY fetched_at DESC) = 1
    ),
    s AS (
        SELECT p.id, e.station_key, p.train_type, p.train_number, p.line, p.arr_planned, p.dep_planned,
               p.arr_path, p.dep_path,
               date_diff('minute', p.arr_planned, l.arr_change) AS arr_fc_delay,
               date_diff('minute', p.dep_planned, l.dep_change) AS dep_fc_delay
        FROM live.plan p JOIN _evas e USING (eva) LEFT JOIN latest l USING (id)
    )
    SELECT a.station_key, a.id AS a_id, b.id AS b_id, a.train_type AS a_train_type, b.train_type AS b_train_type,
           a.train_number AS a_train_number, b.train_number AS b_train_number, a.arr_planned AS a_arr_planned,
           b.dep_planned AS b_dep_planned, date_diff('minute', a.arr_planned, b.dep_planned) AS planned_transfer_min,
           a.arr_fc_delay, b.dep_fc_delay, a.arr_path, b.dep_path
    FROM s a JOIN s b ON a.station_key = b.station_key AND a.id <> b.id
     AND b.dep_planned BETWEEN a.arr_planned + INTERVAL 2 MINUTE AND a.arr_planned + INTERVAL 30 MINUTE
    WHERE a.arr_planned BETWEEN ? AND ? AND coalesce(a.train_number, 'x') <> coalesce(b.train_number, 'y')
    """,
        [now, lo, hi],
    ).df()
    if df.empty:
        return df
    mct = mct_by_station()
    df = df[df["planned_transfer_min"] >= df["station_key"].map(mct)]
    df["a_category"], df["b_category"] = df["a_train_type"].map(category), df["b_train_type"].map(category)
    df = df[~df["a_category"].isin(["bus", "unknown"]) & ~df["b_category"].isin(["bus", "unknown"])]
    df = df[~((df["a_category"] == "sbahn") & (df["b_category"] == "sbahn"))]
    return df.reset_index(drop=True)


def live_features(conns: pd.DataFrame, now: datetime, lead_min: int, hist: pd.DataFrame | None = None) -> pd.DataFrame:
    """Same columns as ml.features. Historical rates come from the training-period tables (`hist`)."""
    f = conns.copy()
    f["lead_min"] = float(lead_min)
    f["a_started"] = f["arr_fc_delay"].notna().astype(float)
    f["a_known_delay"] = f["arr_fc_delay"].fillna(0).astype(float)
    f["b_started"] = f["dep_fc_delay"].notna().astype(float)
    f["b_known_delay"] = f["dep_fc_delay"].fillna(0).astype(float)
    f["naive_slack"] = f["planned_transfer_min"] - f["a_known_delay"] + f["b_known_delay"]
    for c in [
        "a_delay_trend_1",
        "a_delay_trend_2",
        "a_min_since_obs",
        "a_run_progress",
        "b_stops_before_s",
        "temperature_c",
        "precip_mm_1h",
        "precip_mm_3h",
        "wind_gust_kmh",
    ]:
        f[c] = np.nan
    f["a_stops_remaining"] = np.nan
    f["a_planned_min_remaining"] = float(lead_min)
    t = pd.to_datetime(f["a_arr_planned"])
    f["hour"], f["weekday"] = t.dt.hour.astype(float), t.dt.isocalendar().day.astype(float)
    f["is_weekend"] = (f["weekday"] >= 6).astype(float)
    f["is_rush"] = ((f["weekday"] <= 5) & (t.dt.hour.between(6, 8) | t.dt.hour.between(16, 18))).astype(float)
    f["is_public_holiday"] = f["is_school_holiday"] = 0.0
    f["l_hat_planned_min"] = float(CAP)  # unknown without the next hours' plan; conservative
    for c in ["hist_miss_rate", "hist_a_late_rate", "hist_b_cancel_rate"]:
        f[c] = hist[c].mean() if hist is not None else np.nan
    return f


def score_now(live_db: str, now: datetime | None = None, lead_min: int = 30) -> tuple[pd.DataFrame, str]:
    from cre.models.serve import load_scorer

    now = now or datetime.now()
    con = duckdb.connect(live_db, read_only=True)
    fresh = freshness(con, now)
    conns = live_connections(con, now, lead_min)
    if conns.empty:
        return conns, fresh
    feats = live_features(conns, now, lead_min)
    p, conf = load_scorer().score(feats, fresh)
    feats["p_miss"], feats["confidence"] = p, conf
    return feats, fresh


def settings_lead() -> list[int]:
    return settings()["prediction"]["lead_times_min"]
