"""Leakage tests: features must not change when anything after t_pred changes.

Three layers:
1. Synthetic world: perturb every post-t_pred fact (actual times, weather) -> identical features;
   perturb a pre-t_pred fact -> features DO change (proves the test is sensitive).
2. Real warehouse (skipped if not built): recompute sampled rows in a world where every actual time after
   t_pred is erased, compare with the stored features.
3. Historical rates: flipping val/test labels, or a training row's own-week labels, leaves its value unchanged.
"""

from datetime import date, datetime, timedelta

import duckdb
import polars as pl
import pytest

from cre.config import path, settings
from cre.features.build import FEATURES, compute_features, historical_rates

T0 = datetime(2026, 10, 6, 12, 0)  # A planned arrival at S
LEAD = 30
T_PRED = T0 - timedelta(minutes=LEAD)
CMP = ["conn_id"] + [f for f in FEATURES if not f.startswith("hist_")]


def _world(future_shift: int = 0, past_shift: int = 0, wx_future: float = 0.0) -> duckdb.DuckDBPyConnection:
    m = lambda x: T0 + timedelta(minutes=x)
    # A: 3 upstream stops, S is seq 4. Stops 1-2 observed by t_pred (11:30); stop 3 departs after t_pred.
    ev = [
        ("A", 1, m(-90), m(-90 + 2 + past_shift)),
        ("A", 2, m(-50), m(-50 + 5)),
        ("A", 3, m(-15), m(-15 + 9 + future_shift)),  # 11:45 planned: after t_pred -> must be invisible
        ("B", 1, m(-40), m(-40 + 3)),  # observed
        ("B", 2, m(-20), m(-20 + 4 + future_shift)),  # 11:40: after t_pred
    ]
    events = pl.DataFrame(
        [
            {
                "run_id": r,
                "stop_seq": s,
                "dep_planned": p,
                "dep_actual": a,
                "dep_delay_min": int((a - p).total_seconds() // 60),
            }
            for r, s, p, a in ev
        ]
    )
    base = pl.DataFrame(
        [
            {
                "conn_id": "c1",
                "lead_min": LEAD,
                "t_pred": T_PRED,
                "a_run_id": "A",
                "a_seq": 4,
                "a_first_seq": 1,
                "b_run_id": "B",
                "b_seq": 3,
                "b_first_seq": 1,
                "station_key": "leipzig",
                "a_arr_planned": T0,
                "service_date": T0.date(),
                "a_category": "long_distance",
                "b_category": "regional",
                "a_train_type": "ICE",
                "b_train_type": "RE",
                "a_train_number": "1",
                "b_train_number": "2",
                "planned_transfer_min": 9,
                "a_run_progress": 0.5,
                "l_hat_planned_min": 30,
            }
        ]
    )
    # weather: t_pred 11:30 local CEST = 09:30 UTC -> floor hour 09:00 UTC; rows after that are "future"
    wx = pl.DataFrame(
        [
            {
                "station_key": "leipzig",
                "ts_utc": datetime(2026, 10, 6, h),
                "temperature_c": 10.0 + (wx_future if h > 9 else 0),
                "precipitation_mm": 1.0 + (wx_future if h > 9 else 0),
                "wind_gust_kmh": 20.0 + (wx_future if h > 9 else 0),
            }
            for h in range(6, 14)
        ]
    )
    con = duckdb.connect()
    for name, df in {"base": base, "events": events, "wx": wx}.items():
        con.register(f"src_{name}", df.to_arrow())
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM src_{name}")
    compute_features(con, "base", "events", "wx", "feat")
    return con


def _feat(con) -> dict:
    return con.sql("SELECT * FROM feat").pl().select(CMP).row(0, named=True)


def test_future_changes_do_not_change_features():
    base = _feat(_world())
    assert _feat(_world(future_shift=120, wx_future=50.0)) == base


def test_past_changes_do_change_features():
    """Sensitivity check: if this failed, the test above would prove nothing."""
    assert _feat(_world(past_shift=20)) != _feat(_world())


def test_expected_values_in_synthetic_world():
    f = _feat(_world())
    assert f["a_known_delay"] == 5 and f["a_started"] == 1  # last observed = stop 2 (+5)
    assert f["a_delay_trend_1"] == 3  # 5 - 2
    assert f["a_stops_remaining"] == 2  # stops 3 and S
    assert f["b_known_delay"] == 3 and f["b_started"] == 1
    assert f["naive_slack"] == 9 - 5 + 3
    assert f["precip_mm_1h"] == 1.0 and f["precip_mm_3h"] == 3.0  # 07,08,09 UTC only


# ---------- real warehouse ----------

WAREHOUSE = path(settings()["data"]["duckdb_path"])


@pytest.mark.skipif(not WAREHOUSE.exists(), reason="warehouse not built (make data dbt connections features)")
def test_real_rows_identical_when_future_erased():
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    if "features" not in {
        r[0] for r in con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='ml'").fetchall()
    }:
        pytest.skip("features not built")
    sample = con.sql("SELECT conn_id, lead_min FROM ml.features USING SAMPLE 40 ROWS (reservoir, 42)").fetchall()
    for conn_id, lead in sample:
        con.execute(
            f"""CREATE OR REPLACE TEMP TABLE b1 AS
            SELECT c.*, l.l_hat_planned_min, {lead} AS lead_min, c.a_arr_planned - INTERVAL {lead} MINUTE AS t_pred
            FROM ml.connections c JOIN ml.labels l USING (conn_id) WHERE conn_id = ?""",
            [conn_id],
        )
        t_pred = con.execute("SELECT t_pred FROM b1").fetchall()[0][0]
        # the world as seen at t_pred: actual times after t_pred do not exist yet
        con.execute(
            """CREATE OR REPLACE TEMP TABLE ev1 AS
            SELECT run_id, stop_seq, dep_planned,
                   CASE WHEN dep_actual <= ? THEN dep_actual END AS dep_actual,
                   CASE WHEN dep_actual <= ? THEN dep_delay_min END AS dep_delay_min
            FROM marts.fct_stop_events WHERE run_id IN (SELECT a_run_id FROM b1 UNION SELECT b_run_id FROM b1)""",
            [t_pred, t_pred],
        )
        con.execute(
            """CREATE OR REPLACE TEMP TABLE wx1 AS SELECT * FROM staging.stg_weather
            WHERE ts_utc <= date_trunc('hour', timezone('UTC', timezone('Europe/Berlin', ?::TIMESTAMP)))""",
            [t_pred],
        )
        compute_features(con, "b1", "ev1", "wx1", "f1")
        got = con.sql("SELECT * FROM f1").pl().select(CMP).row(0, named=True)
        want = (
            con.sql(f"SELECT * FROM ml.features WHERE conn_id = '{conn_id}' AND lead_min = {lead}")
            .pl()
            .select(CMP)
            .row(0, named=True)
        )
        assert got == want, f"leakage suspected for {conn_id} lead {lead}"


# ---------- historical rates ----------


def _hist_world(flip_eval=False, flip_own_week=False):
    rows = []
    for i in range(400):
        d = date(2026, 8, 3) + timedelta(days=i % 70)  # spans train (<= 09-14) and val/test
        split = "train" if d <= date(2026, 9, 14) else ("val" if d <= date(2026, 9, 30) else "test")
        missed = i % 3 == 0
        if flip_eval and split != "train":
            missed = not missed
        if flip_own_week and split == "train" and d.isocalendar()[1] == 33 and i != 0:
            missed = not missed
        rows.append(
            {
                "conn_id": f"c{i}",
                "service_date": d,
                "split": split,
                "station_key": "leipzig",
                "planned_transfer_min": 9,
                "a_arr_planned": datetime.combine(d, datetime.min.time()) + timedelta(hours=8),
                "missed": missed,
                "a_train_number": "1",
                "b_train_number": "2",
                "mct_min": 5,
                "a_arr_cancelled": False,
                "a_arr_actual": datetime.combine(d, datetime.min.time()) + timedelta(hours=8, minutes=i % 9),
                "b_dep_cancelled": i % 10 == 0,
            }
        )
    con = duckdb.connect()
    con.register("_c", pl.DataFrame(rows).to_arrow())
    con.execute("CREATE TABLE conn AS SELECT * FROM _c")
    historical_rates(con, "conn", date(2026, 9, 14))
    return con.sql("SELECT * FROM _hist ORDER BY conn_id").pl()


def test_hist_rates_ignore_val_and_test_labels():
    a, b = _hist_world(), _hist_world(flip_eval=True)
    assert a["hist_miss_rate"].to_list() == b["hist_miss_rate"].to_list()


def test_hist_rates_out_of_fold_for_training_rows():
    a, b = _hist_world(), _hist_world(flip_own_week=True)
    own_week = a.filter((pl.col("split") == "train") & (pl.col("wk") == 202633))["conn_id"]
    assert own_week.len() > 0
    sel = lambda df: df.filter(pl.col("conn_id").is_in(own_week.to_list()))["hist_miss_rate"].to_list()
    assert sel(a) == sel(b)  # a week's own labels never feed that week's feature
    # ...but other training weeks DO see them (the feature is not trivially constant)
    other = a.filter((pl.col("split") == "train") & (pl.col("wk") != 202633))["hist_miss_rate"].to_list()
    other_b = b.filter((pl.col("split") == "train") & (pl.col("wk") != 202633))["hist_miss_rate"].to_list()
    assert other != other_b
