"""Leakage-free features at prediction time t_pred = planned arrival of A at S - lead (Phase 5).

The rule: a feature may only use information timestamped <= t_pred.
- Realized events: an upstream stop's departure is *observed* iff its actual departure <= t_pred.
  The final recorded value is used; DB publishes realized times within ~1 min (documented assumption).
- Cancellation flags have NO announcement timestamp in the source, so they are never used as features.
- Timetable facts (planned times, stop sequence, the scheduled alternative L_hat) are known in advance.
- Weather: the hourly observation whose hour *ends* at or before floor_hour(t_pred) (UTC).
- Historical rates: training period only; out-of-fold by ISO week *inside* training (a row never sees its
  own week's outcomes), smoothed towards the training mean (m-estimate) so rare keys don't get 0 or 1.

`compute_features` takes table names so the leakage test can rerun it on a world where everything after
t_pred is erased, and assert identical output.
"""

from __future__ import annotations

import logging
from datetime import date

import duckdb

from cre.config import ROOT, path, settings
from cre.evaluate.splits import assign_splits
from cre.logging_utils import log_event, setup

log = logging.getLogger("cre.features")
M_SMOOTH = 50  # m-estimate strength: a key needs ~50 observations before its own rate dominates

NUMERIC = [
    "lead_min",
    "planned_transfer_min",
    "a_known_delay",
    "a_started",
    "a_delay_trend_1",
    "a_delay_trend_2",
    "a_stops_remaining",
    "a_planned_min_remaining",
    "a_min_since_obs",
    "a_run_progress",
    "b_known_delay",
    "b_started",
    "b_stops_before_s",
    "naive_slack",
    "hour",
    "weekday",
    "is_weekend",
    "is_rush",
    "is_public_holiday",
    "is_school_holiday",
    "temperature_c",
    "precip_mm_1h",
    "precip_mm_3h",
    "wind_gust_kmh",
    "hist_miss_rate",
    "hist_a_late_rate",
    "hist_b_cancel_rate",
    "l_hat_planned_min",
]
CATEGORICAL = ["station_key", "a_category", "b_category", "a_train_type", "b_train_type"]
FEATURES = NUMERIC + CATEGORICAL

WINDOW_BUCKET = """CASE WHEN planned_transfer_min <= 7 THEN 'w05_07' WHEN planned_transfer_min <= 10 THEN 'w08_10'
    WHEN planned_transfer_min <= 15 THEN 'w11_15' WHEN planned_transfer_min <= 20 THEN 'w16_20' ELSE 'w21_30' END"""


def base_table(con: duckdb.DuckDBPyConnection, connections: str, out: str, leads: list[int]) -> None:
    """One row per (connection, lead) with t_pred."""
    leads_sql = ", ".join(f"({x})" for x in leads)
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE {out} AS
    SELECT c.*, l.lead_min, c.a_arr_planned - l.lead_min * INTERVAL 1 MINUTE AS t_pred
    FROM {connections} c CROSS JOIN (VALUES {leads_sql}) l(lead_min)
    """)


def compute_features(con: duckdb.DuckDBPyConnection, base: str, events: str, weather: str, out: str) -> None:
    """Real-time + calendar + weather features. Historical-rate features are joined separately."""
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE _obs_a AS
    SELECT b.conn_id, b.lead_min,
           max(e.stop_seq)                                         AS last_seq,
           arg_max(e.dep_delay_min, e.stop_seq)                    AS last_delay,
           arg_max(e.dep_actual, e.stop_seq)                       AS last_obs_time,
           arg_max(e.dep_planned, e.stop_seq)                      AS last_dep_planned,
           list(e.dep_delay_min ORDER BY e.stop_seq DESC)          AS delays_desc
    FROM {base} b
    JOIN {events} e ON e.run_id = b.a_run_id AND e.stop_seq < b.a_seq
     AND e.dep_actual IS NOT NULL AND e.dep_actual <= b.t_pred
    GROUP BY 1, 2
    """)
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE _obs_b AS
    SELECT b.conn_id, b.lead_min, arg_max(e.dep_delay_min, e.stop_seq) AS last_delay
    FROM {base} b
    JOIN {events} e ON e.run_id = b.b_run_id AND e.stop_seq < b.b_seq
     AND e.dep_actual IS NOT NULL AND e.dep_actual <= b.t_pred
    GROUP BY 1, 2
    """)
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE _wx AS
    WITH k AS (
        SELECT conn_id, lead_min, station_key,
               date_trunc('hour', timezone('UTC', timezone('Europe/Berlin', t_pred))) AS h
        FROM {base}
    )
    SELECT k.conn_id, k.lead_min,
           max(w.temperature_c)   FILTER (WHERE w.ts_utc = k.h) AS temperature_c,
           max(w.precipitation_mm) FILTER (WHERE w.ts_utc = k.h) AS precip_mm_1h,
           sum(w.precipitation_mm)                               AS precip_mm_3h,
           max(w.wind_gust_kmh)   FILTER (WHERE w.ts_utc = k.h) AS wind_gust_kmh
    FROM k
    LEFT JOIN {weather} w ON w.station_key = k.station_key
     AND w.ts_utc BETWEEN k.h - INTERVAL 2 HOUR AND k.h
    GROUP BY 1, 2
    """)
    states = {s["key"]: s["state"] for s in settings()["stations"]}
    state_case = "CASE " + " ".join(f"WHEN b.station_key = '{k}' THEN '{v}'" for k, v in states.items()) + " END"
    hol = str(ROOT / "configs" / "holidays_2026.csv")
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE {out} AS
    WITH hol AS (SELECT * FROM read_csv('{hol}', comment='#', header=true))
    SELECT
        b.conn_id, b.lead_min, b.t_pred, b.service_date, b.station_key,
        b.a_category, b.b_category, coalesce(b.a_train_type, 'unknown') AS a_train_type,
        coalesce(b.b_train_type, 'unknown') AS b_train_type, b.a_train_number, b.b_train_number,
        b.planned_transfer_min,
        coalesce(oa.last_delay, 0)                                   AS a_known_delay,
        (oa.last_seq IS NOT NULL)::INT                               AS a_started,
        oa.delays_desc[1] - oa.delays_desc[2]                        AS a_delay_trend_1,
        oa.delays_desc[1] - oa.delays_desc[3]                        AS a_delay_trend_2,
        CASE WHEN oa.last_seq IS NULL THEN b.a_seq - b.a_first_seq ELSE b.a_seq - oa.last_seq END AS a_stops_remaining,
        date_diff('minute', coalesce(oa.last_dep_planned, b.t_pred), b.a_arr_planned) AS a_planned_min_remaining,
        date_diff('minute', oa.last_obs_time, b.t_pred)              AS a_min_since_obs,
        b.a_run_progress,
        coalesce(ob.last_delay, 0)                                   AS b_known_delay,
        (ob.last_delay IS NOT NULL)::INT                             AS b_started,
        b.b_seq - b.b_first_seq                                      AS b_stops_before_s,
        b.planned_transfer_min - coalesce(oa.last_delay, 0) + coalesce(ob.last_delay, 0) AS naive_slack,
        hour(b.a_arr_planned)                                        AS hour,
        isodow(b.a_arr_planned)                                      AS weekday,
        (isodow(b.a_arr_planned) >= 6)::INT                          AS is_weekend,
        (isodow(b.a_arr_planned) <= 5 AND (hour(b.a_arr_planned) BETWEEN 6 AND 8
            OR hour(b.a_arr_planned) BETWEEN 16 AND 18))::INT        AS is_rush,
        EXISTS (SELECT 1 FROM hol WHERE hol.kind = 'public' AND hol.state = {state_case}
                AND b.service_date BETWEEN hol."start" AND hol."end")::INT AS is_public_holiday,
        EXISTS (SELECT 1 FROM hol WHERE hol.kind = 'school' AND hol.state = {state_case}
                AND b.service_date BETWEEN hol."start" AND hol."end")::INT AS is_school_holiday,
        wx.temperature_c, wx.precip_mm_1h, wx.precip_mm_3h, wx.wind_gust_kmh,
        -- scheduled lost-time estimate: no alternative within the cap -> cap; a faster later train -> 0
        least(240, greatest(0, coalesce(b.l_hat_planned_min, 240)))  AS l_hat_planned_min
    FROM {base} b
    LEFT JOIN _obs_a oa USING (conn_id, lead_min)
    LEFT JOIN _obs_b ob USING (conn_id, lead_min)
    LEFT JOIN _wx wx USING (conn_id, lead_min)
    """)


def historical_rates(con: duckdb.DuckDBPyConnection, conn: str, train_end: date) -> None:
    """Training-period rates with out-of-fold (ISO week) values for training rows.

    Three keys: (station, transfer-window bucket, hour) miss rate; A's train number late-arrival rate at S
    (>= MCT late); B's train number cancellation rate at S. Each is smoothed to the training mean.
    """
    specs = {
        "hist_miss_rate": (f"station_key, {WINDOW_BUCKET}, hour(a_arr_planned)", "missed::INT"),
        "hist_a_late_rate": (
            "station_key, a_train_number",
            "(a_arr_cancelled OR date_diff('minute', a_arr_planned, a_arr_actual) >= mct_min)::INT",
        ),
        "hist_b_cancel_rate": ("station_key, b_train_number", "b_dep_cancelled::INT"),
    }
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _hist AS
                    SELECT conn_id, service_date, split, yearweek(service_date) AS wk FROM {conn}""")
    for name, (key, target) in specs.items():
        con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _h AS
        WITH tr AS (
            SELECT conn_id, md5(concat_ws('|', {key})) AS k, yearweek(service_date) AS wk, {target} AS y
            FROM {conn} WHERE split = 'train' AND service_date <= DATE '{train_end}'
        ),
        tot AS (SELECT sum(y) AS s, count(*) AS n FROM tr),
        by_w AS (SELECT wk, sum(y) AS s, count(*) AS n FROM tr GROUP BY wk),
        by_k AS (SELECT k, sum(y) AS s, count(*) AS n FROM tr GROUP BY k),
        by_kw AS (SELECT k, wk, sum(y) AS s, count(*) AS n FROM tr GROUP BY k, wk),
        allrows AS (SELECT conn_id, split, md5(concat_ws('|', {key})) AS k, yearweek(service_date) AS wk FROM {conn}),
        oof AS (  -- for training rows remove the row's own week from BOTH the key counts and the prior
            SELECT a.conn_id,
                   coalesce(bk.s, 0) - CASE WHEN a.split = 'train' THEN coalesce(bw.s, 0) ELSE 0 END AS ks,
                   coalesce(bk.n, 0) - CASE WHEN a.split = 'train' THEN coalesce(bw.n, 0) ELSE 0 END AS kn,
                   (t.s - CASE WHEN a.split = 'train' THEN coalesce(w.s, 0) ELSE 0 END)
                 / (t.n - CASE WHEN a.split = 'train' THEN coalesce(w.n, 0) ELSE 0 END) AS p0
            FROM allrows a CROSS JOIN tot t
            LEFT JOIN by_w w ON w.wk = a.wk
            LEFT JOIN by_k bk ON bk.k = a.k
            LEFT JOIN by_kw bw ON bw.k = a.k AND bw.wk = a.wk
        )
        SELECT conn_id, (ks + {M_SMOOTH} * p0) / (kn + {M_SMOOTH}) AS v FROM oof
        """)
        con.execute(f"""CREATE OR REPLACE TEMP TABLE _hist AS
                        SELECT h.*, _h.v AS {name} FROM _hist h JOIN _h USING (conn_id)""")


def build(con: duckdb.DuckDBPyConnection) -> None:
    cfg = settings()
    train_end = date.fromisoformat(cfg["split"]["train_end"])
    assign_splits(con)  # -> ml.conn_split (connections with split, purged boundary runs)
    con.execute("""CREATE OR REPLACE TEMP TABLE _conn AS
                   SELECT c.*, s.split, l.l_hat_planned_min
                   FROM ml.connections c JOIN ml.conn_split s USING (conn_id) JOIN ml.labels l USING (conn_id)""")
    base_table(con, "_conn", "_base", cfg["prediction"]["lead_times_min"])
    compute_features(con, "_base", "marts.fct_stop_events", "staging.stg_weather", "_feat")
    historical_rates(con, "_conn", train_end)
    con.execute("""
    CREATE OR REPLACE TABLE ml.features AS
    SELECT f.*, h.split, h.hist_miss_rate, h.hist_a_late_rate, h.hist_b_cancel_rate,
           l.missed, l.outcome, l.lost_min_if_missed, l.lost_censored
    FROM _feat f JOIN _hist h USING (conn_id) JOIN ml.labels l USING (conn_id)
    """)
    n = con.execute("SELECT count(*), count(DISTINCT conn_id) FROM ml.features").fetchall()[0]
    log_event(log, "features_built", rows=n[0], connections=n[1])


def main() -> None:
    setup()
    con = duckdb.connect(str(path(settings()["data"]["duckdb_path"])))
    build(con)
    con.close()


if __name__ == "__main__":
    main()
