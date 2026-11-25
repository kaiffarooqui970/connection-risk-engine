"""Candidate connections, labels, lost time and alternatives (Phase 4). The core logic of the project.

Definitions (also in docs/DESIGN.md §6 and DECISIONS.md):
- Candidate: train A arrives at hub S, a *different* train B departs S with planned transfer in
  [max(min_window, MCT(S)), max_window] minutes (a planner never offers a sub-MCT transfer). Excluded: same train number or same line (continuations / wings),
  B whose next stop is A's previous stop (backtracking), buses, and S-Bahn->S-Bahn pairs (nobody plans
  those at 5-10 min headways). For each A we keep only the *earliest* B towards each next stop T,
  ranked by earliest planned ARRIVAL at T (what a journey planner offers); later trains are its fallbacks.
- Label: missed = A's arrival cancelled OR B's departure cancelled OR actual transfer < MCT(S).
  Actual times come from the mart, where cancelled events have no actual time and departures are
  clamped to >= arrival. A held/late B therefore *saves* the connection naturally.
- T = B's next stop (by observed stop order). Fallback F = earliest ACTUAL arrival at T among departures
  from S (any train except B) with actual departure >= A's actual arrival + MCT, not cancelled, within
  `lost_time_cap_min`. Lost time L = F - B_ref, where B_ref = B's actual arrival at T (planned if B never got
  there). No fallback found -> L = cap, flagged censored.
- Alternatives: the K earliest-arriving planned departures from S to T leaving after B. Their *planned* times are known at
  prediction time (timetable), so `alt1` gives a leakage-free estimate of the lost time,
  L_hat = alt1 planned arrival at T - B planned arrival at T. Their actual times are used only to
  score the "commit to the alternative" policy.
"""

from __future__ import annotations

import logging

import duckdb
import polars as pl

from cre.config import path, settings
from cre.contracts.schemas import Connections
from cre.contracts.validate import validate_or_quarantine
from cre.logging_utils import log_event, setup

log = logging.getLogger("cre.connections")
K_ALTERNATIVES = 3


def build(con: duckdb.DuckDBPyConnection) -> None:
    cfg = settings()["connections"]
    wmin, wmax, cap = cfg["min_transfer_window_min"], cfg["max_transfer_window_min"], cfg["lost_time_cap_min"]
    con.execute("CREATE SCHEMA IF NOT EXISTS ml")

    con.execute("""
    CREATE OR REPLACE TEMP TABLE ev AS
    WITH e AS (
        SELECT *, coalesce(nullif(station_key, 'other'), station_name) AS ident FROM marts.fct_stop_events
    )
    SELECT *,
        lag(ident)     OVER w AS prev_ident,
        lead(ident)    OVER w AS next_ident,
        lead(stop_key) OVER w AS next_stop_key
    FROM e WINDOW w AS (PARTITION BY run_id ORDER BY stop_seq)
    """)

    con.execute(f"""
    CREATE OR REPLACE TABLE ml.connections AS
    WITH mct AS (SELECT DISTINCT station_key, mct_min FROM ref.hub_evas),
    cand AS (
        -- earliest *arrival* at T wins (what a journey planner offers), not earliest departure:
        -- otherwise a slow regional "B" is overtaken by a later fast train and lost time goes negative.
        SELECT a.stop_key AS a_key, b.stop_key AS b_key,
               row_number() OVER (PARTITION BY a.stop_key, b.next_ident
                                  ORDER BY bt.arr_planned NULLS LAST, b.dep_planned, b.stop_key) AS rk
        FROM ev a
        JOIN mct m0 ON m0.station_key = a.station_key
        JOIN ev b
          ON a.station_key = b.station_key AND a.is_hub AND b.is_hub
         AND b.dep_planned BETWEEN a.arr_planned + INTERVAL {wmin} MINUTE AND a.arr_planned + INTERVAL {wmax} MINUTE
         -- a planner never offers a transfer shorter than the station MCT -> not a planned connection
         AND b.dep_planned >= a.arr_planned + m0.mct_min * INTERVAL 1 MINUTE
        LEFT JOIN ev bt ON bt.stop_key = b.next_stop_key
        WHERE a.arr_planned IS NOT NULL
          AND a.run_id <> b.run_id
          AND coalesce(a.train_number, 'x') <> coalesce(b.train_number, 'y')
          AND coalesce(a.line_number, 'x') <> coalesce(b.line_number, 'y')
          AND b.next_ident IS NOT NULL
          AND b.next_ident IS DISTINCT FROM a.prev_ident
          AND a.train_category NOT IN ('bus', 'unknown') AND b.train_category NOT IN ('bus', 'unknown')
          AND NOT (a.train_category = 'sbahn' AND b.train_category = 'sbahn')
    )
    SELECT
        a.stop_key || '>' || b.stop_key                               AS conn_id,
        a.station_key, m.mct_min,
        cast(a.arr_planned AS date)                                    AS service_date,
        a.run_id AS a_run_id, a.stop_seq AS a_seq, a.stop_key AS a_stop_key,
        a.train_type AS a_train_type, a.train_category AS a_category, a.train_number AS a_train_number,
        a.arr_planned AS a_arr_planned, a.arr_actual AS a_arr_actual, a.arr_cancelled AS a_arr_cancelled,
        a.run_progress AS a_run_progress, a.first_seq AS a_first_seq, a.prev_ident AS a_prev_ident,
        b.run_id AS b_run_id, b.stop_seq AS b_seq, b.stop_key AS b_stop_key,
        b.train_type AS b_train_type, b.train_category AS b_category, b.train_number AS b_train_number,
        b.dep_planned AS b_dep_planned, b.dep_actual AS b_dep_actual, b.dep_cancelled AS b_dep_cancelled,
        b.run_progress AS b_run_progress, b.first_seq AS b_first_seq,
        b.next_ident AS t_ident,
        bt.arr_planned AS b_t_arr_planned, bt.arr_actual AS b_t_arr_actual,
        coalesce(bt.arr_actual, bt.arr_planned)                        AS b_ref_arr,
        date_diff('minute', a.arr_planned, b.dep_planned)              AS planned_transfer_min,
        date_diff('minute', a.arr_actual, b.dep_actual)                AS actual_transfer_min,
        CASE WHEN a.arr_cancelled THEN 'a_cancelled'
             WHEN b.dep_cancelled THEN 'b_cancelled'
             WHEN date_diff('minute', a.arr_actual, b.dep_actual) < m.mct_min THEN 'too_short'
             ELSE 'made' END                                           AS outcome
    FROM cand c
    JOIN ev a ON a.stop_key = c.a_key
    JOIN ev b ON b.stop_key = c.b_key
    JOIN mct m ON m.station_key = a.station_key
    LEFT JOIN ev bt ON bt.stop_key = b.next_stop_key
    WHERE c.rk = 1
    """)
    n_all = con.execute("SELECT count(*) FROM ml.connections").fetchall()[0][0]
    # Label undefined if an event is neither cancelled nor has an actual time (should be ~0 after contracts)
    con.execute("""DELETE FROM ml.connections
                   WHERE (NOT a_arr_cancelled AND a_arr_actual IS NULL)
                      OR (NOT b_dep_cancelled AND b_dep_actual IS NULL)
                      OR b_t_arr_planned IS NULL""")
    n = con.execute("SELECT count(*) FROM ml.connections").fetchall()[0][0]
    con.execute("ALTER TABLE ml.connections ADD COLUMN missed BOOLEAN")
    con.execute("UPDATE ml.connections SET missed = outcome <> 'made'")

    # Every departure from a hub that reaches some T later on the same run
    con.execute("""
    CREATE OR REPLACE TEMP TABLE reach AS
    SELECT d.station_key AS s, x.ident AS t, d.run_id AS d_run, d.train_type AS d_train_type,
           d.dep_planned, d.dep_actual, d.dep_cancelled,
           x.arr_planned AS t_arr_planned, x.arr_actual AS t_arr_actual, x.arr_cancelled AS t_arr_cancelled
    FROM ev d
    JOIN ev x ON x.run_id = d.run_id AND x.stop_seq > d.stop_seq
    WHERE d.is_hub AND d.dep_planned IS NOT NULL
      AND x.ident IN (SELECT DISTINCT t_ident FROM ml.connections)
    QUALIFY row_number() OVER (PARTITION BY d.stop_key, x.ident ORDER BY x.stop_seq) = 1
    """)

    # Fallback F: computed for EVERY connection (not only missed). For made connections it's the
    # counterfactual "if you don't take B", which the commit policy needs.
    con.execute(f"""
    CREATE OR REPLACE TABLE ml.fallback AS
    SELECT c.conn_id, min(r.t_arr_actual) AS fallback_arr
    FROM ml.connections c
    JOIN reach r
      ON r.s = c.station_key AND r.t = c.t_ident AND r.d_run <> c.b_run_id
     AND NOT r.dep_cancelled AND r.t_arr_actual IS NOT NULL
     AND r.dep_actual >= c.a_arr_actual + c.mct_min * INTERVAL 1 MINUTE
     AND r.dep_actual <= c.a_arr_actual + INTERVAL {cap} MINUTE
    WHERE NOT c.a_arr_cancelled
    GROUP BY 1
    """)

    # Next K planned alternatives to T after B (timetable = known at prediction time)
    con.execute(f"""
    CREATE OR REPLACE TABLE ml.alternatives AS
    SELECT c.conn_id,
           row_number() OVER (PARTITION BY c.conn_id ORDER BY r.t_arr_planned, r.dep_planned, r.d_run) AS alt_rank,
           r.d_run AS alt_run_id, r.d_train_type AS alt_train_type,
           r.dep_planned AS alt_dep_planned, r.dep_actual AS alt_dep_actual, r.dep_cancelled AS alt_dep_cancelled,
           r.t_arr_planned AS alt_t_arr_planned, r.t_arr_actual AS alt_t_arr_actual
    FROM ml.connections c
    JOIN reach r
      ON r.s = c.station_key AND r.t = c.t_ident AND r.d_run <> c.b_run_id
     AND r.dep_planned > c.b_dep_planned
     AND r.dep_planned <= c.b_dep_planned + INTERVAL {cap} MINUTE
     AND r.t_arr_planned IS NOT NULL
    QUALIFY alt_rank <= {K_ALTERNATIVES}
    """)

    con.execute(f"""
    CREATE OR REPLACE TABLE ml.labels AS
    SELECT c.conn_id, c.missed, c.outcome,
           f.fallback_arr,
           CASE WHEN c.a_arr_cancelled THEN NULL
                WHEN f.fallback_arr IS NULL THEN {cap}
                -- floor 0: if the fallback beats B's reference arrival (B cancelled + earlier train late), nothing is lost
                ELSE greatest(0, least({cap}, date_diff('minute', c.b_ref_arr, f.fallback_arr))) END AS lost_min_if_missed,
           (NOT c.a_arr_cancelled AND f.fallback_arr IS NULL)                         AS lost_censored,
           date_diff('minute', c.b_t_arr_planned, a1.alt_t_arr_planned)              AS l_hat_planned_min
    FROM ml.connections c
    LEFT JOIN ml.fallback f USING (conn_id)
    LEFT JOIN ml.alternatives a1 ON a1.conn_id = c.conn_id AND a1.alt_rank = 1
    """)
    log_event(log, "connections_built", candidates=n_all, kept=n, dropped_undefined=n_all - n)


def base_rates(con: duckdb.DuckDBPyConnection) -> dict[str, pl.DataFrame]:
    """Miss rates by transfer window, station and train-type pair (written to reports/base_rates.md)."""
    j = """FROM ml.connections c JOIN ml.labels l USING (conn_id)"""
    agg = """count(*) AS n, round(avg(c.missed::int), 4) AS miss_rate,
             round(avg((c.outcome = 'too_short')::int), 4) AS too_short,
             round(avg((c.outcome = 'b_cancelled')::int), 4) AS b_cancelled,
             round(avg((c.outcome = 'a_cancelled')::int), 4) AS a_cancelled,
             median(l.lost_min_if_missed) FILTER (WHERE c.missed) AS median_lost_if_missed,
             round(avg(l.lost_censored::int) FILTER (WHERE c.missed), 4) AS censored_share"""
    window = """CASE WHEN planned_transfer_min <= 4 THEN '02-04' WHEN planned_transfer_min <= 7 THEN '05-07'
                     WHEN planned_transfer_min <= 10 THEN '08-10' WHEN planned_transfer_min <= 15 THEN '11-15'
                     WHEN planned_transfer_min <= 20 THEN '16-20' ELSE '21-30' END"""
    q = {
        "overall": f"SELECT {agg} {j}",
        "by_window": f"SELECT {window} AS window, {agg} {j} GROUP BY 1 ORDER BY 1",
        "by_station": f"SELECT station_key, mct_min, {agg} {j} GROUP BY 1, 2 ORDER BY 1",
        "by_type_pair": f"SELECT a_category || ' -> ' || b_category AS pair, {agg} {j} GROUP BY 1 ORDER BY miss_rate DESC",
        "by_month": f"SELECT strftime(service_date, '%Y-%m') AS month, {agg} {j} GROUP BY 1 ORDER BY 1",
    }
    return {k: con.sql(v).pl() for k, v in q.items()}


def write_report(tables: dict[str, pl.DataFrame]) -> None:
    lines = [
        "# Base rates of missed connections",
        "",
        "Generated by `python -m cre.connections.build`. All connections, all months (Aug–Oct 2026).",
        "",
    ]
    for name, df in tables.items():
        lines += [f"## {name}", "", _md(df), ""]
    path("reports/base_rates.md").write_text("\n".join(lines))


def _md(df: pl.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in df.iter_rows()]
    return "\n".join([head, sep, *rows])


def validate(con: duckdb.DuckDBPyConnection) -> None:
    """Contract on the built table. Zero tolerance: these rows are produced by our own logic, so any
    violation is a bug in this module, not dirty input."""
    df = con.sql("""SELECT c.*, l.lost_min_if_missed::DOUBLE AS lost_min_if_missed
                    FROM ml.connections c JOIN ml.labels l USING (conn_id)""").pl()
    df = df.with_columns(pl.col("mct_min", "planned_transfer_min").cast(pl.Int64))
    validate_or_quarantine(df, Connections, "ml.connections", max_invalid_frac=0.0)


def main() -> None:
    setup()
    con = duckdb.connect(str(path(settings()["data"]["duckdb_path"])))
    build(con)
    validate(con)
    write_report(base_rates(con))
    con.close()


if __name__ == "__main__":
    main()
