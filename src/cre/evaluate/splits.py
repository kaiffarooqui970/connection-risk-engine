"""Time-based train/validation/test split by service date, purged so no run spans two splits.

WHY purge: a connection belongs to the split of A's arrival date, but a run that crosses the boundary
at midnight could appear in a train connection and a test connection, sharing its delay history.
Dropping those few connections (night trains at the two boundaries) costs almost nothing and removes
a leakage path. Rows before the first configured month (partial edge day) are dropped too.
"""

from __future__ import annotations

from datetime import date

import duckdb

from cre.config import settings


def split_case(col: str) -> str:
    s = settings()["split"]
    return (
        f"CASE WHEN {col} <= DATE '{s['train_end']}' THEN 'train' "
        f"WHEN {col} <= DATE '{s['val_end']}' THEN 'val' ELSE 'test' END"
    )


def assign_splits(con: duckdb.DuckDBPyConnection) -> None:
    start = date.fromisoformat(settings()["data"]["months"][0] + "-01")
    con.execute(f"""
    CREATE OR REPLACE TABLE ml.conn_split AS
    WITH run_span AS (
        SELECT run_id,
               {split_case("cast(min(coalesce(arr_planned, dep_planned)) AS date)")} AS s_first,
               {split_case("cast(max(coalesce(arr_planned, dep_planned)) AS date)")} AS s_last
        FROM marts.fct_stop_events GROUP BY run_id
    )
    SELECT c.conn_id, {split_case("c.service_date")} AS split
    FROM ml.connections c
    JOIN run_span ra ON ra.run_id = c.a_run_id
    JOIN run_span rb ON rb.run_id = c.b_run_id
    WHERE c.service_date >= DATE '{start}'
      AND ra.s_first = ra.s_last AND rb.s_first = rb.s_last
    """)
