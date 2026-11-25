"""Pull the historical subset from the Hugging Face dataset into local parquet + DuckDB.

Why a two-step semi-join instead of "rows at hubs only": features need A's *earlier* stops
(delay trend) and labels need B's *next* stop (lost time), so we keep every stop of every ride
that touches a hub. DuckDB reads the remote parquet with HTTP range requests, so only the
needed columns are transferred and we never store a full 360 MB month on disk.
"""

from __future__ import annotations

import logging
import time

import duckdb

from cre.config import eva_to_station, path, settings

log = logging.getLogger("cre.hf_loader")

COLUMNS = """
    id, eva, station_name, train_type, train_number, line_number, final_destination_station,
    train_line_ride_id, train_line_station_num, arrival_planned_time, arrival_change_time,
    departure_planned_time, departure_change_time, arrival_is_canceled, departure_is_canceled,
    delay_in_min, is_additional_stop, is_replacement_train
"""


def month_url(month: str) -> str:
    return f"{settings()['data']['hf_base']}/data-{month}.parquet"


def load_month(con: duckdb.DuckDBPyConnection, month: str, retries: int = 3) -> int:
    """Write data/raw/stops_<month>.parquet. Idempotent: overwrites the month file."""
    out = path(f"{settings()['data']['raw_dir']}/stops_{month}.parquet")
    evas = ",".join(f"'{e}'" for e in eva_to_station())
    url = month_url(month)
    sql = f"""
        COPY (
            WITH hub_rides AS (
                SELECT DISTINCT train_line_ride_id FROM '{url}' WHERE eva IN ({evas})
            )
            SELECT {COLUMNS}, '{month}' AS source_month
            FROM '{url}' s
            SEMI JOIN hub_rides h USING (train_line_ride_id)
        ) TO '{out}' (FORMAT parquet, COMPRESSION zstd)
    """
    for attempt in range(1, retries + 1):
        try:
            t = time.time()
            con.execute(sql)
            n = con.execute(f"SELECT count(*) FROM '{out}'").fetchall()[0][0]
            log.info("month=%s rows=%d secs=%.1f", month, n, time.time() - t)
            return n
        except duckdb.IOException as e:  # transient HTTP errors from the HF CDN
            wait = 2**attempt
            log.warning("month=%s attempt=%d error=%s retry_in=%ds", month, attempt, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"failed to load {month} after {retries} attempts")


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; SET enable_progress_bar=false")
    return con


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    con = connect()
    for m in settings()["data"]["months"]:
        load_month(con, m)


if __name__ == "__main__":
    main()
