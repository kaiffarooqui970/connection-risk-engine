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
import polars as pl

from cre.config import eva_to_station, path, settings
from cre.contracts.schemas import RawStops
from cre.contracts.validate import validate_or_quarantine
from cre.logging_utils import log_event, setup

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
            log_event(log, "hf_month_downloaded", month=month, rows=n, secs=round(time.time() - t, 1))
            return n
        except duckdb.IOException as e:  # transient HTTP errors from the HF CDN
            wait = 2**attempt
            log_event(log, "hf_retry", level=logging.WARNING, month=month, attempt=attempt, error=str(e), retry_in=wait)
            time.sleep(wait)
    raise RuntimeError(f"failed to load {month} after {retries} attempts")


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; SET enable_progress_bar=false")
    return con


def warehouse() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path(settings()["data"]["duckdb_path"])))


def normalize(df: pl.DataFrame) -> pl.DataFrame:
    """Pin dtypes at the boundary (source parquet uses ns timestamps / int32). The contract then checks
    the *normalized* frame, so an upstream dtype change fails loudly instead of silently propagating."""
    return df.with_columns(
        pl.col(pl.Datetime).dt.cast_time_unit("us"),
        pl.col("train_line_station_num").cast(pl.Int64),
        pl.col("delay_in_min").cast(pl.Int64),
    )


def load_to_warehouse(months: list[str] | None = None) -> None:
    """Validate each month's parquet against `RawStops`; valid -> raw.stops, invalid -> quarantine.stops.

    Idempotent: a month's rows are deleted before insert, so re-running never duplicates.
    """
    months = months or settings()["data"]["months"]
    wh = warehouse()
    wh.execute("CREATE SCHEMA IF NOT EXISTS raw; CREATE SCHEMA IF NOT EXISTS quarantine")
    for m in months:
        f = path(f"{settings()['data']['raw_dir']}/stops_{m}.parquet")
        df = normalize(pl.read_parquet(f))
        res = validate_or_quarantine(df, RawStops, f"raw.stops[{m}]", max_invalid_frac=0.01)
        for schema, tbl in (("raw", res.valid), ("quarantine", res.quarantine)):
            wh.register("_tmp", tbl.to_arrow())
            wh.execute(f"CREATE TABLE IF NOT EXISTS {schema}.stops AS SELECT * FROM _tmp LIMIT 0")
            wh.execute(f"DELETE FROM {schema}.stops WHERE source_month = ?", [m])
            wh.execute(f"INSERT INTO {schema}.stops BY NAME SELECT * FROM _tmp")
            wh.unregister("_tmp")
        log_event(log, "hf_month_loaded", month=m, valid=res.valid.height, quarantined=res.quarantine.height)
    wh.close()


def main() -> None:
    setup()
    con = connect()
    for m in settings()["data"]["months"]:
        if not path(f"{settings()['data']['raw_dir']}/stops_{m}.parquet").exists():
            load_month(con, m)
    load_to_warehouse()


if __name__ == "__main__":
    main()
