"""DWD hourly weather via Bright Sky (https://brightsky.dev, no API key), nearest station per hub.

WHY UTC keys: train times are naive local (Europe/Berlin). On the DST fall-back night local 02:00 occurs
twice, which would break a (station, local_ts) unique key. We store `ts` in UTC and convert at join time.

Semantics: Bright Sky's value at timestamp T covers the hour *ending* at T (precipitation is the
sum over (T-1h, T]). A feature at prediction time t may therefore only use rows with ts <= floor_hour(t).
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta

import duckdb
import httpx
import polars as pl

from cre.config import path, settings
from cre.contracts.schemas import Weather
from cre.contracts.validate import validate_or_quarantine
from cre.logging_utils import log_event, setup

log = logging.getLogger("cre.weather")
URL = "https://api.brightsky.dev/weather"
FIELDS = ["temperature", "precipitation", "wind_speed", "wind_gust_speed", "condition", "source_id"]
SCHEMA = {
    "station_key": pl.Utf8,
    "ts": pl.Datetime("us"),
    "temperature": pl.Float64,
    "precipitation": pl.Float64,
    "wind_speed": pl.Float64,
    "wind_gust_speed": pl.Float64,
    "condition": pl.Utf8,
    "source_id": pl.Int64,
}


def parse(payload: dict, station_key: str) -> pl.DataFrame:
    rows = []
    for w in payload.get("weather", []):
        ts = datetime.fromisoformat(w["timestamp"]).astimezone(UTC).replace(tzinfo=None)
        rows.append({"station_key": station_key, "ts": ts, **{k: w.get(k) for k in FIELDS}})
    df = pl.DataFrame(rows, schema=SCHEMA)
    # Bright Sky can return the boundary hour twice when ranges are chained -> keep one per key
    return df.unique(["station_key", "ts"], keep="first").sort("ts")


def fetch(client: httpx.Client, lat: float, lon: float, start: date, end: date, retries: int = 3) -> dict:
    params = {"lat": lat, "lon": lon, "date": start.isoformat(), "last_date": end.isoformat(), "tz": "UTC"}
    for attempt in range(retries + 1):
        try:
            r = client.get(URL, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            err = f"HTTP {r.status_code}"
        except httpx.TransportError as e:
            err = str(e)
        log_event(log, "weather_retry", level=logging.WARNING, attempt=attempt + 1, error=err)
        import time

        time.sleep(2**attempt)
    raise RuntimeError(f"Bright Sky failed for {lat},{lon} {start}..{end}")


def backfill(
    con: duckdb.DuckDBPyConnection,
    start: date,
    end: date,
    table: str = "raw.weather",
    client: httpx.Client | None = None,
) -> int:
    """Fetch [start, end] in monthly chunks for every hub; idempotent delete+insert per station/range."""
    client = client or httpx.Client()
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {table.split('.')[0]}")
    total = 0
    for st in settings()["stations"]:
        frames, cur = [], start
        while cur <= end:
            nxt = min(end, (cur.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1))
            frames.append(parse(fetch(client, st["lat"], st["lon"], cur, nxt + timedelta(days=1)), st["key"]))
            cur = nxt + timedelta(days=1)
        df = pl.concat(frames).unique(["station_key", "ts"], keep="first").sort("ts")
        res = validate_or_quarantine(df, Weather, f"{table}[{st['key']}]", max_invalid_frac=0.02)
        con.register("_tmp", res.valid.to_arrow())
        con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM _tmp LIMIT 0")
        con.execute(
            f"DELETE FROM {table} WHERE station_key = ? AND ts BETWEEN ? AND ?",
            [st["key"], res.valid["ts"].min(), res.valid["ts"].max()],
        )
        con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM _tmp")
        con.unregister("_tmp")
        total += res.valid.height
        log_event(log, "weather_loaded", station=st["key"], rows=res.valid.height, quarantined=res.quarantine.height)
    return total


def main() -> None:
    setup()
    months = settings()["data"]["months"]
    start = date.fromisoformat(months[0] + "-01") - timedelta(days=1)
    last = date.fromisoformat(months[-1] + "-01")
    end = (last + timedelta(days=32)).replace(day=1)
    con = duckdb.connect(str(path(settings()["data"]["duckdb_path"])))
    backfill(con, start, end)
    con.close()


if __name__ == "__main__":
    main()
