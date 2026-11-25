"""DB Timetables API v1 client: planned timetable (`plan`) + full change feed (`fchg`) for hub stations.

Design choices (WHY):
- Credentials only from env / .env (DB_CLIENT_ID, DB_API_KEY); missing creds -> a clean, logged skip,
  so a scheduled job without secrets is a no-op rather than a red CI run that people learn to ignore.
- A client-side token bucket keeps us under the marketplace limit (60 req/min) *before* the server says
  429. Retries are for genuinely transient failures (429/5xx/timeouts), with exponential backoff + jitter
  and `Retry-After` honoured.
- `plan` rows upsert by stop id (latest plan wins); `fchg` rows append as snapshots keyed by
  (id, fetched_at). Re-running the same fetch is idempotent; the history is what monitoring needs.
"""

from __future__ import annotations

import logging
import os
import random
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import httpx
import polars as pl

from cre.config import ROOT, eva_to_station, path, settings
from cre.contracts.schemas import LiveChange, LivePlan
from cre.contracts.validate import validate_or_quarantine
from cre.logging_utils import log_event, setup

log = logging.getLogger("cre.db_api")
BERLIN = ZoneInfo("Europe/Berlin")
RETRY_STATUS = {429, 500, 502, 503, 504}


class MissingCredentials(RuntimeError):
    pass


def load_dotenv(p: Path = ROOT / ".env") -> None:
    """Tiny .env reader (KEY=VALUE lines) to avoid a dependency; real env vars take precedence."""
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def credentials() -> tuple[str, str]:
    load_dotenv()
    cid, key = os.environ.get("DB_CLIENT_ID", ""), os.environ.get("DB_API_KEY", "")
    if not cid or not key:
        raise MissingCredentials("DB_CLIENT_ID / DB_API_KEY not set")
    return cid, key


class RateLimiter:
    """Token bucket: at most `rate` calls per `per` seconds, smooth (no bursts that trip the server)."""

    def __init__(self, rate: int, per: float = 60.0, clock=time.monotonic, sleep=time.sleep):
        self.capacity, self.per, self.clock, self.sleep = rate, per, clock, sleep
        self.tokens, self.last = float(rate), clock()

    def acquire(self) -> None:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.capacity / self.per)
        self.last = now
        if self.tokens < 1:
            wait = (1 - self.tokens) * self.per / self.capacity
            self.sleep(wait)
            self.tokens, self.last = 0.0, self.clock()
        else:
            self.tokens -= 1


class TimetablesClient:
    def __init__(
        self,
        client: httpx.Client | None = None,
        limiter: RateLimiter | None = None,
        max_retries: int = 4,
        sleep=time.sleep,
    ):
        cfg = settings()["live"]
        if client is None:
            cid, key = credentials()
            client = httpx.Client(
                base_url=cfg["base_url"],
                timeout=20.0,
                headers={"DB-Client-Id": cid, "DB-Api-Key": key, "Accept": "application/xml"},
            )
        self.http, self.max_retries, self.sleep = client, max_retries, sleep
        self.limiter = limiter or RateLimiter(cfg["max_requests_per_min"])

    def get(self, url: str) -> str | None:
        """GET with retries. Returns body, or None for 404 (DB returns 404 for empty hours)."""
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            try:
                r = self.http.get(url)
            except httpx.TransportError as e:
                status, retry_after, err = None, None, str(e)
            else:
                if r.status_code == 200:
                    return r.text
                if r.status_code == 404:
                    return None
                status, retry_after, err = r.status_code, r.headers.get("Retry-After"), r.text[:200]
                if status not in RETRY_STATUS:
                    raise httpx.HTTPStatusError(f"{status} for {url}: {err}", request=r.request, response=r)
            if attempt == self.max_retries:
                break
            wait = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt + random.random()
            log_event(
                log,
                "db_api_retry",
                level=logging.WARNING,
                url=url,
                status=status,
                attempt=attempt + 1,
                wait_s=round(wait, 1),
                error=err,
            )
            self.sleep(wait)
        raise RuntimeError(f"giving up on {url} after {self.max_retries + 1} attempts")

    def plan(self, eva: str, when: datetime) -> str | None:
        return self.get(f"/plan/{int(eva)}/{when:%y%m%d}/{when:%H}")

    def fchg(self, eva: str) -> str | None:
        return self.get(f"/fchg/{int(eva)}")


# ---------------- parsing ----------------


def _ts(v: str | None) -> datetime | None:
    """DB format YYMMddHHmm, local Berlin time -> naive local datetime (same convention as HF data)."""
    return datetime.strptime(v, "%y%m%d%H%M") if v else None


def parse_plan(xml: str, eva: str, fetched_at: datetime) -> pl.DataFrame:
    root = ET.fromstring(xml)
    rows = []
    for s in root.findall("s"):
        tl, ar, dp = s.find("tl"), s.find("ar"), s.find("dp")
        rows.append(
            {
                "id": s.get("id"),
                "eva": f"{int(s.get('eva') or eva):08d}",
                "train_type": tl.get("c") if tl is not None else None,
                "train_number": tl.get("n") if tl is not None else None,
                "line": (ar if ar is not None else dp).get("l") if (ar is not None or dp is not None) else None,
                "arr_planned": _ts(ar.get("pt")) if ar is not None else None,
                "dep_planned": _ts(dp.get("pt")) if dp is not None else None,
                "arr_platform": ar.get("pp") if ar is not None else None,
                "dep_platform": dp.get("pp") if dp is not None else None,
                "arr_path": ar.get("ppth") if ar is not None else None,
                "dep_path": dp.get("ppth") if dp is not None else None,
                "fetched_at": fetched_at,
            }
        )
    return pl.DataFrame(rows, schema=PLAN_SCHEMA)


def parse_fchg(xml: str, eva: str, fetched_at: datetime) -> pl.DataFrame:
    root = ET.fromstring(xml)
    rows = []
    for s in root.findall("s"):
        ar, dp = s.find("ar"), s.find("dp")
        if ar is None and dp is None:
            continue  # message-only entries (e.g. station notices) carry no times
        rows.append(
            {
                "id": s.get("id"),
                "eva": f"{int(s.get('eva') or eva):08d}",
                "arr_change": _ts(ar.get("ct")) if ar is not None else None,
                "dep_change": _ts(dp.get("ct")) if dp is not None else None,
                "arr_cancelled": ar is not None and ar.get("cs") == "c",
                "dep_cancelled": dp is not None and dp.get("cs") == "c",
                "arr_platform": ar.get("cp") if ar is not None else None,
                "dep_platform": dp.get("cp") if dp is not None else None,
                "fetched_at": fetched_at,
            }
        )
    return pl.DataFrame(rows, schema=CHANGE_SCHEMA)


PLAN_SCHEMA = {
    "id": pl.Utf8,
    "eva": pl.Utf8,
    "train_type": pl.Utf8,
    "train_number": pl.Utf8,
    "line": pl.Utf8,
    "arr_planned": pl.Datetime("us"),
    "dep_planned": pl.Datetime("us"),
    "arr_platform": pl.Utf8,
    "dep_platform": pl.Utf8,
    "arr_path": pl.Utf8,
    "dep_path": pl.Utf8,
    "fetched_at": pl.Datetime("us"),
}
CHANGE_SCHEMA = {
    "id": pl.Utf8,
    "eva": pl.Utf8,
    "arr_change": pl.Datetime("us"),
    "dep_change": pl.Datetime("us"),
    "arr_cancelled": pl.Boolean,
    "dep_cancelled": pl.Boolean,
    "arr_platform": pl.Utf8,
    "dep_platform": pl.Utf8,
    "fetched_at": pl.Datetime("us"),
}

# ---------------- storage ----------------

DDL = """
CREATE SCHEMA IF NOT EXISTS live;
CREATE SCHEMA IF NOT EXISTS quarantine;
CREATE TABLE IF NOT EXISTS live.plan (
  id VARCHAR PRIMARY KEY, eva VARCHAR, train_type VARCHAR, train_number VARCHAR, line VARCHAR,
  arr_planned TIMESTAMP, dep_planned TIMESTAMP, arr_platform VARCHAR, dep_platform VARCHAR,
  arr_path VARCHAR, dep_path VARCHAR, fetched_at TIMESTAMP);
CREATE TABLE IF NOT EXISTS live.changes (
  id VARCHAR, eva VARCHAR, arr_change TIMESTAMP, dep_change TIMESTAMP, arr_cancelled BOOLEAN,
  dep_cancelled BOOLEAN, arr_platform VARCHAR, dep_platform VARCHAR, fetched_at TIMESTAMP,
  PRIMARY KEY (id, fetched_at));
CREATE TABLE IF NOT EXISTS live.fetch_log (
  fetched_at TIMESTAMP, endpoint VARCHAR, eva VARCHAR, rows INTEGER, invalid INTEGER, status VARCHAR);
"""


def upsert(con: duckdb.DuckDBPyConnection, table: str, df: pl.DataFrame) -> None:
    if df.height == 0:
        return
    con.register("_tmp", df.to_arrow())
    con.execute(f"INSERT OR REPLACE INTO {table} BY NAME SELECT * FROM _tmp")
    con.unregister("_tmp")


def live_db(p: str | None = None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(p or path("data/live_store.duckdb")))
    con.execute(DDL)
    return con


def collect_once(
    client: TimetablesClient, con: duckdb.DuckDBPyConnection, now: datetime | None = None, hours_ahead: int = 2
) -> dict:
    """One collection cycle for all hub EVAs: plan for this hour + `hours_ahead`, and the full change feed."""
    now = now or datetime.now(BERLIN).replace(tzinfo=None)
    stats = {"plan": 0, "changes": 0, "invalid": 0, "errors": 0}
    for eva in eva_to_station():
        jobs = [("plan", now + timedelta(hours=h)) for h in range(hours_ahead + 1)] + [("fchg", now)]
        for kind, when in jobs:
            try:
                body = client.plan(eva, when) if kind == "plan" else client.fchg(eva)
                df = pl.DataFrame(schema=PLAN_SCHEMA if kind == "plan" else CHANGE_SCHEMA)
                if body:
                    df = (parse_plan if kind == "plan" else parse_fchg)(body, eva, now)
                schema, table = (LivePlan, "live.plan") if kind == "plan" else (LiveChange, "live.changes")
                res = validate_or_quarantine(df, schema, table, max_invalid_frac=0.05)
                upsert(con, table, res.valid)
                stats["plan" if kind == "plan" else "changes"] += res.valid.height
                stats["invalid"] += res.quarantine.height
                status, n, bad = "ok", res.valid.height, res.quarantine.height
            except Exception as e:  # noqa: BLE001 - one bad station/hour must not kill the whole cycle
                stats["errors"] += 1
                status, n, bad = f"error: {type(e).__name__}: {str(e)[:120]}", 0, 0
                log_event(log, "db_api_fetch_failed", level=logging.ERROR, eva=eva, endpoint=kind, error=str(e)[:300])
            con.execute("INSERT INTO live.fetch_log VALUES (?, ?, ?, ?, ?, ?)", [now, kind, eva, n, bad, status])
    log_event(log, "collect_cycle_done", **stats)
    return stats


def main() -> int:
    setup()
    try:
        client = TimetablesClient()
    except MissingCredentials as e:
        log_event(log, "collect_skipped", level=logging.WARNING, reason=str(e))
        return 0
    con = live_db()
    stats = collect_once(client, con)
    con.close()
    total = len(eva_to_station()) * 4
    return 1 if stats["errors"] > total // 2 else 0  # fail the job only if most calls failed


if __name__ == "__main__":
    raise SystemExit(main())
