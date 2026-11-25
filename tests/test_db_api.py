import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from cre.ingest import db_api
from cre.ingest.db_api import (
    RateLimiter,
    TimetablesClient,
    collect_once,
    live_db,
    parse_fchg,
    parse_plan,
)

FIX = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 11, 25, 12, 5)


def test_parse_plan_fixture():
    df = parse_plan((FIX / "plan_8010205.xml").read_text(), "08010205", NOW)
    assert df.height == 3
    ice = df.filter(df["train_number"] == "1502").row(0, named=True)
    assert ice["eva"] == "08010205"
    assert ice["arr_planned"] == datetime(2026, 11, 25, 12, 30) and ice["dep_planned"] == datetime(2026, 11, 25, 12, 38)
    re = df.filter(df["train_number"] == "16511").row(0, named=True)
    assert re["arr_planned"] is None and re["line"] == "RE13"


def test_parse_fchg_fixture_cancellation_and_delay():
    df = parse_fchg((FIX / "fchg_8010205.xml").read_text(), "08010205", NOW)
    assert df.height == 2  # message-only entry skipped
    ice = df.filter(df["id"].str.starts_with("-4810")).row(0, named=True)
    assert ice["arr_change"] == datetime(2026, 11, 25, 12, 41) and not ice["arr_cancelled"]
    re = df.filter(df["id"].str.starts_with("7215")).row(0, named=True)
    assert re["dep_cancelled"] and re["dep_change"] is None


def test_malformed_xml_raises():
    with pytest.raises(ET.ParseError):
        parse_plan("<timetable><s id='x'>", "08010205", NOW)


class FakeClock:
    def __init__(self):
        self.t, self.slept = 0.0, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def test_rate_limiter_never_exceeds_rate():
    clk = FakeClock()
    rl = RateLimiter(60, 60.0, clock=clk, sleep=clk.sleep)
    for _ in range(150):
        rl.acquire()
    # 60 burst tokens, then 1/sec -> 90 more calls need ~90 s
    assert 88 <= clk.t <= 92


def _client(handler, retries=3):
    http = httpx.Client(base_url="https://x", transport=httpx.MockTransport(handler))
    return TimetablesClient(client=http, limiter=RateLimiter(10_000), max_retries=retries, sleep=lambda s: None)


def test_retries_on_429_then_succeeds():
    calls = []

    def h(req):
        calls.append(1)
        return (
            httpx.Response(429, headers={"Retry-After": "1"})
            if len(calls) < 3
            else httpx.Response(200, text="<timetable/>")
        )

    assert _client(h).get("/fchg/8010205") == "<timetable/>"
    assert len(calls) == 3


def test_gives_up_after_max_retries():
    with pytest.raises(RuntimeError, match="giving up"):
        _client(lambda r: httpx.Response(503), retries=2).get("/fchg/1")


def test_auth_error_is_not_retried():
    calls = []

    def h(req):
        calls.append(1)
        return httpx.Response(401, text="unauthorized")

    with pytest.raises(httpx.HTTPStatusError):
        _client(h).get("/fchg/1")
    assert len(calls) == 1


def test_404_means_empty_hour():
    assert _client(lambda r: httpx.Response(404)).get("/plan/1/261125/12") is None


def _router(fail_eva=None):
    plan, fchg = (FIX / "plan_8010205.xml").read_text(), (FIX / "fchg_8010205.xml").read_text()

    def h(req):
        if fail_eva and f"/{int(fail_eva)}" in req.url.path:
            return httpx.Response(500)
        return httpx.Response(200, text=plan if "/plan/" in req.url.path else fchg)

    return h


def test_collect_is_idempotent(tmp_path):
    con = live_db(str(tmp_path / "live_store.duckdb"))
    c = _client(_router(), retries=0)
    collect_once(c, con, now=NOW)
    n1 = (
        con.execute("select count(*) from live.plan").fetchall()[0][0],
        con.execute("select count(*) from live.changes").fetchall()[0][0],
    )
    collect_once(c, con, now=NOW)
    n2 = (
        con.execute("select count(*) from live.plan").fetchall()[0][0],
        con.execute("select count(*) from live.changes").fetchall()[0][0],
    )
    assert n1 == n2 and n1[0] > 0


def test_one_failing_station_does_not_kill_cycle(tmp_path):
    con = live_db(str(tmp_path / "live_store.duckdb"))
    stats = collect_once(_client(_router(fail_eva="08010101"), retries=0), con, now=NOW)
    assert stats["errors"] == 4 and stats["plan"] > 0
    assert con.execute("select count(*) from live.fetch_log where status like 'error%'").fetchall()[0][0] == 4


def test_missing_credentials_is_clean_skip(monkeypatch, tmp_path):
    monkeypatch.delenv("DB_CLIENT_ID", raising=False)
    monkeypatch.delenv("DB_API_KEY", raising=False)
    monkeypatch.setattr(db_api, "load_dotenv", lambda *a, **k: None)
    assert db_api.main() == 0
