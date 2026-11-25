"""Live path on fixture XML: collect -> connections -> features -> freshness."""

from datetime import datetime, timedelta

import duckdb

from cre.ingest.db_api import collect_once, live_db
from cre.monitor.live import freshness, live_connections, live_features
from tests.test_db_api import NOW, _client, _router


def _store(tmp_path):
    p = str(tmp_path / "live_store.duckdb")
    con = live_db(p)
    collect_once(_client(_router(), retries=0), con, now=NOW)
    con.close()
    return duckdb.connect(p, read_only=True)


def test_live_connection_and_skewed_features(tmp_path):
    con = _store(tmp_path)
    # the fixture router serves the same XML for every hub; plan upserts by stop id, so all stops end up
    # under the last-polled EVA. We only care that exactly one valid connection is built.
    conns = live_connections(con, NOW, lead_min=30)  # A arriving 12:30-12:40 -> the ICE at 12:30
    assert len(conns) == 1
    row = conns.iloc[0]
    assert row.a_train_number == "1502" and row.b_train_number == "16511" and row.planned_transfer_min == 12
    f = live_features(conns, NOW, 30).iloc[0]
    assert f.a_known_delay == 11  # DB forecast 12:41 vs 12:30
    assert f.b_started == 0  # B cancelled -> no forecast time
    assert f.naive_slack == 1


def test_freshness_goes_stale(tmp_path):
    con = _store(tmp_path)
    assert freshness(con, NOW + timedelta(minutes=30)) == "FRESH"
    assert freshness(con, NOW + timedelta(hours=3)) == "STALE"


def test_freshness_missing_when_never_fetched(tmp_path):
    con = live_db(str(tmp_path / "live_store.duckdb"))
    assert freshness(con, datetime(2026, 11, 25)) == "MISSING"
