"""Label logic on hand-built scenarios. Each scenario lives on its own day so they can't pair up."""

from datetime import datetime, timedelta

import duckdb
import polars as pl
import pytest

from cre.connections.build import build

HUB = "08010205"  # leipzig, MCT 5


def stop(
    run,
    seq,
    ident,
    arr_p=None,
    arr_a=None,
    dep_p=None,
    dep_a=None,
    *,
    day,
    cat="regional",
    num=None,
    line=None,
    arr_c=False,
    dep_c=False,
):
    d0 = datetime(2026, 10, 1) + timedelta(days=day)
    t = lambda hm: None if hm is None else d0 + timedelta(hours=int(hm[:2]), minutes=int(hm[3:]))
    hub = ident == "leipzig"
    return {
        "stop_key": f"{run}-{seq}",
        "run_id": run,
        "stop_seq": seq,
        "eva": HUB if hub else f"0000{seq:04d}",
        "station_key": "leipzig" if hub else "other",
        "station_name": "Leipzig Hbf" if hub else ident,
        "train_type": {"regional": "RE", "long_distance": "ICE", "sbahn": "S"}[cat],
        "train_category": cat,
        "train_number": num or run,
        "line_number": line,
        "arr_planned": t(arr_p),
        "arr_actual": None if arr_c else t(arr_a),
        "dep_planned": t(dep_p),
        "dep_actual": None if dep_c else t(dep_a),
        "arr_cancelled": arr_c,
        "dep_cancelled": dep_c,
        "is_hub": hub,
        "first_seq": 1,
        "run_progress": 0.5,
    }


def scenario(
    day, a_arr_a, b_dep_a="10:20", *, a_c=False, b_c=False, fallback=True, b_num=None, b_next="Dresden", b_dep_p="10:20"
):
    rows = [
        stop(f"A{day}", 1, "Halle", dep_p="09:30", dep_a="09:30", day=day),
        stop(f"A{day}", 2, "leipzig", arr_p="10:00", arr_a=a_arr_a, day=day, arr_c=a_c),
        stop(f"B{day}", 1, "leipzig", dep_p=b_dep_p, dep_a=b_dep_a, day=day, dep_c=b_c, num=b_num),
        stop(f"B{day}", 2, b_next, arr_p="11:00", arr_a="11:00", day=day),
    ]
    if fallback:
        rows += [
            stop(f"F{day}", 1, "leipzig", dep_p="10:50", dep_a="10:50", day=day),
            stop(f"F{day}", 2, b_next, arr_p="11:40", arr_a="11:45", day=day),
        ]
    return rows


@pytest.fixture(scope="module")
def built():
    rows = (
        scenario(0, "10:02")  # made: 18 min actual transfer
        + scenario(1, "10:17")  # too short: 3 < 5
        + scenario(2, "10:17", b_dep_a="10:26")  # B held/late -> 9 min -> made
        + scenario(3, "10:02", b_c=True)  # B cancelled
        + scenario(4, "10:02", a_c=True)  # A cancelled -> lost undefined
        + scenario(5, "10:17", fallback=False)  # missed, no fallback -> censored at cap
        + scenario(6, "10:02", b_num="A6")  # same train number -> excluded
        + scenario(7, "10:02", b_next="Halle")  # backtracking -> excluded
        + scenario(8, "10:02", b_dep_p="10:03", b_dep_a="10:03")  # planned 3 min < MCT 5 -> excluded
    )
    df = pl.DataFrame(rows)
    con = duckdb.connect()
    con.execute("CREATE SCHEMA marts; CREATE SCHEMA ref")
    con.register("_m", df.to_arrow())
    con.execute("CREATE TABLE marts.fct_stop_events AS SELECT * FROM _m")
    con.execute("CREATE TABLE ref.hub_evas AS SELECT '08010205' eva, 'leipzig' station_key, 5 mct_min")
    build(con)
    return con.sql("""SELECT c.a_run_id, c.b_run_id, c.outcome, c.missed, l.lost_min_if_missed AS L,
                             l.lost_censored, l.l_hat_planned_min AS lhat
                      FROM ml.connections c JOIN ml.labels l USING (conn_id)
                      WHERE c.b_run_id LIKE 'B%'""").pl()


def row(built, day):
    r = built.filter(pl.col("a_run_id") == f"A{day}")
    return None if r.height == 0 else r.row(0, named=True)


def test_made(built):
    r = row(built, 0)
    assert r["outcome"] == "made" and not r["missed"]
    assert r["lhat"] == 40  # alternative F arrives 11:40 planned vs B 11:00 planned


def test_too_short_missed_and_lost_time_uses_actuals(built):
    r = row(built, 1)
    assert r["outcome"] == "too_short" and r["missed"]
    assert r["L"] == 45  # fallback actual 11:45 - B actual 11:00


def test_held_b_saves_connection(built):
    assert row(built, 2)["outcome"] == "made"


def test_b_cancelled(built):
    r = row(built, 3)
    assert r["outcome"] == "b_cancelled" and r["missed"] and r["L"] == 45


def test_a_cancelled_has_undefined_lost_time(built):
    r = row(built, 4)
    assert r["outcome"] == "a_cancelled" and r["missed"] and r["L"] is None


def test_no_fallback_is_censored(built):
    r = row(built, 5)
    assert r["missed"] and r["L"] == 240 and r["lost_censored"]


@pytest.mark.parametrize("day", [6, 7, 8])
def test_excluded_pairs(built, day):
    assert row(built, day) is None
