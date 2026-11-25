"""Feed deliberately broken data and assert the documented handling (docs/DATA_CONTRACTS.md)."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from cre.contracts.schemas import RawStops, Weather
from cre.contracts.validate import (
    ContractViolation,
    check_freshness,
    validate_or_quarantine,
)


def test_valid_frame_passes(good_stops):
    res = validate_or_quarantine(good_stops, RawStops, "t")
    assert res.valid.height == 3 and res.quarantine.height == 0


def test_missing_column_fails_hard(good_stops):
    with pytest.raises(ContractViolation, match="schema-level|column"):
        validate_or_quarantine(good_stops.drop("arrival_is_canceled"), RawStops, "t")


def test_wrong_dtype_fails_hard(good_stops):
    bad = good_stops.with_columns(pl.col("arrival_planned_time").dt.cast_time_unit("ns"))
    with pytest.raises(ContractViolation, match="schema-level"):
        validate_or_quarantine(bad, RawStops, "t")


def _big(good_stops, n=300):
    """Replicate the valid frame with unique ids so threshold maths is meaningful."""
    return pl.concat(
        [
            good_stops.with_columns(
                id=pl.format("{}-26100508{}-{}", pl.lit(i), pl.lit("00"), pl.col("train_line_station_num"))
            )
            for i in range(n)
        ]
    )


@pytest.mark.parametrize(
    "breaker, reason",
    [
        (
            lambda d: d.with_columns(
                arrival_change_time=pl.when(pl.col("train_line_station_num") == 2)
                .then(pl.col("arrival_planned_time") - timedelta(minutes=90))
                .otherwise(pl.col("arrival_change_time"))
            ),
            "arrival delay",
        ),
        (
            lambda d: d.with_columns(
                eva=pl.when(pl.col("train_line_station_num") == 2).then(pl.lit("8010205")).otherwise(pl.col("eva"))
            ),
            "str_matches",
        ),
        (
            lambda d: d.with_columns(
                arrival_planned_time=pl.when(pl.col("train_line_station_num") == 1)
                .then(None)
                .otherwise(pl.col("arrival_planned_time")),
                departure_planned_time=pl.when(pl.col("train_line_station_num") == 1)
                .then(None)
                .otherwise(pl.col("departure_planned_time")),
            ),
            "neither planned",
        ),
    ],
)
def test_bad_rows_are_quarantined_with_reason(good_stops, breaker, reason):
    df = _big(good_stops)
    broken = pl.concat(
        [df, breaker(good_stops.with_columns(id=pl.format("999-2610050800-{}", pl.col("train_line_station_num"))))]
    )
    res = validate_or_quarantine(broken, RawStops, "t", max_invalid_frac=0.05)
    assert res.quarantine.height == 1
    assert reason in res.quarantine["failed_checks"][0]
    assert res.valid.height == broken.height - 1


def test_cancelled_stop_with_absurd_time_is_valid(good_stops):
    """Cancelled stops carry stale change times (EDA problem #2): must not be quarantined."""
    df = good_stops.with_columns(
        arrival_is_canceled=pl.col("train_line_station_num") == 2,
        arrival_change_time=pl.when(pl.col("train_line_station_num") == 2)
        .then(pl.col("arrival_planned_time") - timedelta(minutes=300))
        .otherwise(pl.col("arrival_change_time")),
    )
    assert validate_or_quarantine(df, RawStops, "t").quarantine.height == 0


def test_duplicate_ids_quarantined(good_stops):
    df = pl.concat([_big(good_stops), good_stops.head(1)])
    res = validate_or_quarantine(df, RawStops, "t", max_invalid_frac=0.05)
    assert res.quarantine.height == 2  # both copies: we can't know which is right
    assert all("unique" in r for r in res.quarantine["failed_checks"])


def test_too_many_invalid_rows_fails_the_batch(good_stops):
    df = good_stops.with_columns(eva=pl.lit("bad"))
    with pytest.raises(ContractViolation, match="threshold"):
        validate_or_quarantine(df, RawStops, "t", max_invalid_frac=0.01)


def _weather(**over):
    base = {
        "station_key": ["leipzig", "leipzig"],
        "ts": [datetime(2026, 10, 1, 0), datetime(2026, 10, 1, 1)],
        "temperature": [11.0, 10.5],
        "precipitation": [0.0, 0.2],
        "wind_speed": [3.0, 4.0],
        "wind_gust_speed": [9.0, 10.0],
        "condition": ["dry", "rain"],
        "source_id": [1, 1],
    }
    base.update(over)
    return pl.DataFrame(base, schema_overrides={"ts": pl.Datetime("us")})


@pytest.mark.parametrize(
    "over, reason",
    [
        ({"precipitation": [0.0, -1.0]}, "greater_than_or_equal_to|in_range"),
        ({"temperature": [11.0, 80.0]}, "in_range|less_than"),
        ({"station_key": ["leipzig", "hamburg"]}, "isin"),
        ({"ts": [datetime(2026, 10, 1, 0), datetime(2026, 10, 1, 0)]}, "duplicate"),
    ],
)
def test_weather_contract(over, reason):
    res = validate_or_quarantine(_weather(**over), Weather, "w", max_invalid_frac=1.0)
    assert res.quarantine.height >= 1
    assert any(any(r in fc for r in reason.split("|")) for fc in res.quarantine["failed_checks"])


def test_freshness_states():
    now = datetime(2026, 11, 25, 12, 0)
    assert check_freshness("x", now - timedelta(minutes=30), now, timedelta(hours=2)).status == "FRESH"
    assert check_freshness("x", now - timedelta(hours=3), now, timedelta(hours=2)).status == "STALE"
    assert check_freshness("x", None, now, timedelta(hours=2)).status == "MISSING"
