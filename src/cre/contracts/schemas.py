"""pandera data contracts, one per table that crosses a boundary (source -> warehouse -> model).

Ranges are deliberately *physical* limits, not statistical ones: a train 6 h late is rare but real
and must survive (it's exactly the tail we care about); a train 2 h *early* is a data error.
"""

from __future__ import annotations

from datetime import datetime

import pandera.polars as pa
import polars as pl

STOP_ID_RE = r"^-?\d+-\d{10}-\d+$"
MAX_DELAY_MIN = 1440  # a day; beyond that it's a different service, not a delay
MAX_EARLY_MIN = 30  # >30 min early never happens in practice at a stop -> data error


def _delay_ok(planned: str, actual: str, cancelled: str):
    def check(data) -> pl.LazyFrame:
        d = (pl.col(actual) - pl.col(planned)).dt.total_minutes()
        ok = (
            pl.col(planned).is_null()
            | pl.col(actual).is_null()
            | pl.col(cancelled)
            | d.is_between(-MAX_EARLY_MIN, MAX_DELAY_MIN)
        )
        return data.lazyframe.select(ok)

    return check


class RawStops(pa.DataFrameModel):
    """HF `monthly_processed_data`, one row per (run, stop)."""

    id: str = pa.Field(unique=True, str_matches=STOP_ID_RE)
    eva: str = pa.Field(str_matches=r"^\d{8}$")
    station_name: str
    train_type: str = pa.Field(nullable=True)  # 0.07% null in source; tolerated, mapped to 'unknown'
    train_number: str = pa.Field(nullable=True)
    line_number: str = pa.Field(nullable=True)
    final_destination_station: str = pa.Field(nullable=True)
    train_line_ride_id: str
    train_line_station_num: int = pa.Field(ge=1, le=500)
    arrival_planned_time: datetime = pa.Field(nullable=True)
    arrival_change_time: datetime = pa.Field(nullable=True)
    departure_planned_time: datetime = pa.Field(nullable=True)
    departure_change_time: datetime = pa.Field(nullable=True)
    arrival_is_canceled: bool
    departure_is_canceled: bool
    is_additional_stop: bool = pa.Field(nullable=True)
    is_replacement_train: bool = pa.Field(nullable=True)

    @pa.dataframe_check(error="stop has neither planned arrival nor planned departure")
    def has_planned_time(cls, data) -> pl.LazyFrame:
        return data.lazyframe.select(
            pl.col("arrival_planned_time").is_not_null() | pl.col("departure_planned_time").is_not_null()
        )

    @pa.dataframe_check(error="planned departure before planned arrival")
    def planned_order(cls, data) -> pl.LazyFrame:
        return data.lazyframe.select(
            pl.col("arrival_planned_time").is_null()
            | pl.col("departure_planned_time").is_null()
            | (pl.col("departure_planned_time") >= pl.col("arrival_planned_time"))
        )

    @pa.dataframe_check(error="arrival delay outside physical range")
    def arr_delay_range(cls, data) -> pl.LazyFrame:
        return _delay_ok("arrival_planned_time", "arrival_change_time", "arrival_is_canceled")(data)

    @pa.dataframe_check(error="departure delay outside physical range")
    def dep_delay_range(cls, data) -> pl.LazyFrame:
        return _delay_ok("departure_planned_time", "departure_change_time", "departure_is_canceled")(data)

    class Config:
        strict = False  # extra columns allowed (source adds columns over time); missing ones fail
        coerce = False


class Weather(pa.DataFrameModel):
    """Bright Sky hourly observations, one row per (station_key, ts)."""

    station_key: str = pa.Field(isin=["leipzig", "erfurt", "halle", "berlin"])
    ts: datetime
    temperature: float = pa.Field(nullable=True, in_range={"min_value": -40, "max_value": 50})
    precipitation: float = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 150})
    wind_speed: float = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 250})
    wind_gust_speed: float = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 300})
    condition: str = pa.Field(nullable=True)
    source_id: int = pa.Field(nullable=True)

    @pa.dataframe_check(error="duplicate (station_key, ts)")
    def unique_key(cls, data) -> pl.LazyFrame:
        return data.lazyframe.select(~pl.struct("station_key", "ts").is_duplicated())

    class Config:
        strict = False


class LivePlan(pa.DataFrameModel):
    """DB Timetables `plan` parsed rows, one row per stop id."""

    id: str = pa.Field(unique=True)
    eva: str = pa.Field(str_matches=r"^\d{8}$")
    train_type: str = pa.Field(nullable=True)
    train_number: str = pa.Field(nullable=True)
    arr_planned: datetime = pa.Field(nullable=True)
    dep_planned: datetime = pa.Field(nullable=True)
    fetched_at: datetime

    @pa.dataframe_check(error="stop has neither planned arrival nor planned departure")
    def has_planned_time(cls, data) -> pl.LazyFrame:
        return data.lazyframe.select(pl.col("arr_planned").is_not_null() | pl.col("dep_planned").is_not_null())

    class Config:
        strict = False


class LiveChange(pa.DataFrameModel):
    """DB Timetables `fchg` parsed rows, one row per (stop id, fetch)."""

    id: str
    eva: str = pa.Field(str_matches=r"^\d{8}$")
    arr_change: datetime = pa.Field(nullable=True)
    dep_change: datetime = pa.Field(nullable=True)
    arr_cancelled: bool
    dep_cancelled: bool
    fetched_at: datetime

    @pa.dataframe_check(error="duplicate (id, fetched_at)")
    def unique_key(cls, data) -> pl.LazyFrame:
        return data.lazyframe.select(~pl.struct("id", "fetched_at").is_duplicated())

    class Config:
        strict = False


class Connections(pa.DataFrameModel):
    """ml.connections joined with ml.labels: one row per candidate connection (Phase 4)."""

    conn_id: str = pa.Field(unique=True)
    station_key: str = pa.Field(isin=["leipzig", "erfurt", "halle", "berlin"])
    mct_min: int = pa.Field(ge=1, le=20)
    planned_transfer_min: int = pa.Field(ge=1, le=60)
    outcome: str = pa.Field(isin=["made", "too_short", "b_cancelled", "a_cancelled"])
    missed: bool
    lost_min_if_missed: float = pa.Field(nullable=True, ge=0, le=240)

    @pa.dataframe_check(error="planned transfer below station MCT")
    def above_mct(cls, data) -> pl.LazyFrame:
        return data.lazyframe.select(pl.col("planned_transfer_min") >= pl.col("mct_min"))

    @pa.dataframe_check(error="missed must equal outcome != made")
    def label_consistent(cls, data) -> pl.LazyFrame:
        return data.lazyframe.select(pl.col("missed") == (pl.col("outcome") != "made"))

    @pa.dataframe_check(error="lost time must be null iff A was cancelled")
    def lost_null_iff_a_cancelled(cls, data) -> pl.LazyFrame:
        return data.lazyframe.select(pl.col("lost_min_if_missed").is_null() == (pl.col("outcome") == "a_cancelled"))

    class Config:
        strict = False
