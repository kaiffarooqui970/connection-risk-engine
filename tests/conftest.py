from datetime import datetime

import polars as pl
import pytest


@pytest.fixture
def good_stops() -> pl.DataFrame:
    """A tiny valid RawStops frame: one run, 3 stops."""
    t = lambda h, m: datetime(2026, 10, 5, h, m)
    return pl.DataFrame(
        {
            "id": ["111-2610050800-1", "111-2610050800-2", "111-2610050800-3"],
            "eva": ["08010101", "08010205", "08011160"],
            "station_name": ["Erfurt Hbf", "Leipzig Hbf", "Berlin Hauptbahnhof"],
            "train_type": ["ICE", "ICE", "ICE"],
            "train_number": ["1502"] * 3,
            "line_number": [None] * 3,
            "final_destination_station": ["Berlin Hauptbahnhof"] * 3,
            "train_line_ride_id": ["111"] * 3,
            "train_line_station_num": [1, 2, 3],
            "arrival_planned_time": [None, t(8, 40), t(10, 0)],
            "arrival_change_time": [None, t(8, 45), t(10, 5)],
            "departure_planned_time": [t(8, 0), t(8, 45), None],
            "departure_change_time": [t(8, 2), t(8, 49), None],
            "arrival_is_canceled": [False] * 3,
            "departure_is_canceled": [False] * 3,
            "is_additional_stop": [False] * 3,
            "is_replacement_train": [False] * 3,
        },
        schema_overrides={
            "arrival_planned_time": pl.Datetime("us"),
            "arrival_change_time": pl.Datetime("us"),
            "departure_planned_time": pl.Datetime("us"),
            "departure_change_time": pl.Datetime("us"),
            "line_number": pl.Utf8,
        },
    )
