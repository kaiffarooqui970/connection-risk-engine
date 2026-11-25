from datetime import datetime

from cre.ingest.weather import parse


def test_parse_converts_to_utc_and_dedups_dst():
    payload = {
        "weather": [
            # DST fall-back: two different UTC instants that are both local 02:30 -> both kept (distinct UTC)
            {
                "timestamp": "2026-10-25T02:00:00+02:00",
                "temperature": 8.0,
                "precipitation": 0.0,
                "wind_speed": 3,
                "wind_gust_speed": 5,
                "condition": "dry",
                "source_id": 1,
            },
            {
                "timestamp": "2026-10-25T02:00:00+01:00",
                "temperature": 7.5,
                "precipitation": 0.1,
                "wind_speed": 3,
                "wind_gust_speed": 5,
                "condition": "rain",
                "source_id": 1,
            },
            # chained-range duplicate of the same instant -> dropped
            {
                "timestamp": "2026-10-25T01:00:00+00:00",
                "temperature": 7.5,
                "precipitation": 0.1,
                "wind_speed": 3,
                "wind_gust_speed": 5,
                "condition": "rain",
                "source_id": 1,
            },
        ]
    }
    df = parse(payload, "leipzig")
    assert df["ts"].to_list() == [datetime(2026, 10, 25, 0, 0), datetime(2026, 10, 25, 1, 0)]
