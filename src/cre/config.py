"""Load `configs/settings.toml` once. One config file keeps scope and assumptions auditable."""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@lru_cache
def settings() -> dict:
    with open(ROOT / "configs" / "settings.toml", "rb") as f:
        return tomllib.load(f)


def path(rel: str) -> Path:
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def eva_to_station() -> dict[str, str]:
    """Map every EVA number to a station key; multi-level hubs collapse to one station."""
    return {eva: s["key"] for s in settings()["stations"] for eva in s["evas"]}


def mct_by_station() -> dict[str, int]:
    return {s["key"]: s["mct_min"] for s in settings()["stations"]}


def write_dbt_seed() -> None:
    """Generate dbt/seeds/hub_evas.csv from settings.toml so station config has ONE source of truth."""
    lines = ["eva,station_key,mct_min"]
    lines += [f"{e},{s['key']},{s['mct_min']}" for s in settings()["stations"] for e in s["evas"]]
    (ROOT / "dbt" / "seeds" / "hub_evas.csv").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    write_dbt_seed()
