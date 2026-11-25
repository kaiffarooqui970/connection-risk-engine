"""Concrete, testable operating thresholds (docs/OPERATIONS.md is the human-readable version).

Every threshold is a constant here so an alert in CI and the doc can't drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

# freshness
LIVE_MAX_AGE_MIN = 120  # change feed older than this -> fallback model + reduced-confidence flag
WEATHER_MAX_AGE_MIN = 180

# calibration / value, on matured predictions (outcome known), rolling 7 days
ECE_WARN, ECE_RETRAIN = 0.03, 0.05
MIN_DAILY_N = 500  # don't judge calibration on tiny samples
VALUE_NEG_DAYS_ROLLBACK = 3  # 3 consecutive days with the model losing to always-keep -> roll back

# drift
PSI_WARN, PSI_RETRAIN = 0.10, 0.25
PSI_KEY_FEATURES = ["a_known_delay", "naive_slack", "planned_transfer_min", "hist_miss_rate", "precip_mm_1h"]

# shadow -> promotion
SHADOW_MIN_DAYS = 14
SHADOW_MIN_LOGLOSS_GAIN = 0.002  # candidate must beat champion on log loss by at least this...
SHADOW_MAX_ECE = 0.03  # ...and stay calibrated...
SHADOW_MIN_VALUE_GAIN = 0.0  # ...and not lose decision value


@dataclass
class DayStatus:
    day: str
    n: int
    ece: float | None
    saved_per_risky: float | None
    max_key_psi: float
    freshness: str  # FRESH / STALE / MISSING


def actions(history: list[DayStatus]) -> list[str]:
    """Return the operational actions implied by the most recent days (newest last)."""
    out = []
    if not history:
        return ["NO_DATA: no monitored days yet"]
    today = history[-1]
    if today.freshness != "FRESH":
        out.append(f"FALLBACK: live data {today.freshness}; serving historical model with reduced confidence")
    judged = [d for d in history[-7:] if d.n >= MIN_DAILY_N and d.ece is not None]
    if judged:
        ece7 = sum(d.ece * d.n for d in judged) / sum(d.n for d in judged)
        if ece7 > ECE_RETRAIN:
            out.append(f"RETRAIN: 7-day ECE {ece7:.3f} > {ECE_RETRAIN}")
        elif ece7 > ECE_WARN:
            out.append(f"WARN: 7-day ECE {ece7:.3f} > {ECE_WARN}")
    tail = history[-VALUE_NEG_DAYS_ROLLBACK:]
    if len(tail) == VALUE_NEG_DAYS_ROLLBACK and all(
        d.saved_per_risky is not None and d.saved_per_risky < 0 for d in tail
    ):
        out.append(f"ROLLBACK: model lost to always-keep {VALUE_NEG_DAYS_ROLLBACK} days running")
    if today.max_key_psi > PSI_RETRAIN:
        out.append(f"RETRAIN: key-feature PSI {today.max_key_psi:.2f} > {PSI_RETRAIN}")
    elif today.max_key_psi > PSI_WARN:
        out.append(f"WARN: key-feature PSI {today.max_key_psi:.2f} > {PSI_WARN}")
    return out or ["OK"]


def promote(
    days: int, champ_logloss: float, cand_logloss: float, cand_ece: float, value_gain: float
) -> tuple[bool, str]:
    if days < SHADOW_MIN_DAYS:
        return False, f"shadow period {days}d < {SHADOW_MIN_DAYS}d"
    if champ_logloss - cand_logloss < SHADOW_MIN_LOGLOSS_GAIN:
        return False, f"log-loss gain {champ_logloss - cand_logloss:.4f} < {SHADOW_MIN_LOGLOSS_GAIN}"
    if cand_ece > SHADOW_MAX_ECE:
        return False, f"candidate ECE {cand_ece:.3f} > {SHADOW_MAX_ECE}"
    if value_gain < SHADOW_MIN_VALUE_GAIN:
        return False, f"decision value change {value_gain:+.3f} < 0"
    return True, "promote"
