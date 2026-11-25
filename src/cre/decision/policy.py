"""Cost model and recommendation (Phase 7). See docs/DESIGN.md §1.

At prediction time we know P(miss) and Ê: the planned extra time of the best alternative Ĉ that is still
reachable given A's current delay (Ĉ = earliest-arriving planned departure to T, after B, that leaves at least
MCT after A's expected arrival). If B is missed, the passenger ends up on Ĉ too, so

    E[cost | keep]   = w_late * P * Ê        (unplanned waiting, only if missed)
    cost   | commit  = w_plan * Ê            (planned waiting, certain)

=> commit  iff  P > w_plan / w_late  (for Ê > 0). The decision is a threshold on a *calibrated* probability,
which is why calibration is a release guardrail and not a nice-to-have.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CAP = 240


def expected_costs(p, e_hat, w_late: float, w_plan: float, l_keep=None):
    l_keep = e_hat if l_keep is None else l_keep
    return w_late * np.asarray(p) * np.asarray(l_keep), w_plan * np.asarray(e_hat)


def commit_decision(p, e_hat, w_late: float, w_plan: float, l_keep=None) -> np.ndarray:
    keep, commit = expected_costs(p, e_hat, w_late, w_plan, l_keep)
    return commit < keep  # ties -> keep the plan (don't disturb the passenger for nothing)


def realized_costs(missed, lost, e_hat, w_late: float, w_plan: float):
    """Generalized cost with ACTUAL outcomes. `lost` = actual extra minutes at T if B is not taken (>= 0).

    keep:   0 if made, else every lost minute is unplanned.
    commit: minutes up to the promised Ê are planned; anything beyond (the promised alternative failed) is unplanned.
    """
    missed, lost, e_hat = (np.asarray(x, dtype=float) for x in (missed, lost, e_hat))
    keep = np.where(missed > 0, w_late * lost, 0.0)
    commit = w_plan * np.minimum(lost, e_hat) + w_late * np.maximum(0.0, lost - e_hat)
    return keep, commit


@dataclass
class Recommendation:
    p_miss: float
    expected_lost_min: float  # P * Ê : expected minutes lost if you keep the plan
    alt_extra_min: float  # Ê
    action: str  # "keep" | "take_alternative"
    reason: str
    confidence: str  # "full" | "reduced (fallback model)"


def recommend(
    p: float, e_hat: float, w_late: float = 2.0, w_plan: float = 1.0, fallback: bool = False
) -> Recommendation:
    e = float(min(CAP, max(0.0, e_hat)))
    commit = bool(commit_decision(np.array([p]), np.array([e]), w_late, w_plan)[0])
    thr = w_plan / w_late
    if e >= CAP:
        reason = "No alternative to your next stop within 4 h, so keep the plan."
        commit = False
    elif commit:
        reason = f"P(miss) {p:.0%} is above {thr:.0%}: planning on the alternative (+{e:.0f} min) beats gambling."
    else:
        reason = f"P(miss) {p:.0%} is below {thr:.0%}: the expected loss ({p * e:.0f} min) doesn't justify switching."
    return Recommendation(
        p,
        p * e,
        e,
        "take_alternative" if commit else "keep",
        reason,
        "reduced (fallback model)" if fallback else "full",
    )
