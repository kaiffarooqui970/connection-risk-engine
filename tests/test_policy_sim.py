"""Phase 7/8 evaluation invariants on synthetic data (no warehouse needed)."""

import numpy as np
import pandas as pd
import pytest

from cre.evaluate.failure import slice_stats
from cre.evaluate.policy_sim import evaluate, paired_diff, policies, tune_rules


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 4000
    slack = rng.integers(-15, 30, n)
    p_true = 1 / (1 + np.exp(0.4 * (slack - 3)))
    missed = rng.random(n) < p_true
    e_hat = rng.integers(10, 60, n).astype(float)
    lost = np.where(missed, e_hat + rng.integers(-5, 10, n).clip(min=0), 0.0)
    p = np.clip(p_true + rng.normal(0, 0.05, n), 0, 1)
    return pd.DataFrame(
        {
            "service_date": pd.to_datetime("2026-10-01") + pd.to_timedelta(rng.integers(0, 20, n), unit="D"),
            "naive_slack": slack,
            "planned_transfer_min": rng.integers(5, 31, n),
            "missed": missed,
            "lost": lost,
            "e_hat": e_hat,
            "p_lgb_cal": p,
            "p_logreg": p,
            "p_hist": np.full(n, missed.mean()),
            "p_rule_slack": p,
        }
    )


def test_pure_minutes_tuned_rules_never_commit(frame):
    # w_late == w_plan: committing is weakly dominated (DESIGN §1.1), so tuning must pick "never"
    rx = tune_rules(frame, 1.0, 1.0)
    assert rx["rule_slack"] == -1e9 and rx["rule_transfer"] == -1e9


def test_oracle_bounds_every_policy_and_keep_is_zero(frame):
    rx = tune_rules(frame, 2.0, 1.0)
    res = evaluate(frame, policies(frame, 2.0, 1.0, rx), 2.0, 1.0, n_boot=20)
    assert res["always_keep"]["all"]["saved_per_conn"] == 0.0
    for name, r in res.items():
        for scope in ("risky", "all"):
            assert r[scope]["saved_per_conn"] <= res["oracle"][scope]["saved_per_conn"] + 1e-9, name
    assert res["oracle"]["all"]["unnecessary_switch_rate"] == 0.0
    # an informative model must beat always-keep on this synthetic data
    assert res["model"]["risky"]["saved_per_conn"] > 0


def test_paired_diff_of_identical_policies_is_zero_and_antisymmetric(frame):
    dec = policies(frame, 2.0, 1.0, tune_rules(frame, 2.0, 1.0))
    pt, lo, hi = paired_diff(frame, dec["model"], dec["model"], 2.0, 1.0, n_boot=20)
    assert pt == lo == hi == 0.0
    ab = paired_diff(frame, dec["model"], dec["always_keep"], 2.0, 1.0, n_boot=20)[0]
    ba = paired_diff(frame, dec["always_keep"], dec["model"], 2.0, 1.0, n_boot=20)[0]
    assert ab == pytest.approx(-ba)


def test_slice_stats_consistent_with_headline(frame):
    s = slice_stats(frame, 2.0, 1.0, rule_x=4)
    assert s["n"] == len(frame)
    assert s["miss_rate"] == pytest.approx(frame.missed.mean())
    res = evaluate(frame, policies(frame, 2.0, 1.0, {"rule_slack": 4, "rule_transfer": -1e9}), 2.0, 1.0, n_boot=5)
    assert s["model_saved_risky"] == pytest.approx(res["model"]["risky"]["saved_per_conn"])
    assert s["rule_slack_saved_risky"] == pytest.approx(res["rule_slack"]["risky"]["saved_per_conn"])
