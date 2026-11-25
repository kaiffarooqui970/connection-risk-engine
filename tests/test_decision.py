import numpy as np

from cre.decision.policy import commit_decision, realized_costs, recommend


def test_threshold_is_cost_ratio():
    p = np.array([0.49, 0.51])
    assert commit_decision(p, np.array([20, 20]), 2.0, 1.0).tolist() == [False, True]
    assert commit_decision(np.array([0.99]), np.array([20]), 1.0, 1.0).tolist() == [False]  # pure minutes: never


def test_pure_minutes_commit_never_beats_keep():
    """DESIGN §1.1: with equal weights, keep <= commit in every realized outcome."""
    rng = np.random.default_rng(0)
    missed, lost, e = rng.integers(0, 2, 1000), rng.integers(0, 120, 1000), rng.integers(0, 120, 1000)
    keep, commit = realized_costs(missed, lost, e, 1.0, 1.0)
    assert np.all(keep <= commit)


def test_failed_alternative_is_charged_as_unplanned():
    keep, commit = realized_costs([1], [50], [20], 2.0, 1.0)
    assert keep[0] == 100 and commit[0] == 20 + 2 * 30


def test_recommend_no_alternative_keeps():
    assert recommend(0.9, 240).action == "keep"
    assert recommend(0.9, 15).action == "take_alternative"
