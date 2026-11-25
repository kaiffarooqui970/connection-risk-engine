import numpy as np
import pandas as pd

from cre.monitor.drift import psi_categorical, psi_numeric
from cre.monitor.rules import DayStatus, actions, promote


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 20000))
    assert psi_numeric(ref, pd.Series(rng.normal(0, 1, 20000))) < 0.01
    assert psi_numeric(ref, pd.Series(rng.normal(1.5, 1, 20000))) > 0.25


def test_psi_counts_missing_as_its_own_bin():
    ref = pd.Series([1.0, 2.0, 3.0, np.nan] * 1000)
    cur = pd.Series([np.nan] * 4000)
    assert psi_numeric(ref, cur) > 1


def test_psi_categorical_new_level():
    assert psi_categorical(pd.Series(["a", "b"] * 500), pd.Series(["a", "c"] * 500)) > 0.25


def d(day, ece=0.01, saved=0.5, psi=0.02, fresh="FRESH", n=5000):
    return DayStatus(day, n, ece, saved, psi, fresh)


def test_actions_ok_and_fallback_and_rollback():
    assert actions([d("1"), d("2")]) == ["OK"]
    assert any(a.startswith("FALLBACK") for a in actions([d("1", fresh="STALE")]))
    assert any(a.startswith("ROLLBACK") for a in actions([d("1", saved=-1), d("2", saved=-1), d("3", saved=-0.1)]))
    assert any(a.startswith("RETRAIN") for a in actions([d(str(i), ece=0.08) for i in range(7)]))
    assert not any(a.startswith("RETRAIN") for a in actions([d("1", ece=0.5, n=10)]))  # too small to judge


def test_promotion_rules():
    assert promote(20, 0.40, 0.39, 0.02, 0.1) == (True, "promote")
    assert not promote(5, 0.40, 0.30, 0.02, 0.1)[0]
    assert not promote(20, 0.40, 0.399, 0.02, 0.1)[0]
    assert not promote(20, 0.40, 0.39, 0.06, 0.1)[0]
    assert not promote(20, 0.40, 0.39, 0.02, -0.1)[0]
