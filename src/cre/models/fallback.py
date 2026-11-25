"""Fallback model for when live data is stale/missing (Phase 9).

Uses only features that don't depend on the real-time feed: timetable, calendar, weather and historical
rates. It is trained and evaluated exactly like the main model, so "reduced confidence" has a number
attached (its test Brier / ECE / decision value), not just a label.
"""

from __future__ import annotations

import json
import pickle

import numpy as np

from cre.config import path
from cre.evaluate.metrics import suite
from cre.features.build import CATEGORICAL, NUMERIC
from cre.logging_utils import log_event, setup
from cre.models import calibration, miss_prob
from cre.models.data import apply_categories, categories, load

REALTIME = {
    "a_known_delay",
    "a_started",
    "a_delay_trend_1",
    "a_delay_trend_2",
    "a_min_since_obs",
    "b_known_delay",
    "b_started",
    "naive_slack",
    "a_planned_min_remaining",
    "a_stops_remaining",
}
STATIC_NUMERIC = [f for f in NUMERIC if f not in REALTIME]
STATIC_FEATURES = STATIC_NUMERIC + CATEGORICAL


def main() -> None:
    import lightgbm as lgb

    setup()
    df = load()
    tr, va, te = (df[df.split == s].copy() for s in ("train", "val", "test"))
    cats = categories(tr)
    va, te = apply_categories(va, cats), apply_categories(te, cats)
    params = json.loads((path("models_store/meta.json")).read_text())["params"]
    p = {**miss_prob.BASE, **params, "seed": 0}
    dtr = lgb.Dataset(tr[STATIC_FEATURES], label=tr["y"], categorical_feature=CATEGORICAL)
    dva = lgb.Dataset(va[STATIC_FEATURES], label=va["y"], categorical_feature=CATEGORICAL, reference=dtr)
    b = lgb.train(p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    pv, pt = (b.predict(x[STATIC_FEATURES], num_iteration=b.best_iteration) for x in (va, te))
    method = calibration.choose(calibration.compare(pv, va["y"].to_numpy(), va["service_date"].dt.date.to_numpy()))
    cal = calibration.fit(method, pv, va["y"].to_numpy())
    b.save_model(str(path("models_store/fallback_lgb.txt")), num_iteration=b.best_iteration)
    with open(path("models_store/fallback_calibrator.pkl"), "wb") as f:
        pickle.dump(cal, f)
    m = {
        "features": STATIC_FEATURES,
        "calibration": method,
        "test": suite(te["y"].to_numpy(), cal.predict(pt)),
        "test_by_lead": {
            int(ld): suite(g["y"].to_numpy(), cal.predict(pt[np.flatnonzero(te["lead_min"] == ld)]))
            for ld, g in te.groupby("lead_min")
        },
    }
    path("reports/fallback_metrics.json").write_text(json.dumps(m, indent=2, default=float))
    log_event(__import__("logging").getLogger("cre.fallback"), "fallback_trained", test=m["test"])


if __name__ == "__main__":
    main()
