"""Phase 6 orchestrator: baselines + LightGBM (3 seeds) + calibration, metrics with day-bootstrap CIs.

Outputs: `ml.predictions` (every model's P(miss) for val + test), `models_store/` artifacts,
`reports/model_metrics.json`, and reliability / importance figures.
"""

from __future__ import annotations

import json
import logging
import pickle
import time

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cre.config import path, settings  # noqa: E402
from cre.evaluate.metrics import bootstrap_by_day, ece, reliability, suite  # noqa: E402
from cre.features.build import FEATURES  # noqa: E402
from cre.logging_utils import log_event, setup  # noqa: E402
from cre.models import baselines, calibration, miss_prob  # noqa: E402
from cre.models.data import apply_categories, categories, load  # noqa: E402

log = logging.getLogger("cre.train")
SEEDS = [0, 1, 2]
STORE = path("models_store/x").parent
FIG = path("reports/figures/x").parent


def segment_metrics(df: pd.DataFrame, cols: list[str]) -> dict:
    out = {}
    for key in ["lead_min", "station_key", "pair"]:
        out[key] = {}
        for val, g in df.groupby(key, observed=True):
            out[key][str(val)] = {c: suite(g["y"].to_numpy(), g[c].to_numpy()) for c in cols}
    return out


def main() -> None:
    setup()
    t0 = time.time()
    df = load()
    tr, va, te = (df[df.split == s].copy() for s in ("train", "val", "test"))
    cats = categories(tr)
    va, te = apply_categories(va, cats), apply_categories(te, cats)
    log_event(
        log,
        "data_loaded",
        train=len(tr),
        val=len(va),
        test=len(te),
        base_rates=[round(x["y"].mean(), 4) for x in (tr, va, te)],
    )

    # ---------------- baselines ----------------
    preds = {}
    for name, model in {"rule_transfer": baselines.rule_transfer(), "rule_slack": baselines.rule_slack()}.items():
        model.fit(tr)
        preds[name] = (model.predict(va), model.predict(te))
    preds["hist"] = (baselines.hist_predict(va), baselines.hist_predict(te))
    lr = baselines.logreg().fit(baselines.to_lr_frame(tr), tr["y"])
    preds["logreg"] = tuple(lr.predict_proba(baselines.to_lr_frame(x))[:, 1] for x in (va, te))
    log_event(log, "baselines_done", secs=round(time.time() - t0))

    # ---------------- LightGBM ----------------
    best = miss_prob.tune(tr, va, n_trials=20)
    boosters = [miss_prob.fit(tr, va, best, seed=s) for s in SEEDS]
    log_event(log, "lgb_fitted", best_iterations=[b.best_iteration for b in boosters], secs=round(time.time() - t0))
    raw_seed = {
        s: (
            b.predict(va[FEATURES], num_iteration=b.best_iteration),
            b.predict(te[FEATURES], num_iteration=b.best_iteration),
        )
        for s, b in zip(SEEDS, boosters, strict=True)
    }
    preds["lgb_raw"] = (
        np.mean([v[0] for v in raw_seed.values()], axis=0),
        np.mean([v[1] for v in raw_seed.values()], axis=0),
    )

    # ---------------- calibration ----------------
    vday = va["service_date"].dt.date.to_numpy()
    cal_cmp = calibration.compare(preds["lgb_raw"][0], va["y"].to_numpy(), vday)
    chosen = calibration.choose(cal_cmp)
    cal = calibration.fit(chosen, preds["lgb_raw"][0], va["y"].to_numpy())
    preds["lgb_cal"] = (cal.predict(preds["lgb_raw"][0]), cal.predict(preds["lgb_raw"][1]))
    log_event(
        log,
        "calibration_chosen",
        method=chosen,
        cv={k: {m: round(v[m], 5) for m in ("log_loss", "ece", "brier")} for k, v in cal_cmp.items()},
    )

    # per-seed (each seed calibrated by its own calibrator, same method) -> seed variability on test
    per_seed = {}
    for s, (pv, pt) in raw_seed.items():
        c = calibration.fit(chosen, pv, va["y"].to_numpy())
        per_seed[s] = suite(te["y"].to_numpy(), c.predict(pt))

    # ---------------- metrics ----------------
    y_te, day_te = te["y"].to_numpy(), te["service_date"].dt.date.to_numpy()
    test_metrics = {k: suite(y_te, v[1]) for k, v in preds.items()}
    val_metrics = {k: suite(va["y"].to_numpy(), v[0]) for k, v in preds.items()}

    def brier(y, p):
        return float(np.mean((p - y) ** 2))

    def auc(y, p):
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y, p))

    def logloss(y, p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    cis = {}
    for k in ("lgb_cal", "logreg", "rule_slack", "hist"):
        cis[k] = {
            name: bootstrap_by_day(y_te, preds[k][1], day_te, fn, n_boot=300)
            for name, fn in {"brier": brier, "log_loss": logloss, "roc_auc": auc, "ece": ece}.items()
        }
    # paired difference CI: does LightGBM beat logistic regression? (same resampled days for both)
    cis["brier_diff_lgb_minus_logreg"] = bootstrap_by_day(
        y_te,
        np.stack([preds["lgb_cal"][1], preds["logreg"][1]], axis=1),
        day_te,
        lambda y, pp: brier(y, pp[:, 0]) - brier(y, pp[:, 1]),
        n_boot=300,
    )

    te_seg = te.assign(
        pair=te["a_category"].astype(str) + "->" + te["b_category"].astype(str), **{k: v[1] for k, v in preds.items()}
    )
    seg = segment_metrics(te_seg, ["lgb_cal", "logreg", "rule_slack"])

    imp = pd.Series(boosters[0].feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    report = {
        "rows": {"train": len(tr), "val": len(va), "test": len(te)},
        "best_params": best,
        "best_iterations": [b.best_iteration for b in boosters],
        "calibration_cv_on_val": cal_cmp,
        "calibration_chosen": chosen,
        "val": val_metrics,
        "test": test_metrics,
        "test_ci95_by_day": cis,
        "test_per_seed_calibrated": per_seed,
        "test_segments": seg,
        "feature_importance_gain_seed0": (imp / imp.sum()).round(4).to_dict(),
    }
    path("reports/model_metrics.json").write_text(json.dumps(report, indent=2, default=float))

    # ---------------- persist ----------------
    for s, b in zip(SEEDS, boosters, strict=True):
        b.save_model(str(STORE / f"lgb_seed{s}.txt"), num_iteration=b.best_iteration)
    with open(STORE / "calibrator.pkl", "wb") as f:
        pickle.dump(cal, f)
    with open(STORE / "logreg.pkl", "wb") as f:
        pickle.dump(lr, f)
    (STORE / "meta.json").write_text(
        json.dumps(
            {
                "features": FEATURES,
                "categories": cats,
                "calibration": chosen,
                "seeds": SEEDS,
                "params": best,
                "train_end": settings()["split"]["train_end"],
                "val_end": settings()["split"]["val_end"],
                "rule_slack_p": baselines.rule_slack().fit(tr).p,
            },
            indent=2,
            default=str,
        )
    )

    out = pd.concat(
        [
            pd.DataFrame(
                {
                    "conn_id": x["conn_id"].to_numpy(),
                    "lead_min": x["lead_min"].to_numpy().astype(int),
                    "split": split,
                    **{f"p_{k}": v[i] for k, v in preds.items()},
                }
            )
            for i, (x, split) in enumerate([(va, "val"), (te, "test")])
        ]
    )
    con = duckdb.connect(str(path(settings()["data"]["duckdb_path"])))
    con.register("_p", out)
    con.execute("CREATE OR REPLACE TABLE ml.predictions AS SELECT * FROM _p")
    con.close()

    # ---------------- figures ----------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for k, lab in [
        ("lgb_raw", "LightGBM raw"),
        ("lgb_cal", f"LightGBM + {chosen}"),
        ("logreg", "Logistic reg."),
        ("hist", "Historical rate"),
    ]:
        mp, my, _ = reliability(y_te, preds[k][1])
        ax[0].plot(mp, my, marker="o", ms=4, label=f"{lab} (ECE {ece(y_te, preds[k][1]):.3f})")
    ax[0].plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax[0].set(title="Reliability on test (Oct 2026)", xlabel="predicted P(miss)", ylabel="observed miss rate")
    ax[0].legend(fontsize=8)
    ax[1].hist(preds["lgb_cal"][1], bins=50, color="#3b6ea5")
    ax[1].set(title="Distribution of calibrated P(miss), test", xlabel="P(miss)", yscale="log")
    fig.tight_layout()
    fig.savefig(FIG / "reliability_test.png", dpi=130)
    fig, ax = plt.subplots(figsize=(6, 6))
    top = (imp / imp.sum()).head(20)[::-1]
    ax.barh(top.index, top.values, color="#3b6ea5")
    ax.set(title="LightGBM feature importance (gain share, seed 0)")
    fig.tight_layout()
    fig.savefig(FIG / "feature_importance.png", dpi=130)
    log_event(
        log,
        "train_done",
        secs=round(time.time() - t0),
        test={
            k: {m: round(v.get(m, float("nan")), 4) for m in ("brier", "log_loss", "roc_auc", "ece")}
            for k, v in test_metrics.items()
        },
    )


if __name__ == "__main__":
    main()
