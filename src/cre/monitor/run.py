"""Daily monitoring: calibration, decision value, feature drift (PSI), freshness -> operational actions.

Modes
- `replay` (default locally): walk the held-out test month day by day as if it were live. Champion = the
  trained model; shadow candidate = same recipe retrained on data through Sep 22 and calibrated on
  Sep 23-30 (i.e. "fresher data"), scored silently next to the champion and judged by `rules.promote`.
- `live` (scheduled `monitor.yml`): reads the live store. Without DB API credentials it has no data and
  writes a NO_DATA report (exit 0): visible, not noisy.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cre.config import path, settings  # noqa: E402
from cre.decision.policy import commit_decision, realized_costs  # noqa: E402
from cre.evaluate.metrics import ece, suite  # noqa: E402
from cre.evaluate.policy_sim import load_frame  # noqa: E402
from cre.features.build import CATEGORICAL, NUMERIC  # noqa: E402
from cre.logging_utils import log_event, setup  # noqa: E402
from cre.monitor import rules  # noqa: E402
from cre.monitor.drift import psi_table  # noqa: E402

log = logging.getLogger("cre.monitor")
OUT = path("reports/monitoring/x").parent


def day_value(g: pd.DataFrame, pcol: str, w_late: float, w_plan: float) -> float | None:
    risky = g["naive_slack"] < settings()["connections"]["risky_slack_min"]
    if risky.sum() == 0:
        return None
    keep, commit = realized_costs(g["missed"], g["lost"], g["e_hat"], w_late, w_plan)
    d = commit_decision(g[pcol].to_numpy(), g["e_hat"].to_numpy(), w_late, w_plan)
    return float((keep - np.where(d, commit, keep))[risky.to_numpy()].mean())


def train_candidate(feats: pd.DataFrame) -> pd.Series:
    """Shadow candidate: champion recipe on fresher data (train + val through Sep 22), calibrated on Sep 23-30."""

    from cre.models import calibration, miss_prob
    from cre.models.data import apply_categories, categories

    cut = pd.Timestamp("2026-09-22")
    tr = feats[(feats.split == "train") | ((feats.split == "val") & (feats.service_date <= cut))]
    ca = feats[(feats.split == "val") & (feats.service_date > cut)]
    te = feats[feats.split == "test"]
    cats = categories(tr)
    ca, te = apply_categories(ca, cats), apply_categories(te, cats)
    params = json.loads(path("models_store/meta.json").read_text())["params"]
    b = miss_prob.fit(tr, ca, params, seed=0)
    cal = calibration.fit(
        "isotonic", b.predict(ca[miss_prob.FEATURES], num_iteration=b.best_iteration), ca["y"].to_numpy()
    )
    p = cal.predict(b.predict(te[miss_prob.FEATURES], num_iteration=b.best_iteration))
    b.save_model(str(path("models_store/candidate_lgb.txt")), num_iteration=b.best_iteration)
    return pd.Series(p, index=pd.MultiIndex.from_arrays([te.conn_id, te.lead_min.astype(int)]))


def replay() -> dict:
    from cre.models.data import load

    cfg = settings()["decision"]
    w_late, w_plan = cfg["w_late"], cfg["w_plan"]
    con = duckdb.connect(str(path(settings()["data"]["duckdb_path"])), read_only=True)
    feats = load(con)
    sim = load_frame(con)
    sim = sim[sim.split == "test"].copy()
    cand = train_candidate(feats)
    sim["p_candidate"] = cand.reindex(pd.MultiIndex.from_arrays([sim.conn_id, sim.lead_min.astype(int)])).to_numpy()
    ref = feats[feats.split == "train"].sample(300_000, random_state=0)
    cur_all = feats[feats.split == "test"]
    counts = sim.groupby("service_date").size()
    median_n = counts.median()

    days, history = [], []
    for day, g in sim.groupby("service_date"):
        cur = cur_all[cur_all.service_date == pd.Timestamp(day)]
        psi = psi_table(ref, cur, NUMERIC, CATEGORICAL)
        y = g["missed"].to_numpy(float)
        fresh = "FRESH" if len(g) >= 0.5 * median_n else "STALE"  # replay proxy: a data gap = stale feed
        st = rules.DayStatus(
            str(day)[:10],
            len(g),
            ece(y, g["p_lgb_cal"].to_numpy()),
            day_value(g, "p_lgb_cal", w_late, w_plan),
            float(psi[rules.PSI_KEY_FEATURES].max()),
            fresh,
        )
        history.append(st)
        days.append(
            {
                "day": st.day,
                "n": st.n,
                "freshness": fresh,
                "miss_rate": float(y.mean()),
                "mean_p": float(g["p_lgb_cal"].mean()),
                "ece": st.ece,
                "brier": suite(y, g["p_lgb_cal"].to_numpy())["brier"],
                "saved_per_risky": st.saved_per_risky,
                "max_key_psi": st.max_key_psi,
                "top_psi_feature": psi.index[0],
                "top_psi": float(psi.iloc[0]),
                "cand_ece": ece(y, g["p_candidate"].to_numpy()),
                "cand_log_loss": suite(y, g["p_candidate"].to_numpy())["log_loss"],
                "champ_log_loss": suite(y, g["p_lgb_cal"].to_numpy())["log_loss"],
                "cand_saved_per_risky": day_value(g, "p_candidate", w_late, w_plan),
                "actions": rules.actions(history),
            }
        )
    d = pd.DataFrame(days)
    d.to_csv(OUT / "replay_daily.csv", index=False)

    # shadow verdict after the minimum shadow period, on pooled rows of those days
    shadow_days = d.day.iloc[: max(rules.SHADOW_MIN_DAYS, 1)]
    pool = sim[sim.service_date.astype(str).str[:10].isin(shadow_days)]
    y = pool["missed"].to_numpy(float)
    champ, candm = suite(y, pool["p_lgb_cal"].to_numpy()), suite(y, pool["p_candidate"].to_numpy())
    gain = (day_value(pool, "p_candidate", w_late, w_plan) or 0) - (day_value(pool, "p_lgb_cal", w_late, w_plan) or 0)
    ok, why = rules.promote(len(shadow_days), champ["log_loss"], candm["log_loss"], candm["ece"], gain)
    summary = {
        "mode": "replay",
        "days": len(d),
        "actions_by_day": {r.day: r.actions for r in d.itertuples()},
        "days_with_non_ok_actions": int((d.actions.map(lambda a: a != ["OK"])).sum()),
        "shadow": {
            "days": len(shadow_days),
            "champion": champ,
            "candidate": candm,
            "value_gain": gain,
            "promote": ok,
            "reason": why,
        },
        "psi_test_month_vs_train": psi_table(ref, cur_all, NUMERIC, CATEGORICAL).round(4).to_dict(),
    }
    (OUT / "replay_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    plot(d)
    return summary


def plot(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    x = pd.to_datetime(d.day)
    ax[0].plot(x, d.miss_rate, label="observed miss rate", marker=".")
    ax[0].plot(x, d.mean_p, label="mean predicted P", marker=".")
    ax[0].set(title="Predicted vs actual, daily")
    ax[0].legend(fontsize=8)
    ax[1].plot(x, d.ece, label="champion ECE", marker=".")
    ax[1].plot(x, d.cand_ece, label="shadow candidate ECE", marker=".", alpha=0.7)
    ax[1].axhline(rules.ECE_WARN, ls="--", c="orange", lw=1, label="warn")
    ax[1].axhline(rules.ECE_RETRAIN, ls="--", c="red", lw=1, label="retrain")
    ax[1].set(title="Daily calibration error")
    ax[1].legend(fontsize=8)
    ax[2].plot(x, d.max_key_psi, marker=".", label="max PSI, key features")
    ax[2].axhline(rules.PSI_WARN, ls="--", c="orange", lw=1)
    ax[2].axhline(rules.PSI_RETRAIN, ls="--", c="red", lw=1)
    ax[2].set(title="Feature drift vs training")
    for a in ax:
        a.tick_params(axis="x", labelrotation=45, labelsize=7)
    fig.tight_layout()
    fig.savefig(path("reports/figures/monitoring_replay.png"), dpi=130)


def live() -> dict:
    from cre.monitor.live import freshness

    p = path("data/live_store.duckdb")
    now = datetime.now()
    if not p.exists():
        summary = {"mode": "live", "status": "NO_DATA", "actions": ["NO_DATA: live store not found"]}
    else:
        con = duckdb.connect(str(p), read_only=True)
        fresh = freshness(con, now)
        n = con.execute("SELECT count(*) FROM live.changes WHERE fetched_at >= ? - INTERVAL 1 DAY", [now]).fetchall()[
            0
        ][0]
        errors = con.execute(
            "SELECT count(*) FROM live.fetch_log WHERE fetched_at >= ? - INTERVAL 1 DAY AND status LIKE 'error%'", [now]
        ).fetchall()[0][0]
        acts = rules.actions([rules.DayStatus(now.date().isoformat(), n, None, None, 0.0, fresh)])
        summary = {
            "mode": "live",
            "status": fresh,
            "changes_last_24h": n,
            "fetch_errors_last_24h": errors,
            "actions": acts,
        }
    (OUT / "live_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["replay", "live"], default="replay")
    s = replay() if ap.parse_args().mode == "replay" else live()
    log_event(log, "monitor_done", **{k: v for k, v in s.items() if k in ("mode", "status", "days", "shadow")})


if __name__ == "__main__":
    main()
