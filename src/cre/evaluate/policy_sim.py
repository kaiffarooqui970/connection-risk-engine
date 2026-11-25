"""Phase 7: simulate decision policies on the held-out month with ACTUAL outcomes.

Headline metric: generalized minutes saved per *risky* connection vs "always keep", 95% CI by day bootstrap.
Risky (pre-registered in DESIGN): naive slack at prediction time < 10 min.
Connections whose A was cancelled are excluded (the passenger never reaches S; no decision at S helps).

Rule thresholds are tuned on VALIDATION. Probability policies are NOT tuned: the threshold follows
from the cost model (P > w_plan / w_late). If a probability policy needed a tuned threshold, its
calibration would be wrong.
"""

from __future__ import annotations

import json
import logging

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cre.config import path, settings  # noqa: E402
from cre.connections.build import fallback_sql, make_events, make_reach  # noqa: E402
from cre.decision.policy import CAP, commit_decision, realized_costs  # noqa: E402
from cre.evaluate.metrics import bootstrap_by_day  # noqa: E402
from cre.logging_utils import log_event, setup  # noqa: E402

log = logging.getLogger("cre.policy_sim")
FIG = path("reports/figures/x").parent
PROB_POLICIES = {"historical_rate": "p_hist", "logreg": "p_logreg", "model": "p_lgb_cal"}


def load_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per (connection, lead) in val+test with predictions, Ê and actual outcomes."""
    return con.sql(
        """
    WITH ehat AS (
        SELECT f.conn_id, f.lead_min,
               min(a.alt_t_arr_planned) FILTER (
                   WHERE a.alt_dep_planned >= c.a_arr_planned + (f.a_known_delay + c.mct_min) * INTERVAL 1 MINUTE
               ) AS c_arr_planned,
               any_value(c.b_t_arr_planned) AS b_t_arr_planned
        FROM ml.features f
        JOIN ml.connections c USING (conn_id)
        LEFT JOIN ml.alternatives a USING (conn_id)
        WHERE f.split IN ('val', 'test')
        GROUP BY 1, 2
    )
    SELECT f.conn_id, f.lead_min, f.split, f.service_date, f.station_key, f.a_category, f.b_category,
           f.naive_slack, f.planned_transfer_min, f.a_known_delay, f.hour, f.missed, f.outcome,
           f.lost_min_if_missed AS lost, f.lost_censored, c.mct_min, c.actual_transfer_min,
           c.a_arr_cancelled, c.b_dep_cancelled,
           least({cap}, greatest(0, coalesce(date_diff('minute', e.b_t_arr_planned, e.c_arr_planned), {cap}))) AS e_hat,
           p.* EXCLUDE (conn_id, lead_min, split)
    FROM ml.features f
    JOIN ml.connections c USING (conn_id)
    JOIN ehat e USING (conn_id, lead_min)
    JOIN ml.predictions p USING (conn_id, lead_min)
    WHERE f.outcome <> 'a_cancelled'
    """.replace("{cap}", str(CAP))
    ).df()


def policies(df: pd.DataFrame, w_late: float, w_plan: float, rule_x: dict[str, float]) -> dict[str, np.ndarray]:
    out = {
        "always_keep": np.zeros(len(df), bool),
        "rule_transfer": (df["planned_transfer_min"] < rule_x["rule_transfer"]).to_numpy(),
        "rule_slack": (df["naive_slack"] < rule_x["rule_slack"]).to_numpy(),
    }
    for name, col in PROB_POLICIES.items():
        out[name] = commit_decision(df[col].to_numpy(), df["e_hat"].to_numpy(), w_late, w_plan)
    return out


def tune_rules(val: pd.DataFrame, w_late: float, w_plan: float) -> dict[str, float]:
    """Pick each rule's threshold to maximize savings over ALL validation connections."""
    keep, commit = realized_costs(val["missed"], val["lost"], val["e_hat"], w_late, w_plan)
    best = {}
    for name, col, grid in [
        ("rule_transfer", "planned_transfer_min", range(5, 31)),
        ("rule_slack", "naive_slack", range(-30, 21)),
    ]:
        x = val[col].to_numpy()
        saves = {g: float(np.sum(np.where(x < g, keep - commit, 0.0))) for g in grid}
        g_best = max(saves, key=saves.get)
        best[name] = g_best if saves[g_best] > 0 else -1e9  # never committing beats every threshold
    return best


def evaluate(
    df: pd.DataFrame, decisions: dict[str, np.ndarray], w_late: float, w_plan: float, n_boot: int = 300
) -> dict:
    keep, commit = realized_costs(df["missed"], df["lost"], df["e_hat"], w_late, w_plan)
    risky = (df["naive_slack"] < settings()["connections"]["risky_slack_min"]).to_numpy()
    day = df["service_date"].astype(str).to_numpy()
    oracle = commit < keep
    res = {}
    for name, d in {**decisions, "oracle": oracle}.items():
        cost = np.where(d, commit, keep)
        saving = keep - cost  # vs always keep
        unnecessary = d & (commit > keep)  # switched, keeping would have been cheaper
        missed_warn = ~d & (keep > commit)  # kept, switching would have been cheaper
        r = {}
        for scope, m in (("risky", risky), ("all", np.ones_like(risky))):
            pt, lo, hi = bootstrap_by_day(
                np.zeros(m.sum()), saving[m], day[m], lambda _y, s: float(np.mean(s)), n_boot=n_boot
            )
            r[scope] = {
                "n": int(m.sum()),
                "saved_per_conn": pt,
                "ci95": [lo, hi],
                "commit_rate": float(d[m].mean()),
                "unnecessary_switch_rate": float(unnecessary[m].mean()),
                "unnecessary_switch_cost_per_conn": float(np.where(unnecessary, commit - keep, 0)[m].mean()),
                "missed_warning_rate": float(missed_warn[m].mean()),
                "missed_warning_cost_per_conn": float(np.where(missed_warn, keep - commit, 0)[m].mean()),
            }
        r["commit_rate_non_risky"] = float(d[~risky].mean())
        res[name] = r
    return res


def paired_diff(df, a: np.ndarray, b: np.ndarray, w_late, w_plan, n_boot=300) -> list[float]:
    """CI of (savings of policy a - savings of policy b) per risky connection, same resampled days."""
    keep, commit = realized_costs(df["missed"], df["lost"], df["e_hat"], w_late, w_plan)
    risky = (df["naive_slack"] < settings()["connections"]["risky_slack_min"]).to_numpy()
    diff = (np.where(b, commit, keep) - np.where(a, commit, keep))[risky]
    day = df["service_date"].astype(str).to_numpy()[risky]
    return list(bootstrap_by_day(np.zeros(len(diff)), diff, day, lambda _y, s: float(np.mean(s)), n_boot=n_boot))


def mct_sensitivity(con, test: pd.DataFrame, rule_x, w_late, w_plan) -> dict:
    """If the true MCT differs from our assumption by +/-2 min: relabel test with the shifted MCT
    (missed and actual lost time), keep the model as trained, and re-score the policies."""
    make_events(con)
    make_reach(con)
    out = {}
    for off in (-2, 0, 2):
        fb = con.sql(fallback_sql(CAP, off, "c.service_date > DATE '" + settings()["split"]["val_end"] + "'")).df()
        t = test.drop(columns=["lost"]).merge(fb, on="conn_id", how="left")
        b_ref = con.sql("SELECT conn_id, b_ref_arr FROM ml.connections").df()
        t = t.merge(b_ref, on="conn_id", how="left")
        t["lost"] = ((t["fallback_arr"] - t["b_ref_arr"]).dt.total_seconds() / 60).clip(0, CAP).fillna(CAP)
        t["missed"] = t["b_dep_cancelled"] | (t["actual_transfer_min"] < t["mct_min"] + off)
        r = evaluate(t, policies(t, w_late, w_plan, rule_x), w_late, w_plan, n_boot=100)
        out[f"mct{off:+d}"] = {
            k: {
                "saved": v["risky"]["saved_per_conn"],
                "ci95": v["risky"]["ci95"],
                "miss_rate": float(t["missed"].mean()),
            }
            for k, v in r.items()
        }
    return out


def main() -> None:
    setup()
    cfg = settings()["decision"]
    con = duckdb.connect(str(path(settings()["data"]["duckdb_path"])))
    df = load_frame(con)
    val, test = df[df.split == "val"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
    log_event(log, "sim_frame", val=len(val), test=len(test), test_risky=int((test.naive_slack < 10).sum()))

    w_late, w_plan = cfg["w_late"], cfg["w_plan"]
    rule_x = tune_rules(val, w_late, w_plan)
    headline = evaluate(test, policies(test, w_late, w_plan, rule_x), w_late, w_plan, n_boot=500)
    dec = policies(test, w_late, w_plan, rule_x)
    best_base = max(
        ["rule_transfer", "rule_slack", "historical_rate", "logreg"],
        key=lambda k: headline[k]["risky"]["saved_per_conn"],
    )
    diff = paired_diff(test, dec["model"], dec[best_base], w_late, w_plan, n_boot=500)
    by_lead = {
        int(ld): evaluate(
            g.reset_index(drop=True),
            policies(g.reset_index(drop=True), w_late, w_plan, rule_x),
            w_late,
            w_plan,
            n_boot=200,
        )
        for ld, g in test.groupby("lead_min")
    }

    weights = {}
    for w in cfg["w_late_grid"]:
        rx = tune_rules(val, w, w_plan)
        r = evaluate(test, policies(test, w, w_plan, rx), w, w_plan, n_boot=200)
        weights[str(w)] = {
            "rule_x": rx,
            **{
                k: {
                    "saved": v["risky"]["saved_per_conn"],
                    "ci95": v["risky"]["ci95"],
                    "commit_rate": v["risky"]["commit_rate"],
                }
                for k, v in r.items()
            },
        }
    mct = mct_sensitivity(con, test, rule_x, w_late, w_plan)

    # pure minutes, model policy, as a sanity statement (should be exactly 0: never commits)
    report = {
        "w_late": w_late,
        "w_plan": w_plan,
        "rule_thresholds_tuned_on_val": rule_x,
        "test_headline": headline,
        "best_baseline": best_base,
        "model_minus_best_baseline_per_risky_ci95": diff,
        "by_lead": by_lead,
        "sensitivity_w_late": weights,
        "sensitivity_mct_offset": mct,
    }
    path("reports/policy_metrics.json").write_text(json.dumps(report, indent=2, default=float))
    plot(headline, weights, w_late)
    log_event(
        log,
        "policy_sim_done",
        best_baseline=best_base,
        diff=diff,
        saved={k: round(v["risky"]["saved_per_conn"], 3) for k, v in headline.items()},
    )
    con.close()


def plot(headline: dict, weights: dict, w_late: float) -> None:
    names = ["always_keep", "rule_transfer", "rule_slack", "historical_rate", "logreg", "model", "oracle"]
    labels = [
        "Always keep",
        "Transfer rule",
        "Slack rule\n(DB-forecast proxy)",
        "Historical\nrate",
        "Logistic\nregression",
        "LightGBM\n(calibrated)",
        "Oracle\n(hindsight)",
    ]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    v = [headline[n]["risky"]["saved_per_conn"] for n in names]
    lo = [v[i] - headline[n]["risky"]["ci95"][0] for i, n in enumerate(names)]
    hi = [headline[n]["risky"]["ci95"][1] - v[i] for i, n in enumerate(names)]
    colors = ["#9aa5b1"] * 5 + ["#3b6ea5", "#d9d9d9"]
    ax[0].bar(labels, v, yerr=[lo, hi], color=colors, capsize=4)
    ax[0].axhline(0, c="black", lw=0.8)
    ax[0].set(
        title=f"Generalized minutes saved per risky connection vs always-keep\n(test month Oct 2026, w_late={w_late}, 95% CI by day)",
        ylabel="minutes saved",
    )
    ax[0].tick_params(axis="x", labelsize=8)
    ws = sorted(weights, key=float)
    for n, c in [("rule_slack", "#9aa5b1"), ("logreg", "#c49a6c"), ("model", "#3b6ea5"), ("oracle", "#d9d9d9")]:
        ax[1].plot([float(w) for w in ws], [weights[w][n]["saved"] for w in ws], marker="o", label=n, color=c)
    ax[1].axhline(0, c="black", lw=0.8)
    ax[1].set(title="Sensitivity to the cost of unplanned waiting", xlabel="w_late / w_plan", ylabel="minutes saved")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "policy_comparison.png", dpi=130)


if __name__ == "__main__":
    main()
