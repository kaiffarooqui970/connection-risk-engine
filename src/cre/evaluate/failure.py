"""Phase 8: where does the policy fail? Slices of the test month by day type, station, window, train pair.

Disruption days are identified from the data (no external strike/storm calendar is available offline):
days whose miss rate or cancellation share is in the top decile of the test month, plus explicit
weather extremes (max hourly gust >= 60 km/h or precipitation >= 5 mm/h at the hub).
"""

from __future__ import annotations

import json

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


def slice_stats(g: pd.DataFrame, w_late: float, w_plan: float, rule_x: float) -> dict:
    keep, commit = realized_costs(g.missed, g.lost, g.e_hat, w_late, w_plan)
    risky = (g.naive_slack < 10).to_numpy()
    out = {"n": len(g), "miss_rate": float(g.missed.mean()), "risky_share": float(risky.mean())}
    y = g.missed.to_numpy(float)
    out |= {
        "brier": suite(y, g.p_lgb_cal.to_numpy())["brier"],
        "ece": ece(y, g.p_lgb_cal.to_numpy()),
        "mean_p": float(g.p_lgb_cal.mean()),
    }
    for name, d in {
        "model": commit_decision(g.p_lgb_cal.to_numpy(), g.e_hat.to_numpy(), w_late, w_plan),
        "rule_slack": (g.naive_slack < rule_x).to_numpy(),
    }.items():
        s = keep - np.where(d, commit, keep)
        out[f"{name}_saved_risky"] = float(s[risky].mean()) if risky.any() else None
    return out


def main() -> None:
    cfg = settings()["decision"]
    w_late, w_plan = cfg["w_late"], cfg["w_plan"]
    pm = json.loads(path("reports/policy_metrics.json").read_text())
    rx = pm["rule_thresholds_tuned_on_val"]["rule_slack"]
    con = duckdb.connect(str(path(settings()["data"]["duckdb_path"])), read_only=True)
    df = load_frame(con)
    df = df[df.split == "test"].copy()
    df["day"] = df.service_date.astype(str).str[:10]
    wx = con.sql("""SELECT cast(timezone('Europe/Berlin', timezone('UTC', ts_utc)) AS date)::VARCHAR AS day, station_key,
                           max(wind_gust_kmh) AS gust, max(precipitation_mm) AS precip
                    FROM staging.stg_weather GROUP BY 1, 2""").df()
    df = df.merge(wx, on=["day", "station_key"], how="left")
    daily = df.groupby("day").agg(miss=("missed", "mean"), canc=("outcome", lambda s: (s == "b_cancelled").mean()))
    hi_miss = set(daily.index[daily.miss >= daily.miss.quantile(0.9)])
    hi_canc = set(daily.index[daily.canc >= daily.canc.quantile(0.9)])
    df["day_type"] = np.select(
        [df.day.isin(hi_miss | hi_canc), (df.gust >= 60) | (df.precip >= 5)],
        ["disruption (top-decile miss/cancel day)", "weather extreme (gust>=60 or rain>=5mm/h)"],
        "normal",
    )
    df["window"] = pd.cut(
        df.planned_transfer_min, [4, 7, 10, 15, 20, 30], labels=["5-7", "8-10", "11-15", "16-20", "21-30"]
    )
    df["pair"] = df.a_category + "->" + df.b_category

    res = {"disruption_days": sorted(hi_miss | hi_canc)}
    for col in ["day_type", "station_key", "window", "pair", "lead_min"]:
        res[col] = {str(k): slice_stats(g, w_late, w_plan, rx) for k, g in df.groupby(col, observed=True)}
    res["worst_days_model"] = (
        pd.Series(
            {
                d: slice_stats(g, w_late, w_plan, rx)["model_saved_risky"]
                - slice_stats(g, w_late, w_plan, rx)["rule_slack_saved_risky"]
                for d, g in df.groupby("day")
            }
        )
        .sort_values()
        .head(5)
        .round(3)
        .to_dict()
    )
    # guardrail check: are commits on NON-risky connections right or wrong?
    keep, commit = realized_costs(df.missed, df.lost, df.e_hat, w_late, w_plan)
    d = commit_decision(df.p_lgb_cal.to_numpy(), df.e_hat.to_numpy(), w_late, w_plan)
    nr = (df.naive_slack >= 10).to_numpy()
    sel = d & nr
    res["non_risky_commits"] = {
        "rate": float(sel.sum() / nr.sum()),
        "n": int(sel.sum()),
        "share_correct_in_hindsight": float((commit[sel] < keep[sel]).mean()),
        "net_saved_per_commit": float((keep[sel] - commit[sel]).mean()),
        "b_cancelled_share": float((df.outcome[sel] == "b_cancelled").mean()),
    }
    path("reports/failure_analysis.json").write_text(json.dumps(res, indent=2, default=float))

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for a, col in zip(ax, ["day_type", "window", "pair"], strict=True):
        t = pd.DataFrame(res[col]).T
        x = np.arange(len(t))
        a.bar(x - 0.2, t.model_saved_risky.astype(float), 0.4, label="LightGBM", color="#3b6ea5")
        a.bar(x + 0.2, t.rule_slack_saved_risky.astype(float), 0.4, label="slack rule", color="#9aa5b1")
        a.set_xticks(x, [s.split(" (")[0] for s in t.index], rotation=35, ha="right", fontsize=8)
        a.set(title=f"Saved per risky connection by {col}")
        a.axhline(0, c="black", lw=0.8)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path("reports/figures/failure_slices.png"), dpi=130)
    print(
        json.dumps(
            {
                k: v
                for k, v in res.items()
                if k in ("day_type", "non_risky_commits", "worst_days_model", "disruption_days")
            },
            indent=1,
            default=float,
        )[:4000]
    )


if __name__ == "__main__":
    main()
