"""Streamlit dashboard: "Will I make my connection?"

Runs from `app/bundle/` only (no warehouse, no model files), so it deploys on Streamlit Community Cloud.
Rebuild the bundle with `make app-bundle`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cre.config import settings  # noqa: E402
from cre.decision.policy import recommend  # noqa: E402

BUNDLE = ROOT / "app" / "bundle"
st.set_page_config(page_title="Will I make my connection?", page_icon="🚆", layout="wide")


@st.cache_data
def data():
    df = pd.read_parquet(BUNDLE / "test_connections.parquet")
    daily = pd.read_parquet(BUNDLE / "daily.parquet")
    mm = json.loads((BUNDLE / "model_metrics.json").read_text())
    pm = json.loads((BUNDLE / "policy_metrics.json").read_text())
    fm = json.loads((BUNDLE / "fallback_metrics.json").read_text())
    return df, daily, mm, pm, fm


df, daily, mm, pm, fm = data()
w_late = pm["w_late"]
st.title("🚆 Will I make my connection?")
st.caption(
    "Calibrated miss probability and a keep-or-switch recommendation for transfers at Leipzig, Erfurt, Halle "
    "and Berlin Hbf. Replay on held-out Oct 2026 data (Deutsche Bahn via piebro/deutsche-bahn-data, CC BY 4.0)."
)

tab1, tab2, tab3 = st.tabs(["Check a connection", "Calibration & policies", "Predicted vs actual"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    station = c1.selectbox("Station", sorted(df.station_key.unique()))
    sdf = df[df.station_key == station]
    day = c2.selectbox("Day", sorted(sdf.service_date.astype(str).unique(), reverse=True))
    sdf = sdf[sdf.service_date.astype(str) == day]
    lead = c3.radio("Asking how long before arrival?", [30, 10], horizontal=True, format_func=lambda x: f"{x} min")
    stale = c4.toggle("Simulate stale live data", help="Live feed older than 2 h → the static fallback model is used")
    sdf = sdf[sdf.lead_min == lead].sort_values("a_arr_planned")
    labels = (
        sdf.a_arr_planned.dt.strftime("%H:%M")
        + "  "
        + sdf.train_a
        + "  →  "
        + sdf.b_dep_planned.dt.strftime("%H:%M")
        + "  "
        + sdf.train_b
        + " (to "
        + sdf.next_stop_b
        + ")"
    )
    if sdf.empty:
        st.info("No connections for this selection.")
    else:
        risky_first = st.checkbox("Show risky connections first (naive slack < 10 min)", value=True)
        order = sdf.assign(lbl=labels.values)
        if risky_first:
            order = order.sort_values(["naive_slack"])
        pick = st.selectbox("Connection", order.lbl.tolist())
        r = order[order.lbl == pick].iloc[0]
        p = float(r.p_fallback if stale else r.p_lgb_cal)
        rec = recommend(
            p,
            float(r.e_hat),
            w_late=w_late,
            fallback=stale,
            naive_slack=float(r.naive_slack),
            gate_slack=settings()["decision"].get("commit_gate_slack_min"),
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("P(miss)", f"{p:.0%}")
        m2.metric("Expected lost time if you keep the plan", f"{rec.expected_lost_min:.0f} min")
        m3.metric("Alternative costs", f"+{rec.alt_extra_min:.0f} min" if rec.alt_extra_min < 240 else "none in 4 h")
        m4.metric("Confidence", rec.confidence)
        (st.warning if rec.action == "take_alternative" else st.success)(
            f"**Recommendation: {'take the alternative' if rec.action == 'take_alternative' else 'keep your plan'}.** "
            f"{rec.reason}"
        )
        if stale:
            st.info(
                f"Fallback model (no live delays). Test Brier {fm['test']['brier']:.3f} vs "
                f"{mm['test']['lgb_cal']['brier']:.3f} for the full model: treat with care."
            )
        with st.expander("What we knew at prediction time / what actually happened"):
            k1, k2 = st.columns(2)
            k1.write(
                {
                    "planned transfer (min)": int(r.planned_transfer_min),
                    "A's known delay (min)": int(r.a_known_delay),
                    "A already running": bool(r.a_started),
                    "B's known delay (min)": int(r.b_known_delay),
                    "naive slack (min)": int(r.naive_slack),
                }
            )
            k2.write(
                {
                    "outcome": r.outcome,
                    "A actual arrival": str(r.a_arr_actual),
                    "B actual departure": str(r.b_dep_actual),
                    "lost minutes if B not taken": float(r.lost),
                }
            )

with tab2:
    a, b = st.columns(2)
    test = df.copy()
    bins = np.linspace(0, 1, 11)
    test["bin"] = pd.cut(test.p_lgb_cal, bins, include_lowest=True)
    rel = (
        test.groupby("bin", observed=True)
        .agg(pred=("p_lgb_cal", "mean"), obs=("missed", "mean"), n=("missed", "size"))
        .reset_index(drop=True)
    )
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="grey")
    pts = (
        alt.Chart(rel)
        .mark_line(point=True)
        .encode(
            x=alt.X("pred", title="predicted P(miss)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("obs", title="observed miss rate", scale=alt.Scale(domain=[0, 1])),
            tooltip=["pred", "obs", "n"],
        )
    )
    a.subheader("Calibration (bundle sample of the test month)")
    a.altair_chart(diag.encode(x="x", y="y") + pts, width="stretch")
    t = mm["test"]["lgb_cal"]
    a.caption(f"Full test month: Brier {t['brier']:.4f}, ECE {t['ece']:.4f}, ROC-AUC {t['roc_auc']:.3f}.")
    h = pm["test_headline"]
    names = {
        "always_keep": "Always keep",
        "rule_transfer": "Transfer rule",
        "rule_slack": "Slack rule",
        "historical_rate": "Historical rate",
        "logreg": "Logistic reg.",
        "model": "LightGBM (ours)",
        "oracle": "Oracle",
    }
    pc = pd.DataFrame(
        [
            {
                "policy": names[k],
                "saved": h[k]["risky"]["saved_per_conn"],
                "lo": h[k]["risky"]["ci95"][0],
                "hi": h[k]["risky"]["ci95"][1],
            }
            for k in names
        ]
    )
    bars = (
        alt.Chart(pc)
        .mark_bar()
        .encode(
            x=alt.X("policy", sort=None, title=None),
            y=alt.Y("saved", title="generalized min saved / risky connection"),
            color=alt.condition(alt.datum.policy == "LightGBM (ours)", alt.value("#3b6ea5"), alt.value("#9aa5b1")),
        )
    )
    err = alt.Chart(pc).mark_errorbar().encode(x=alt.X("policy", sort=None), y="lo", y2="hi")
    b.subheader("Policy comparison (test month, 95% CI by day)")
    b.altair_chart(bars + err, width="stretch")
    b.caption(
        f"Unplanned waiting weighted {w_late}× planned waiting. At 1× (pure minutes) no policy can beat "
        "'always keep' by construction; see docs/DESIGN.md §1.1."
    )

with tab3:
    st.subheader("Daily predicted vs actual miss rate (replay of the held-out month)")
    st.caption(
        "Mean predicted P(miss) vs observed share missed, per day. In production this view reads the live "
        "store (`make collect` hourly); without DB API credentials the demo replays Oct 2026."
    )
    dl = daily.melt("day", ["predicted", "observed"], var_name="series", value_name="rate")
    st.altair_chart(
        alt.Chart(dl).mark_line(point=True).encode(x="day:T", y=alt.Y("rate", title="miss rate"), color="series"),
        width="stretch",
    )
    st.dataframe(daily.rename(columns={"ece": "daily ECE"}), width="stretch", hide_index=True)
