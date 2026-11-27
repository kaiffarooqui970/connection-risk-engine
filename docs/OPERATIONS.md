# OPERATIONS: fallback, monitoring, retraining, rollback, shadow mode

All thresholds below are constants in `src/cre/monitor/rules.py`; this doc and the code can't disagree
silently. Tests: `tests/test_monitor.py`, `tests/test_live.py`.

## 1. Data flow in production

```
hourly  collect.yml  → DB Timetables plan + fchg (8 hub EVAs, ≤ 32 req/run, 60 req/min limit) → live store
on request           → live connections (A arriving in lead±5 min) → features → P(miss) → recommendation
daily   monitor.yml  → freshness, coverage, (matured) calibration, decision value, PSI → actions
monthly (or on alert) retrain → shadow ≥ 14 days → promote / reject
```

## 2. Fallback (stale or invalid live data)

| Condition | Detection | Behaviour |
|---|---|---|
| Change feed older than **120 min** for the station | `live.fetch_log` latest successful `fchg` (`monitor.live.freshness`) | Serve the **static fallback model** (timetable + calendar + weather + historical rates, no live delays). Output flagged `confidence = "reduced (fallback model)"` |
| No feed at all (`MISSING`) | same | same, plus an alert |
| Live rows fail contracts (> 5% invalid) | `ContractViolation` in the collector | Batch rejected, nothing written. Freshness ages → fallback kicks in automatically |
| Weather older than 3 h | freshness check | Weather features become NaN (the model has seen NaN) |

The fallback is not a guess: it is trained and evaluated like the main model
(`reports/fallback_metrics.json`), so "reduced confidence" has a measured cost. See EVALUATION.md.

## 3. Daily monitoring (what `cre.monitor.run` computes)

| Signal | How | Warn | Act |
|---|---|---|---|
| Freshness | age of last OK `fchg` | n/a | **> 120 min → fallback** |
| Calibration | ECE on predictions whose outcome has matured (B departed), rolling 7 days, days with n ≥ 500 | ECE > 0.03 | **ECE > 0.05 → retrain** |
| Decision value | generalized minutes saved per risky connection vs always-keep, daily | n/a | **< 0 for 3 consecutive days → roll back** to the previous model |
| Drift | PSI vs training on key features (`a_known_delay`, `naive_slack`, `planned_transfer_min`, `hist_miss_rate`, `precip_mm_1h`) | PSI > 0.10 | **PSI > 0.25 → retrain** (after checking it isn't a data bug) |
| Coverage | share of requests answered by the full model | < 90% | investigate collector / credentials |

PSI > 0.25 is a *prompt to investigate first*. A PSI spike after a feed change is more often a parsing bug than a
world change, and retraining on a bug bakes it in.

## 4. Retraining

- **Scheduled:** monthly, when the next HF month is published (`make reproduce` with the month list advanced).
- **Triggered:** any RETRAIN action above.
- **Recipe:** identical pipeline. Training window = the most recent 6 weeks, validation = the following ~2 weeks,
  historical-rate features recomputed on the new training window only.
- **Release gates** (all must hold on the new validation period): commit rate on non-risky connections ≤ 1% (with the
  serving gate `commit_gate_slack_min` applied); log loss ≤ champion's; ECE ≤ 0.03; decision value ≥
  champion's (paired, by-day bootstrap CI must not be entirely below 0); leakage tests pass.

## 5. Shadow mode

A candidate is scored on every request next to the champion, but **only the champion's output is shown**.
Both are logged. After **≥ 14 days**, promote iff:
- candidate log loss better by ≥ 0.002,
- candidate ECE ≤ 0.03,
- decision value not worse than the champion's.

Otherwise reject and keep the champion. `rules.promote()` implements exactly this.
The replay in `reports/monitoring/replay_summary.json` runs a real shadow comparison on Oct 2026. Results are in EVALUATION.md.

## 6. Rollback

- Models are immutable files (`models_store/*`, versioned by training window). The serving config points at one.
- Roll back = point to the previous model. No retraining needed, takes minutes.
- **Automatic trigger:** decision value < 0 for 3 consecutive days. **Manual trigger:** any incident where outputs look wrong.
- After a rollback, the failed model goes to shadow (not deleted) so it can be diagnosed on live traffic.

## 7. Known operational limitations (honest list)

- **Live store = GitHub Actions cache.** Caches unused for 7 days are evicted. That's fine for a demo; production needs object
  storage plus a small database.
- **No DB API credentials in this build.** The live path is tested on fixture XML; scheduled jobs log `collect_skipped` /
  `NO_DATA`. Calibration on *live* outcomes therefore hasn't been observed yet. The replay on Oct 2026 stands in for it.
- **Train/serve skew:** live "known delay" = DB's forecast at S; training used the last realized upstream delay
  (DESIGN §5.1). The first two weeks of live operation must be treated as a shadow period for the champion itself.
- **Outcome maturity:** a prediction's outcome is known only after B departs (and the fallback after up to 4 h). Daily
  monitoring therefore judges *yesterday*.

## 8. Runbook (first 15 minutes of an alert)

1. `FALLBACK` → check `live.fetch_log` for 401/403 (credentials) vs 429 (rate) vs 5xx (DB side). Credentials: rotate
   secrets. Rate: check that no other job shares the key. DB side: wait. The fallback is already serving.
2. `RETRAIN (ECE)` → look at the reliability plot by station and lead. One station broken is usually a timetable change
   or a construction site. Exclude it or retrain.
3. `RETRAIN (PSI)` → compare the feature's histogram today vs training. Parsing bug? Fix ingestion, don't retrain.
4. `ROLLBACK` → roll back first, investigate second.
