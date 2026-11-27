# DECISIONS log

Format: **date · decision** — alternatives — reason.

## Phase 0

- **2026-11-25 · Data subset = 4 hubs, Aug–Oct 2026.** Alternatives: more hubs, more months. Reason: the
  brief says "manageable"; local disk is constrained (~5 GB free); these are the most recent complete
  months, so live monitoring continues from them. Oct = test month.
- **2026-11-25 · Berlin Hauptbahnhof's 3 EVAs (08011160, 08098160, 08089021) = one station.** Alternative: treat
  them separately. Reason: passengers change between levels; an upper→deep transfer is a real transfer.
  The larger walk is priced via MCT = 8 min.
- **2026-11-25 · "Known delay" = last realized upstream delay, not DB's forecast.** Alternative: DB forecast
  from change snapshots. Reason: snapshots in the HF data are ~6 h apart (checked on Mar 2026,
  Leipzig Hbf), so no forecast at t−30 min exists. The train/serve skew is documented in DESIGN §5.1.
- **2026-11-25 · Baseline (c) "DB forecast" replaced by a delay-propagation proxy.** Reason: same as above.
- **2026-11-25 · Cost model weights unplanned waiting (`w_late/w_plan = 2.0` default).** Alternative: pure
  minutes. Reason: in pure minutes, committing to the next train is weakly dominated by "always keep"
  (proof sketch in DESIGN §1.1), so the success metric would be 0 by construction. The weight is
  stress-tested from 1.0 to 3.0, and the pure-minute result is always reported next to it.
- **2026-11-25 · Fallback target T = B's next stop.** Alternatives: B's final destination; next "major"
  stop. Reason: B's final destination is often far away and few alternatives reach it, which inflates L.
  "Major" needs an arbitrary threshold. The next stop is simple, reproducible and is where a
  passenger can always re-plan from. Known weakness: it understates the loss for long-distance trips.
- **2026-11-25 · Risky connection = naive slack < 10 min at prediction time.** Reason: it fixes the
  denominator before seeing results, so it can't be tuned to flatter the model.
- **2026-11-25 · Bootstrap by day, not by connection.** Reason: connections on the same day share weather
  and disruption, and i.i.d. resampling would produce CIs that are too narrow.
- **2026-11-25 · Repo location `~/connection-risk-engine`, Python 3.12 via uv.** Reason: the Desktop is
  iCloud-synced (venvs break). 3.12 because the system Python is 3.15 and lightgbm/dbt wheels lag; the
  brief requires 3.11+.

## Phase 1

- **2026-11-25 · `run_id` = ride id + start timestamp parsed from stop `id`.** Alternative: `train_line_ride_id`
  as documented. Reason: the documented id groups many runs of a line (duplicates, 50% time-order agreement);
  the parsed key is unique per (run, stop) and 99.92% time-ordered.
- **2026-11-25 · Effective actual departure = max(departure_change, arrival_change).** Alternative: drop rows.
  Reason: 15.6k rows depart "before arriving" by ~1 min, which is a stale departure forecast. Dropping them would bias
  against S-Bahn; clamping is physically correct.
- **2026-11-25 · Cancellation flags override times.** Reason: cancelled stops still carry change times.
- **2026-11-25 · Platforms not used.** Reason: absent from the processed table; only in 6-hourly snapshots.
- **2026-11-25 · Added matplotlib** (figures for docs/README). Lightest standard option.

## Phase 2

- **2026-11-25 · pandera (polars backend) + quarantine tables, fail batch above an invalid-share threshold.**
  Alternatives: Great Expectations (heavier), dbt tests only (too late: after load). Reason: contracts at the
  boundary, quarantine keeps bad rows inspectable, threshold separates "dirty row" from "broken source".
- **2026-11-25 · Normalize dtypes before validating.** Reason: the source uses ns timestamps / int32. Loosening
  the contract would hide future drift.
- **2026-11-25 · Weather keyed in UTC.** Reason: DST fall-back duplicates local hours.
- **2026-11-25 · httpx** (dependency). Reason: timeouts plus `MockTransport` for offline tests of retry/limit logic.
  stdlib urllib would need hand-rolled mocking.
- **2026-11-25 · Live store = DuckDB in GitHub Actions cache.** Alternatives: commit to a data branch (violates
  "never commit raw data"), S3 (needs a cloud account). Reason: zero-cost demo. Documented as not durable.
- **2026-11-25 · Missing DB creds → clean skip (exit 0).** Reason: no creds are available in this build. A red
  scheduled job would be ignored noise.

## Phase 3

- **2026-11-25 · EDA fixes encoded once in `stg_stops`** (run_id, cancelled → no actual, dep = max(dep, arr)),
  each guarded by a singular dbt test. Reason: a fix that lives in one place can't drift between pipelines.
- **2026-11-25 · Hub config → dbt seed generated from `settings.toml`.** Alternative: dbt vars / a hand-written
  seed. Reason: one source of truth for station config.
- **2026-11-25 · Mart dedups the same physical departure under two run ids** (EDA #9; keep lowest run_id).
- **2026-11-25 · `platform` kept as an explicit null column.** Reason: makes the absence visible to consumers.
- **2026-11-25 · dbt-duckdb as a dev dependency.** Reason: the brief requires it. It is the SQL layer, not a runtime dependency of
  the scoring code.

## Phase 4

- **2026-11-25 · Candidate needs planned transfer ≥ MCT(S)** (on top of the 2–30 min window). Alternative: keep
  2–4 min transfers. Reason: on the first build, sub-MCT transfers missed 79% of the time *by construction*. A
  planner never offers them, so they aren't planned connections, and they'd make the model look good for free.
- **2026-11-25 · One B per (A, T), chosen by earliest planned *arrival* at T.** Alternatives: every pair (3.7M, mostly
  meaningless), earliest departure. Reason: earliest departure made slow trains "B" and produced negative lost times
  when a later fast train overtook them. Earliest arrival is what a journey planner offers.
- **2026-11-25 · S-Bahn→S-Bahn pairs and buses as A/B excluded.** Reason: at 5–10 min headways nobody plans
  that connection; buses are replacement services with unreliable data. Buses can still be the *fallback*.
- **2026-11-25 · Backtracking excluded** (B's next stop = A's previous stop), as are same train number and same line.
- **2026-11-25 · Lost time floored at 0.** Reason: 2.6% of missed connections reached T *before* B's reference arrival
  (95% are B-cancelled cases where an earlier, delayed train got there first). Nothing was lost.
- **2026-11-25 · A-cancelled connections: labelled missed, lost time undefined (null).** Reason: the passenger never
  reaches S, so no decision at S can help. They're kept for the classifier but excluded from the policy simulation.
- **2026-11-25 · Leakage-free lost-time estimate L̂ = scheduled best alternative's arrival − B's planned arrival at T.**
  Reason: the timetable is known at prediction time. corr(L̂, actual L | missed) = 0.54.

## Phase 5

- **2026-11-25 · Upstream stop "observed" iff actual departure ≤ t_pred; final recorded value used.** Alternative:
  snapshots (unavailable). Reason: realized departures are published within ~1 min. Assumption documented.
- **2026-11-25 · Cancellation flags never used as features.** Reason: no announcement timestamp, so leakage can't be ruled out.
- **2026-11-25 · Historical rates: training only, out-of-fold by ISO week (counts AND prior), m-estimate m=50.**
  Alternative: K-fold by row (leaks via same-day correlation). Reason: weeks are long enough to contain a
  disruption episode. See POSTMORTEM #1 for the prior.
- **2026-11-25 · Train number → smoothed late/cancel rates instead of a raw categorical.** Reason: ~4.4k levels.
- **2026-11-25 · Holidays from OpenHolidays API, frozen into `configs/holidays_2026.csv`.** Alternatives: `holidays`
  package (public holidays only), ferien-api.de (only SN for 2026). Reason: one source for both kinds, no new dependency.
- **2026-11-25 · Split purge: drop connections whose A- or B-run touches two splits** (1,743 rows).

## Phase 6

- **2026-11-25 · Split: train 08-01…09-14, val 09-15…09-30, test = October.** Reason: forward-in-time only;
  October also contains the autumn holidays and a DST change, which is a fair stress test.
- **2026-11-25 · LightGBM tuned lightly (20 Optuna trials on a 400k training subsample, early stopping on val).**
  Reason: more trials = more looks at validation; the spread between trials was small.
- **2026-11-25 · 3-seed ensemble (mean of probabilities).** Seed spread on test is in the 4th decimal (Brier 0.0767–0.0768),
  so seeds are not a source of uncertainty worth worrying about here. Day-to-day variation is (see CIs).
- **2026-11-25 · Calibrator chosen by 2-fold by-day CV inside validation → Platt** (CV log loss 0.2816 vs isotonic 0.2818 vs
  raw 0.2836). **On test, Platt made ECE worse (0.004 raw → 0.020).** The test month's miss rate is 18.5% vs 21.1% in validation,
  and the calibrator learned validation's level. We did *not* switch to "raw" after seeing test: that would be selecting
  on test. Logged in POSTMORTEM #2. Both stay inside the 0.03 guardrail.
- **2026-11-25 · pandas added** (LightGBM native categoricals + sklearn ColumnTransformer work on pandas dtypes).

## Phase 7

- **2026-11-27 · Probability policies use the cost-model threshold `P > w_plan / w_late`, never a tuned one.**
  Alternative: tune the threshold on validation like the rules. Reason: with a calibrated P the threshold follows from
  the costs. Needing a tuned threshold would mean the calibration is wrong.
- **2026-11-27 · Rule thresholds tuned on validation over all connections, grid-searched; "never commit" (−1e9) wins
  if no threshold saves anything.** Result: transfer-time rule = never (at w=2), slack rule X = 4 min.
- **2026-11-27 · Ê = planned arrival at T of the earliest alternative still reachable given A's known delay + MCT,
  minus B's planned arrival at T, capped at 240.** Reason: uses only the timetable plus information known at `t_pred`.
- **2026-11-27 · Realized commit cost charges minutes beyond the promised Ê as unplanned (`w_late`).** Reason: if the
  alternative itself fails, the passenger is waiting unexpectedly again.
- **2026-11-27 · Model vs best baseline is a paired by-day bootstrap** (same resampled days for both). Reason:
  unpaired CIs overlap even when one policy wins on almost every day.
- **2026-11-27 · Non-risky commit-rate guardrail breached (1.74% > 1%); not fixed on October.** Alternative: gate
  commits to risky connections now. Reason: changing the policy after seeing test would make October a validation
  set. The fix options are listed in EVALUATION §5 and must be decided before the next month's data is scored.

## Phase 8

- **2026-11-27 · Disruption days derived from the data (top-decile daily miss or cancellation share) + weather
  extremes (gust ≥ 60 km/h or rain ≥ 5 mm/h).** Alternative: an external strike/storm calendar. Reason: none is
  available offline in a reproducible form; the data-driven definition is circular-ish (high-miss days) and is
  reported as such.

## Phase 9

- **2026-11-27 · Replay freshness proxy: a test day with < 50% of the median row count = STALE.** Reason: historical
  data has no feed timestamps. In October no day triggered it.
- **2026-11-27 · Shadow candidate = champion recipe on fresher data (train through Sep 22), isotonic calibration on
  Sep 23–30, scored on Oct 1–14 (14 days = `SHADOW_MIN_DAYS`).** Reason: it tests the operational answer to
  POSTMORTEM #2 (refit closer to deployment). Verdict: promote (log loss −0.0032, ECE 0.012).
- **2026-11-27 · Temperature PSI 3.06 does not trigger retraining** because it is not a key feature in
  `rules.PSI_KEY_FEATURES`. Logged as a seasonality risk for the next retrain (EVALUATION §6).

## Phase 10

- **2026-11-27 · The Streamlit app reads only `app/bundle/` (5.8 MB, committed).** Alternative: query the warehouse.
  Reason: the 1.5 GB DuckDB file can't be deployed to Streamlit Community Cloud. Row-level explorer = last 7 days
  of October; daily calibration covers the whole month.
