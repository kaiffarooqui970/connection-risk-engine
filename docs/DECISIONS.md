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
