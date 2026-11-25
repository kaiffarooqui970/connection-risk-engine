# FEATURES: one sentence each

Prediction time `t_pred = A's planned arrival at S − lead` (lead = 30 or 10 min). Every feature uses only
information timestamped ≤ t_pred. This is enforced by `tests/test_features_leakage.py`, which recomputes rows in a
world where everything after t_pred is erased and requires identical output.

| Feature | One sentence |
|---|---|
| `lead_min` | How far ahead we predict: at 30 min there's more time for A's delay to change than at 10. |
| `planned_transfer_min` | The scheduled buffer between A's arrival and B's departure: the starting slack before any delay. |
| `a_known_delay` | A's delay at its last stop already departed by t_pred, i.e. the best real-time estimate of how late A is now (0 if A hasn't started). |
| `a_started` | Whether A has departed any stop yet: if not, `a_known_delay = 0` means "unknown", not "on time". |
| `a_delay_trend_1` / `_2` | How A's delay changed over its last one/two observed stops: growing delays tend to keep growing. |
| `a_stops_remaining` | Stops A still has to serve before S: each stop is another chance to lose (or recover) time. |
| `a_planned_min_remaining` | Scheduled minutes from A's last observed departure to S: the time over which delay can still develop. |
| `a_min_since_obs` | Minutes since A's last observation: a stale reading is less trustworthy. |
| `a_run_progress` | Where S lies in A's run (0 = origin, 1 = terminus): delay accumulates along long-distance runs (EDA). |
| `b_known_delay` | B's current delay if B is already running: a late B *saves* connections. |
| `b_started` | Whether B is already running; most B's start at or near S, so their delay is still unknown. |
| `b_stops_before_s` | How many stops B serves before S: B's own exposure to delay before it reaches S. |
| `naive_slack` | Planned transfer − A's known delay + B's known delay: the "DB-app" estimate of the remaining buffer, and the input to the rule baselines. |
| `hour`, `weekday`, `is_weekend`, `is_rush` | Delay patterns follow the day: long-distance lateness rises from ~20% at 06:00 to ~60% after noon (EDA). |
| `is_public_holiday`, `is_school_holiday` | Holidays change traffic volume and timetables (state-specific, OpenHolidays API). |
| `temperature_c`, `precip_mm_1h`, `precip_mm_3h`, `wind_gust_kmh` | Weather at the hub in the last completed hour(s): heavy rain, storms and heat cause infrastructure faults. |
| `hist_miss_rate` | Training-period miss rate for (station, transfer-window bucket, hour), out-of-fold by week and smoothed: the prior for this kind of connection. |
| `hist_a_late_rate` | How often *this* train number A arrived at S ≥ MCT late in training: some trains are chronically late. |
| `hist_b_cancel_rate` | How often *this* train number B was cancelled at S in training: cancellations are 13% of misses. |
| `l_hat_planned_min` | Scheduled lost time if missed (best alternative to T after B vs B, clamped to [0, 240]): used by the cost model, and as a feature (frequent services are timed differently). |
| `station_key`, `a_category`, `b_category`, `a_train_type`, `b_train_type` | Categorical context: station layout (MCT), train types (ICE ≈ 56% ≥5 min late vs S-Bahn ≈ 8%). |

## Deliberately NOT used

| Candidate | Why not |
|---|---|
| Cancellation flags of A/B | The source has no announcement time, so we can't know whether the cancellation was known at t_pred. |
| A's actual arrival / B's actual departure at S | They *are* the label. |
| DB's own forecast (change snapshots) | Only ~6-hourly in the HF data (see DECISIONS). |
| Platform | Not in the processed data. |
| Train number as a raw categorical | ~4,400 levels → memorization; replaced by smoothed out-of-fold rates. |
