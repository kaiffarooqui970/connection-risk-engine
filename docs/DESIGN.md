# DESIGN — "Will I make my connection?"

*Status: Phase 0, written before any modelling code. Updated only via `DECISIONS.md` entries.*

## 1. The user and the decision

**User.** A passenger on train **A**, heading for station **S** (a hub), with a planned change to train **B**.
Some minutes before A's planned arrival (the *lead time*, 30 or 10 min) they look at their phone.

**Decision.** *Keep the plan* (go for B) or *commit to the alternative*: the next departure **C** from S
that also reaches B's next stop **T** with a comfortable buffer. "Commit" means the passenger
re-plans now: they tell whoever is waiting, stop sprinting, and treat the extra time as planned time.

**What each error costs.**

| Outcome | Keep plan | Commit to C |
|---|---|---|
| Connection would have held | 0 | certain extra time `E = arr_T(C) − arr_T(B)` (*unnecessary switch*) |
| Connection would have been missed | `L` = lost time until the next *viable* train reaches T (*missed connection*), experienced as **unplanned** waiting | `E`, experienced as **planned** waiting |

### 1.1 The structural catch (and why the cost model has a weight in it)

If unplanned and planned minutes are valued the same, committing to C **can never beat** keeping the plan
with hindsight. When B is missed, the first viable fallback is C or something earlier, so `L ≤ E`.
When B is made, keeping costs 0. In pure minutes "always keep" is optimal by construction, and no model
can save time.

The decision has value only because a missed connection costs more than the clock says: an
uncertain wait on a platform, a sprint, a missed pickup. Transport appraisal practice weights
unplanned waiting and lateness well above planned time (UK appraisal guidance, e.g. WebTAG / PDFH, uses
multipliers of roughly 2–3×). We make this explicit:

```
cost(keep)   = w_late · L      if missed, else 0
cost(commit) = w_plan · E      (+ w_late · extra loss if C itself is missed)
```

Default `w_late / w_plan = 2.0`. **This is an assumption, not a fact.** Phase 7 reports results at
1.0 (pure minutes, where we expect zero value), 1.5, 2.0 and 3.0. The headline states the weight
it uses.

## 2. Success metric and guardrails

**Primary metric.** *Generalized minutes saved per risky connection*, compared with each baseline policy on a
held-out month, using **actual** outcomes, with a 95% bootstrap CI (resampled by day, because
connections on the same day share weather and disruption).

**Risky connection** (the denominator, fixed before modelling): a connection whose *naive slack* at prediction
time, `planned transfer − known delay(A) + known delay(B)`, is below **10 min**. Connections with
lots of slack are trivially "keep", and including them would dilute any effect.

**Guardrails** (they can block a release even if minutes look good):
- Calibration: ECE ≤ 0.03 on test, overall and per lead time. Reliability curve close to the diagonal.
- Coverage: share of connections where the full-information model is used (rather than the fallback) ≥ 90%.
- Regret on *non-risky* connections: the policy must not tell people to abandon safe connections
  (commit rate on non-risky connections ≤ 1%).
- Pure-minute (`w = 1`) loss vs always-keep reported, so nobody can claim time savings that are
  only weighting.

Not success metrics: accuracy, MAE of delay. They don't measure the decision.

## 3. Baselines to beat

| Policy | Rule |
|---|---|
| (a) Always keep | Never commit. The honest null. |
| (b) Transfer-time rule | Commit if *naive slack* < X min (X tuned on validation). This is the "DB-style" rule. |
| (c) DB forecast | **Not available historically.** The HF change snapshots are ~6 h apart, so DB's forecast at t−30 min isn't in the data. We use the closest honest proxy, a *delay-propagation* rule: assume A keeps its current delay and B its current delay, and miss if the propagated transfer is < MCT. This is (b) with X = MCT. |
| (d) Historical rate | P(miss) = training-period miss rate for (station, transfer-window bucket, hour), plugged into the same cost model. |
| (e) Logistic regression | Same features as the main model, plugged into the same cost model. |

Main model: **LightGBM classifier for P(miss)**, calibrated, plugged into the cost model.

## 4. Approaches considered and rejected

| Approach | Why rejected |
|---|---|
| Deep learning / sequence models first | Tabular data, ~10⁵–10⁶ rows, and the value is in framing and calibration. GBDTs are the strong tabular default. DL adds tuning cost and hurts explainability. Revisit only if GBDT plateaus and the data grows a lot. |
| Pure regression on delay, then a threshold | Optimizes the wrong thing. The miss is a function of *two* delays, a cancellation and a min transfer time. Regressing one delay ignores B being held, gives no calibrated probability, and errors in the tail (where misses live) are exactly what MAE under-weights. |
| Survival / full arrival-time distribution | Principled (it yields P(miss) for any MCT), but heavier, and the calibrated classifier answers the decision question directly. Listed as a future extension. |
| Full journey planner (RAPTOR) for re-routing alternatives | The real "alternative" can be a re-route via another hub. Out of scope. Alternatives here are limited to departures from S that reach T. |
| Accuracy / F1 as the target | Classes are imbalanced, and the costs are asymmetric and depend on L and E. A cost-weighted decision on calibrated probabilities is the right objective. |
| Random train/test split | Leaks through shared days, weather and disruptions. Time-based split only. |

## 5. Data

- **Historical:** HF `piebro/deutsche-bahn-data` (CC BY 4.0), `monthly_processed_data` (one row per train
  stop, final planned/changed times, cancellation flags, ride id, stop index). Near-national coverage
  (5,301 stations in Mar 2026), so A's earlier stops and B's later stops are both present.
  Subset: 4 hubs (Leipzig Hbf, Erfurt Hbf, Halle (Saale) Hbf, Berlin Hauptbahnhof; Berlin has 3 EVA
  numbers, upper level / deep level / S-Bahn, treated as one station), **Aug–Oct 2026**, all stops of
  every ride that calls at a hub.
- **Live:** DB Timetables API v1 (`plan`, `fchg`), same hubs, hourly. This feeds monitoring and shadow
  mode. History comes from HF.
- **Weather:** DWD via Bright Sky, hourly, nearest station to each hub.

### 5.1 What "known at prediction time" means here (most important data decision)

`t_pred = planned_arrival_A(S) − lead`. A feature may only use information **timestamped ≤ t_pred**:
- *known delay of A* = A's realized delay at its last stop with actual departure ≤ t_pred (0 and a flag if
  A hasn't started yet);
- *known delay of B* = the same for B (most B's start at S or nearby, so this is often "not yet
  running", which is itself informative);
- historical rates from the training period only, out-of-fold inside training.

**Train/serve skew risk:** live, the natural input is DB's *forecast* at S. In training we have the
*last realized upstream delay*. They are close (DB's forecast mostly propagates the last realized
delay) but not identical. Live features are computed the same way, from the change feed's
latest realized times, and the skew is monitored (Phase 9).

## 6. Labels

- `missed = 1` if `actual_dep_B(S) − actual_arr_A(S) < MCT(S)`, OR B is cancelled at S, OR A's arrival at S is
  cancelled. B being held or late is handled naturally, because we compare *actual* times.
- `MCT(S)` default 5 min, Berlin Hbf 8 min (multi-level station). Configurable and sensitivity-tested.
- `L` (lost time if missed): earliest **actual** arrival at T among departures from S with actual
  departure ≥ actual_arr_A + MCT, not cancelled, calling at T later on the same ride, minus B's
  actual arrival at T. Censored at 240 min if nothing is found. *T = B's next stop*; see DECISIONS.

## 7. Assumptions

1. Times in the dataset are local (Europe/Berlin), minute resolution.
2. `*_change_time` on past stops is the realized time.
3. Every candidate connection is a plausible passenger itinerary. In reality the demand is unknown
   (no passenger counts), so all connections are equally weighted. *This is a big assumption.*
4. MCT is constant per station (no platform-pair walking times).
5. The w_late weight (§1.1).

## 8. Risks

| Risk | Mitigation |
|---|---|
| Leakage via "actual" times | pytest that recomputes every feature with all post-t_pred rows removed and asserts equality |
| Too-good results | Treat AUC > 0.95 as a bug until proven otherwise |
| Candidate connections dominated by S-Bahn pairs, nobody's real itinerary | Report by train-type pair. Exclude backtracking pairs and same-line pairs |
| Strikes / storms in the test month | Report with and without disruption days. Failure analysis |
| Live API creds unavailable | Live code tested on fixture XML. Live jobs skip cleanly without secrets |
| Weak result | Reported as-is. A weak, honest result beats a fabricated one |

## 9. Out of scope

Re-routing via other hubs; passenger-flow weighting; seat reservations / ticket rules (Zugbindung);
platform-specific walking times; operator-side "hold B" decisions; buses and replacement services as
A or B (they can still count as fallback C).

## 10. Open questions (answered in later phases, logged in DECISIONS)

- How large is the candidate set, and does it need sampling? (Phase 4)
- Is weather predictive at all over three summer/autumn months? (Phase 5/6)
- Is the w_late weight a credible way to sell this, or should the product be framed as "probability
  communication" with the decision as a secondary output? (Phase 7/11)
