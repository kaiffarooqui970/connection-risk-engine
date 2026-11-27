# POSTMORTEM: real failures during the build

Format per incident: what happened · impact · how it was detected · fix · lesson.
Only things that actually happened in this build are listed.

## 1. Prior of the smoothed historical rate leaked the row's own week (Phase 5)

- **What:** `hist_*` features use an m-estimate `(k + m·p0)/(n + m)`. Counts `k, n` were correctly out-of-fold by ISO
  week, but the prior `p0` was the mean over *all* training rows, including the row's own week.
- **Impact:** a small leak into training features (the global mean moves only slightly), but it's exactly the kind that
  produces optimistic validation numbers and then degrades silently in production. No model had been trained on it.
- **Detected:** `test_hist_rates_out_of_fold_for_training_rows` flips one week's labels and asserts that week's feature
  values don't move. They did (0.333 → 0.344).
- **Fix:** the prior is out-of-fold too (`tot − own week`).
- **Lesson:** "out-of-fold" has to cover *every* statistic computed from labels, including smoothing priors. Test
  the invariant (perturb labels, assert no change), not the implementation.

## 2. Calibration step made the test month *less* calibrated (Phase 6)

- **What:** Platt scaling won the calibration comparison on validation (2-fold by-day CV: ECE 0.0145 → 0.0030).
  On the October test month it raised ECE from 0.0043 (raw LightGBM) to 0.0195.
- **Impact:** still within the 0.03 guardrail overall, but some segments (long-distance → anything) exceed 0.03.
  Because the decision is a *threshold on P* (P > 0.5), miscalibration directly moves decisions.
- **Detected:** the test-set reliability comparison of raw vs calibrated (`reports/model_metrics.json`).
- **Root cause:** base-rate shift. Miss rate 21.1% in the validation fortnight vs 18.5% in October. A post-hoc calibrator
  fitted on a short window learns that window's level; the raw model, trained on 6 weeks, was closer to October's level.
- **Fix (not applied retroactively):** we kept the pre-registered choice to avoid selecting on test. Going forward:
  (a) fit the calibrator on a longer window, or with a recency-weighted refit; (b) monitor ECE daily (Phase 9), which is
  exactly the alarm for this; (c) treat "calibration" as a model component that is re-validated, not a free win.
- **Lesson:** calibration is fitted to a *period*. On non-stationary data a calibrator can hurt; it has to be monitored like the model.

## 3. App bundle export crashed on a duplicated column (Phase 10)

- **What:** `make app-bundle` failed with `KeyError: ['a_known_delay'] not in index`. `load_frame` already returns
  `a_known_delay`, and the export merged it in again from the feature table, so pandas renamed both copies to `_x`/`_y`.
- **Impact:** the app had no bundle (`app/bundle/` was empty), so the dashboard could not start. No metrics were affected.
- **Detected:** running the target end to end for the first time after the Phase 7 frame gained the column.
- **Fix:** take the column from the sim frame only (`src/cre/evaluate/export_app.py`). App smoke-tested headless with
  `streamlit.testing.AppTest`, including the stale-data toggle.
- **Lesson:** every report-producing script needs a `make` target that is actually run. Steps only ever run by hand rot
  silently when an upstream frame changes.
