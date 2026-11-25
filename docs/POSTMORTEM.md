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
