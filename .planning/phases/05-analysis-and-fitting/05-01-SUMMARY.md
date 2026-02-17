---
phase: 05-analysis-and-fitting
plan: "01"
subsystem: analysis
tags: [scipy, power-law, nonlinear-least-squares, outlier-detection, scaling-laws]

# Dependency graph
requires:
  - phase: 04-training-engine
    provides: "Training results as list[dict] with compute_budget, model_size, num_tokens, final_loss"
provides:
  - "Linear-space NLS power law fitting: y = L_inf + k * x^a via scipy.optimize.least_squares"
  - "IQR-based outlier detection on initial fit residuals before final fitting"
  - "PowerLawFit.l_inf field for irreducible loss baseline"
  - "PowerLawFit.predict() updated to add l_inf when present"
affects:
  - 05-analysis-and-fitting
  - future phases using ScalingAnalysis.predict_optimal_size

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-pass NLS fitting: rough log-space pass for outlier detection, then linear-space pass for final fit"
    - "Log-parametrize k as log10(k) in optimization for scale-invariant bounds [-10, 5]"
    - "IQR-based outlier detection: Q1-1.5*IQR to Q3+1.5*IQR on absolute residuals"
    - "l_inf >= 0 bound constraint enforces physically meaningful irreducible loss"

key-files:
  created: []
  modified:
    - src/flops_fit/analyzer.py
    - tests/test_analyzer.py

key-decisions:
  - "Linear-space NLS replaces log-space regression: unbiased when loss has additive baseline (irreducible entropy)"
  - "Parametrize optimization as [log10(k), a, l_inf] for scale-invariant bounds; k = 10^log_k"
  - "Initial guess for log_k from polyfit on log-log data (bounds-safe, clipped to [-10+eps, 5-eps])"
  - "IQR outlier removal uses rough log-space fit residuals (not final residuals) for speed and numerical safety"
  - "Test x range adjusted from logspace(10,20) to logspace(1,5) for l_inf recovery test: l_inf=1.5 is ~83% of y_min at lower scale vs 0.015% at higher scale (mathematically undetectable)"

patterns-established:
  - "TDD red-green cycle: write failing tests first, then implement to pass"
  - "Backward compat: l_inf=None default preserves existing predict() behavior (k * x^a)"
  - "Auto-fix rule 1: test data corrected when plan spec creates mathematically infeasible assertions"

# Metrics
duration: 18min
completed: 2026-02-17
---

# Phase 5 Plan 01: Linear-Space Power Law Fitting with L_inf and IQR Outlier Detection Summary

**Refactored fit_power_law() from log-space linear regression to linear-space NLS (scipy.optimize.least_squares) with explicit irreducible loss term and two-pass IQR outlier detection; 152 tests passing.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-02-17T12:05:50Z
- **Completed:** 2026-02-17T12:23:50Z
- **Tasks:** 2 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments
- Replaced log-space linear regression in `fit_power_law()` with linear-space `scipy.optimize.least_squares` fitting model `y = L_inf + k * x^a`
- Added `l_inf: float | None = None` field to `PowerLawFit` dataclass; `predict()` adds l_inf when set; `to_dict()` includes l_inf and updated formula string
- Added two-pass outlier detection: rough log-space fit computes IQR residuals, points outside [Q1-1.5*IQR, Q3+1.5*IQR] excluded before final fit
- 8 new tests added (TestPowerLawFitLinearSpace + TestOutlierDetection), all 152 tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: RED - Write failing tests for linear-space fitting and outlier detection** - `bce3807` (test)
2. **Task 2: GREEN - Implement linear-space fitting and outlier detection in analyzer.py** - `1d8024a` (feat)

**Plan metadata:** (this commit)

_Note: TDD tasks have two commits (test -> feat)_

## Files Created/Modified
- `src/flops_fit/analyzer.py` - PowerLawFit.l_inf field, predict/to_dict updated, fit_power_law() replaced with linear-space NLS + IQR outlier detection
- `tests/test_analyzer.py` - Added TestPowerLawFitLinearSpace (5 tests) and TestOutlierDetection (3 tests)

## Decisions Made
- Linear-space NLS over log-space regression: avoids bias from additive irreducible loss baseline (Chinchilla-standard approach)
- Parametrize optimization as `[log10(k), a, l_inf]` with bounds `[-10, 5] x [-1, 2] x [0, inf]` for scale-invariant optimization
- Initial guess derived from `np.polyfit` on log-log data, clamped within bounds, for numerically safe initialization
- `exclude_outliers=True` default makes clean fitting the happy path; pass `exclude_outliers=False` to disable
- Outlier detection skipped when fewer than 5 points (not enough for reliable IQR statistics)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed infeasible test assertion for l_inf recovery**
- **Found during:** Task 2 (GREEN implementation)
- **Issue:** Plan specified `x = np.logspace(10, 20, 20)` for `test_fit_power_law_with_irreducible_loss`. At this x-scale, `l_inf=1.5` represents ~0.015% of `y_min=10050`. No NLS solver can recover such a negligible additive constant; optimizer correctly converges to `l_inf=0`. Also affected `test_fit_power_law_l_inf_stored_in_result` with `logspace(12, 20, 15)`.
- **Fix:** Changed x range to `logspace(1, 5, 20)` in both tests (x in [10, 1e5]). At this scale `l_inf=1.5` is ~83% of `y_min`, making it physically detectable by the optimizer.
- **Files modified:** tests/test_analyzer.py
- **Verification:** `test_fit_power_law_with_irreducible_loss` passes with `fit.l_inf == approx(1.5, abs=0.5)`
- **Committed in:** `1d8024a` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - test data bug)
**Impact on plan:** Required fix for mathematical correctness. x range change preserves the behavioral intent (l_inf recovery from data with known baseline) while making the test physically feasible. No scope creep.

## Issues Encountered

- **Initial guess out of bounds**: First implementation of `fit_power_law()` used `k_init_arg = mean(y) - min(y)` which produces `log10(k_init_arg) >> 5` for large-scale y values, causing `scipy.optimize.least_squares` to raise `ValueError: Initial guess is outside of provided bounds`. Fixed by computing initial `log_k` from `np.polyfit` on log-log data and clamping to bounds.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `fit_power_law()` now produces accurate fits for data with irreducible loss baseline, ready for Phase 5 plan 02
- `PowerLawFit.l_inf` available for downstream `predict_optimal_size()` calls
- All 152 tests passing; no regressions

---
*Phase: 05-analysis-and-fitting*
*Completed: 2026-02-17*
