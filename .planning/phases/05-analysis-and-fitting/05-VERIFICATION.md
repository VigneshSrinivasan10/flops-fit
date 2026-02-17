---
phase: 05-analysis-and-fitting
verified: 2026-02-17T15:00:00Z
status: gaps_found
score: 3/4 must-haves verified
gaps:
  - truth: "Library fits N_opt, D_opt, and L_opt vs compute power laws with R-squared values"
    status: verified
    reason: "All three fits computed and stored in ScalingAnalysis"
    evidence:
      - "fit_power_law(C, N, 'N_opt'), fit_power_law(C, D, 'D_opt'), fit_power_law(C, L, 'L_opt') called in analyze() lines 375-377"
      - "PowerLawFit.r_squared computed in linear space on inlier points (lines 339-341)"
      - "All 27 tests passing; test_analyze_produces_valid_fits verifies all three fits have r_squared > 0"
  - truth: "Outlier experiments are automatically detected and excluded before fitting"
    status: verified
    reason: "Two-pass IQR outlier detection implemented and tested"
    evidence:
      - "fit_power_law() implements two-pass fitting with IQR outlier detection (lines 278-310)"
      - "Default exclude_outliers=True makes outlier detection the happy path"
      - "Tests TestOutlierDetection validates exclusion: test_outlier_detection_excludes_anomalous_points shows fit_clean.r_squared > fit_all.r_squared"
      - "3 outlier detection tests, all passing"
  - truth: "Chinchilla table output shows optimal N, D, and predicted loss for a range of compute budgets"
    status: verified
    reason: "chinchilla_table() method implemented and thoroughly tested"
    evidence:
      - "ScalingAnalysis.chinchilla_table() method at lines 117-163 returns markdown table string"
      - "Header contains 'Compute Budget | Optimal N | Optimal D | D/N Ratio | Predicted Loss'"
      - "Defaults to np.logspace(18, 22, 9) (9 log-spaced budgets) per spec"
      - "Custom compute_budgets list supported"
      - "6 tests in TestChinchillaTable all passing; test_chinchilla_table_default_has_9_rows confirms 11 lines (header + separator + 9 data rows)"
  - truth: "Fitting uses linear-space nonlinear least squares (not log-space) with irreducible loss term"
    status: verified
    reason: "scipy.optimize.least_squares with y = L_inf + k*x^a model"
    evidence:
      - "fit_power_law() uses scipy.optimize.least_squares at lines 329-333"
      - "Residual function: y_pred = l_inf + k * x^a (line 326)"
      - "NOT log-space: uses linear-space residuals (y_clean - y_pred) not log-space"
      - "Irreducible loss term l_inf fitted with bounds [0, inf] (line 332)"
      - "PowerLawFit.l_inf field added and stored (line 52, 352)"
      - "Tests TestPowerLawFitLinearSpace validates l_inf recovery and storage; all 5 tests passing"
---

# Phase 05: Analysis and Fitting Verification Report

**Phase Goal:** The library fits power laws to training results and produces Chinchilla-style predictions

**Verified:** 2026-02-17T15:00:00Z

**Status:** GAPS_FOUND (see below for critical issue)

**Score:** 3/4 must-haves verified + 1 critical issue found

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Library fits N_opt, D_opt, and L_opt vs compute power laws with R-squared values | ✓ VERIFIED | All three power laws fitted with r_squared computed in linear space; test suite validates fits have r_squared > 0 |
| 2 | Outlier experiments are automatically detected and excluded before fitting | ✓ VERIFIED | Two-pass IQR outlier detection in fit_power_law(); default exclude_outliers=True; comprehensive tests show clean fits achieve higher r_squared |
| 3 | Chinchilla table output shows optimal N, D, and predicted loss for a range of compute budgets | ✓ VERIFIED | ScalingAnalysis.chinchilla_table() returns markdown table with 5 columns and 9 default budgets; tests validate structure and formatting |
| 4 | Fitting uses linear-space nonlinear least squares (not log-space) with irreducible loss term | ✓ VERIFIED | scipy.optimize.least_squares with y = L_inf + k*x^a model; l_inf fitted and stored; tests validate recovery |

### Required Artifacts

| Artifact | Status | Details |
|----------|--------|---------|
| `src/flops_fit/analyzer.py` | ✓ VERIFIED | PowerLawFit.l_inf field, fit_power_law() with linear-space NLS + IQR outlier detection, ScalingAnalysis.chinchilla_table() method all present and substantive |
| `tests/test_analyzer.py` | ✓ VERIFIED | 27 tests total; 5 TestPowerLawFitLinearSpace, 3 TestOutlierDetection, 6 TestChinchillaTable, all passing |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `fit_power_law()` | `scipy.optimize.least_squares` | Linear-space residual function with l_inf + k*x^a | ✓ WIRED | Line 329-333: least_squares called with residuals_final function |
| `analyze()` | `fit_power_law()` | Three successive calls for N_opt, D_opt, L_opt fits | ✓ WIRED | Lines 375-377: three fit_power_law calls with exclude_outliers=True (default) |
| `ScalingAnalysis.chinchilla_table()` | `predict_optimal_size()` | Calls predict_optimal_size(budget) for each budget in list | ✓ WIRED | Line 149: for loop calls pred = self.predict_optimal_size(budget) |
| `predict_optimal_size()` | `l_opt_fit.predict()` | l_opt = self.l_opt_fit.predict(...) | ✓ WIRED | Line 107: l_opt computed via self.l_opt_fit.predict() which includes l_inf |

### Requirements Coverage

All phase success criteria from ROADMAP.md achieved:

1. ✓ ANLZ-01: Library fits N_opt, D_opt, L_opt with R-squared > 0 (test_analyze_produces_valid_fits)
2. ✓ ANLZ-02: Outlier detection working (test_outlier_detection_excludes_anomalous_points)
3. ✓ ANLZ-03: Chinchilla table output (test_chinchilla_table_end_to_end)
4. ✓ ANLZ-04: Linear-space fitting with l_inf (test_fit_power_law_with_irreducible_loss)

### Anti-Patterns Found

#### CRITICAL: ScalingLawAnalyzer.predict() doesn't account for l_inf

**Severity:** 🛑 **BLOCKER** — Expected loss predictions will be incorrect when using the `predict()` method

**File:** `src/flops_fit/analyzer.py` lines 436-438

**Issue:** The `ScalingLawAnalyzer.predict()` method loads saved JSON and manually reconstructs predictions WITHOUT adding l_inf:

```python
l_opt = l_fit["coefficient_k"] * (target_compute ** l_fit["exponent_a"])
# Missing: l_opt += l_fit.get("l_inf", 0)
```

The JSON contains l_inf (line 75 in to_dict()), but predict() ignores it. The correct path `ScalingAnalysis.predict_optimal_size()` (line 107) correctly uses `self.l_opt_fit.predict()` which includes l_inf, but this alternate API breaks the contract.

**Tests affected:** `test_predict_returns_optimal_config()` passes because it only validates key existence and positive values, not correctness of l_inf propagation.

**Impact:** If users call `analyzer.predict(1e20)` after analyze(), the expected_loss will be missing the irreducible loss baseline, producing incorrect Chinchilla predictions for that use path.

**Note:** Main code path uses `ScalingAnalysis.predict_optimal_size()` (lines 149, 479) which is correct. The `analyze()` method returns `ScalingAnalysis`, and the main CLI uses that object. The broken method is a secondary API that's less commonly used but still exported and tested.

### Test Results

**All 27 tests passing:**

```
tests/test_analyzer.py::TestPowerLawFit::test_predict PASSED
tests/test_analyzer.py::TestPowerLawFit::test_to_dict PASSED
tests/test_analyzer.py::TestScalingLawAnalyzer::test_fit_power_law PASSED
tests/test_analyzer.py::TestScalingLawAnalyzer::test_fit_power_law_handles_invalid PASSED
tests/test_analyzer.py::TestScalingLawAnalyzer::test_analyze_produces_valid_fits PASSED
tests/test_analyzer.py::TestScalingLawAnalyzer::test_predict_returns_optimal_config PASSED
tests/test_analyzer.py::TestScalingLawAnalyzer::test_find_optimal_per_budget_uses_2_decimal_rounding PASSED
tests/test_analyzer.py::TestScalingLawAnalyzer::test_find_optimal_per_budget_selects_min_loss PASSED
tests/test_analyzer.py::TestScalingLawAnalyzer::test_load_results_filters_to_completed PASSED
tests/test_analyzer.py::TestScalingLawAnalyzer::test_fit_power_law_requires_minimum_points PASSED
tests/test_analyzer.py::TestScalingAnalysis::test_predict_optimal_size PASSED
tests/test_analyzer.py::TestScalingAnalysis::test_predict_optimal_size_uses_l_inf_for_loss PASSED
tests/test_analyzer.py::TestScalingAnalysis::test_to_dict PASSED
tests/test_analyzer.py::TestPowerLawFitLinearSpace::test_fit_power_law_with_irreducible_loss PASSED
tests/test_analyzer.py::TestPowerLawFitLinearSpace::test_fit_power_law_l_inf_stored_in_result PASSED
tests/test_analyzer.py::TestPowerLawFitLinearSpace::test_power_law_fit_predict_with_l_inf PASSED
tests/test_analyzer.py::TestPowerLawFitLinearSpace::test_power_law_fit_predict_without_l_inf_backward_compat PASSED
tests/test_analyzer.py::TestPowerLawFitLinearSpace::test_power_law_fit_to_dict_includes_l_inf PASSED
tests/test_analyzer.py::TestChinchillaTable::test_chinchilla_table_returns_string PASSED
tests/test_analyzer.py::TestChinchillaTable::test_chinchilla_table_default_has_9_rows PASSED
tests/test_analyzer.py::TestChinchillaTable::test_chinchilla_table_custom_budgets PASSED
tests/test_analyzer.py::TestChinchillaTable::test_chinchilla_table_contains_header PASSED
tests/test_analyzer.py::TestChinchillaTable::test_chinchilla_table_values_from_predict PASSED
tests/test_analyzer.py::TestChinchillaTable::test_chinchilla_table_end_to_end PASSED
tests/test_analyzer.py::TestOutlierDetection::test_outlier_detection_excludes_anomalous_points PASSED
tests/test_analyzer.py::TestOutlierDetection::test_outlier_detection_disabled_when_exclude_outliers_false PASSED
tests/test_analyzer.py::TestOutlierDetection::test_outlier_detection_skipped_when_fewer_than_5_points PASSED
```

### Human Verification Required

None - all functionality verified programmatically via comprehensive test suite.

## Gap Summary

**1 blocker found that prevents full goal achievement:**

The `ScalingLawAnalyzer.predict()` method (lines 410-446) reconstructs predictions from saved JSON but ignores the l_inf field when computing expected_loss. This breaks the contract that "Fitting uses irreducible loss term" for users who call this method directly.

**However:** The main code path (analyze() returns ScalingAnalysis, main CLI uses predict_optimal_size()) is correct and fully functional. The issue only affects the secondary `predict()` API which is less commonly used but still exported as part of the class.

**Root cause:** When JSON is deserialized, the data dict contains l_inf, but lines 436-438 manually compute predictions without checking for it:

```python
# Line 438: Missing l_inf addition
l_opt = l_fit["coefficient_k"] * (target_compute ** l_fit["exponent_a"])
# Should be:
# l_opt = l_fit.get("l_inf", 0) + l_fit["coefficient_k"] * (target_compute ** l_fit["exponent_a"])
```

**To fix:**
- Add l_inf handling to lines 436-438 to match the main path logic
- Add test case `test_analyzer_predict_includes_l_inf()` to verify the fix
- Or: Deprecate this method in favor of always using `ScalingAnalysis.predict_optimal_size()` (the correct path)

---

_Verified: 2026-02-17_
_Verifier: Claude (gsd-verifier)_
