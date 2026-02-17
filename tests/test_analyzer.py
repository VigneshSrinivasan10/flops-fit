"""Tests for the scaling law analyzer module."""

import json

import pytest
import numpy as np
import pandas as pd

from flops_fit.analyzer import ScalingLawAnalyzer, ScalingAnalysis, PowerLawFit
from flops_fit.planner import SweepPlanner
from flops_fit.trainer import TrainingRunner


class TestPowerLawFit:
    """Test suite for PowerLawFit."""

    def test_predict(self):
        """Test power law prediction."""
        fit = PowerLawFit(
            name="test",
            coefficient_k=2.0,
            exponent_a=0.5,
            r_squared=0.99,
        )

        x = np.array([1, 4, 9, 16])
        y = fit.predict(x)

        # y = 2 * x^0.5 = 2 * sqrt(x)
        expected = 2 * np.sqrt(x)
        np.testing.assert_allclose(y, expected)

    def test_to_dict(self):
        """Test serialization."""
        fit = PowerLawFit(
            name="N_opt",
            coefficient_k=1e-8,
            exponent_a=0.73,
            r_squared=0.95,
        )

        d = fit.to_dict()

        assert d["name"] == "N_opt"
        assert d["coefficient_k"] == pytest.approx(1e-8)
        assert d["exponent_a"] == pytest.approx(0.73)
        assert "formula" in d


class TestScalingLawAnalyzer:
    """Test suite for ScalingLawAnalyzer."""

    def test_fit_power_law(self, tmp_path):
        """Test power law fitting."""
        # Create analyzer
        analyzer = ScalingLawAnalyzer(
            results_path=tmp_path / "results.json",
            output_dir=tmp_path / "analysis",
        )

        # Generate synthetic data: y = 0.1 * x^0.5
        x = np.logspace(10, 20, 20)
        y = 0.1 * np.power(x, 0.5) * np.random.uniform(0.95, 1.05, 20)

        fit = analyzer.fit_power_law(x, y, "test")

        # Should recover approximate exponent
        assert fit.exponent_a == pytest.approx(0.5, abs=0.1)
        assert fit.r_squared > 0.9

    def test_fit_power_law_handles_invalid(self, tmp_path):
        """Test that fitting handles invalid values gracefully."""
        analyzer = ScalingLawAnalyzer(
            results_path=tmp_path / "results.json",
            output_dir=tmp_path / "analysis",
        )

        # Include some invalid values
        x = np.array([1e10, 0, 1e12, -1, 1e14, np.nan])
        y = np.array([100, 50, 1000, 200, 10000, 500])

        fit = analyzer.fit_power_law(x, y, "test")

        # Should still produce a fit (using valid points only)
        assert fit.r_squared >= 0

    def test_analyze_produces_valid_fits(self, tmp_path):
        """End-to-end test: planner -> trainer -> analyzer produces valid power law fits."""
        # Use 5 compute budgets so optimal N varies across budgets (3 is too few
        # and can produce the same optimal model size for every budget, giving r²=0).
        planner = SweepPlanner(
            min_flops=1e17,
            max_flops=1e21,
            num_compute_budgets=5,
            num_model_sizes=7,
        )
        sweep_path = tmp_path / "sweep.json"
        planner.save_sweep(sweep_path)

        # Run mock training
        runner = TrainingRunner(
            mode="mock",
            sweep_path=sweep_path,
            output_dir=tmp_path,
        )
        np.random.seed(42)
        runner.run_sweep(resume=False)

        # Analyze results
        analyzer = ScalingLawAnalyzer(
            results_path=tmp_path / "results.json",
            output_dir=tmp_path / "analysis",
        )
        analysis = analyzer.analyze()

        assert analysis.n_opt_fit.r_squared > 0
        assert analysis.d_opt_fit.r_squared > 0
        assert analysis.l_opt_fit.r_squared > 0
        assert len(analysis.optimal_points) > 0
        assert analysis.optimal_ratio is not None
        assert analysis.optimal_ratio > 0
        assert (tmp_path / "analysis" / "scaling_laws.json").exists()

    def test_predict_returns_optimal_config(self, tmp_path):
        """End-to-end test: analyze then predict returns valid optimal config."""
        planner = SweepPlanner(
            min_flops=1e17,
            max_flops=1e21,
            num_compute_budgets=5,
            num_model_sizes=7,
        )
        sweep_path = tmp_path / "sweep.json"
        planner.save_sweep(sweep_path)

        runner = TrainingRunner(
            mode="mock",
            sweep_path=sweep_path,
            output_dir=tmp_path,
        )
        np.random.seed(42)
        runner.run_sweep(resume=False)

        analyzer = ScalingLawAnalyzer(
            results_path=tmp_path / "results.json",
            output_dir=tmp_path / "analysis",
        )
        analyzer.analyze()

        # Predict for a new compute budget
        result = analyzer.predict(1e20)

        assert "target_compute" in result
        assert "optimal_params" in result
        assert "optimal_tokens" in result
        assert "expected_loss" in result
        assert "tokens_per_param" in result
        assert result["optimal_params"] > 0
        assert result["optimal_tokens"] > 0
        assert result["expected_loss"] > 0

    def test_find_optimal_per_budget_uses_2_decimal_rounding(self, tmp_path):
        """Characterization: analyzer uses np.round(np.log10(budget), 2) for bucketing.

        Visualizer uses 1-decimal -- known inconsistency.

        log10(1.001e17) = 17.0004, log10(1.004e17) = 17.0017 -- both round to
        17.00 at 2 decimals, so they should be in the SAME bucket.
        """
        analyzer = ScalingLawAnalyzer(
            results_path=tmp_path / "results.json",
            output_dir=tmp_path / "analysis",
        )

        df = pd.DataFrame([
            {
                "compute_budget": 1.001e17,
                "model_size": 1_000_000,
                "num_tokens": 10_000_000,
                "final_loss": 3.5,
            },
            {
                "compute_budget": 1.004e17,
                "model_size": 2_000_000,
                "num_tokens": 5_000_000,
                "final_loss": 3.2,
            },
        ])

        optimal = analyzer.find_optimal_per_budget(df)
        # Both budgets round to 17.00 at 2 decimals -> same bucket -> 1 row
        assert len(optimal) == 1

    def test_find_optimal_per_budget_selects_min_loss(self, tmp_path):
        """find_optimal_per_budget returns the experiment with minimum loss per bucket."""
        analyzer = ScalingLawAnalyzer(
            results_path=tmp_path / "results.json",
            output_dir=tmp_path / "analysis",
        )

        df = pd.DataFrame([
            {
                "compute_budget": 1e18,
                "model_size": 1_000_000,
                "num_tokens": 10_000_000,
                "final_loss": 3.5,
            },
            {
                "compute_budget": 1e18,
                "model_size": 2_000_000,
                "num_tokens": 5_000_000,
                "final_loss": 2.8,
            },
            {
                "compute_budget": 1e18,
                "model_size": 3_000_000,
                "num_tokens": 3_000_000,
                "final_loss": 3.1,
            },
        ])

        optimal = analyzer.find_optimal_per_budget(df)
        assert len(optimal) == 1
        assert optimal.iloc[0]["final_loss"] == pytest.approx(2.8)

    def test_load_results_filters_to_completed(self, tmp_path):
        """load_results only returns experiments with status 'completed'."""
        results = [
            {"experiment_id": "exp_0001", "compute_budget": 1e18, "model_size": 1e6,
             "num_tokens": 1e7, "final_loss": 3.0, "status": "completed"},
            {"experiment_id": "exp_0002", "compute_budget": 1e18, "model_size": 2e6,
             "num_tokens": 5e6, "final_loss": 2.8, "status": "completed"},
            {"experiment_id": "exp_0003", "compute_budget": 1e18, "model_size": 3e6,
             "num_tokens": 3e6, "final_loss": 3.1, "status": "completed"},
            {"experiment_id": "exp_0004", "compute_budget": 1e18, "model_size": 4e6,
             "num_tokens": 2e6, "final_loss": float("nan"), "status": "failed"},
        ]

        results_path = tmp_path / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f)

        analyzer = ScalingLawAnalyzer(
            results_path=results_path,
            output_dir=tmp_path / "analysis",
        )
        df = analyzer.load_results()
        assert len(df) == 3

    def test_fit_power_law_requires_minimum_points(self, tmp_path):
        """fit_power_law raises ValueError when fewer than 2 valid points."""
        analyzer = ScalingLawAnalyzer(
            results_path=tmp_path / "results.json",
            output_dir=tmp_path / "analysis",
        )

        # Only 1 valid point (others are zero or negative and get filtered)
        x = np.array([1e10, 0, -1])
        y = np.array([100, 50, 200])

        with pytest.raises(ValueError, match="Not enough valid points"):
            analyzer.fit_power_law(x, y, "test")

    def test_analyzer_predict_includes_l_inf(self, tmp_path):
        """predict() expected_loss must include l_inf from saved scaling_laws.json.

        ScalingLawAnalyzer.predict() manually reconstructs the power law from
        the JSON dict. If it forgets to add l_inf, it returns the bare power-law
        value instead of l_inf + k * C^a.

        Setup: write a known scaling_laws.json with l_opt_fit.l_inf=1.5,
        coefficient_k=2.0, exponent_a=0.5.
        For target_compute=4.0: expected = 1.5 + 2.0 * 4.0^0.5 = 1.5 + 4.0 = 5.5.
        Without the fix the code returns 4.0, not 5.5.
        """
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Build a minimal scaling_laws.json with a known l_inf in l_opt_fit
        def _fit_dict(name, k, a, l_inf=None):
            return {
                "name": name,
                "coefficient_k": k,
                "exponent_a": a,
                "r_squared": 0.99,
                "k_ci": None,
                "a_ci": None,
                "l_inf": l_inf,
                "formula": f"{name} = {k} * C^{a}",
            }

        scaling_laws = {
            "n_opt_fit": _fit_dict("N_opt", k=1.0, a=0.5),
            "d_opt_fit": _fit_dict("D_opt", k=1.0, a=0.5),
            "l_opt_fit": _fit_dict("L_opt", k=2.0, a=0.5, l_inf=1.5),
            "optimal_points": [],
            "optimal_ratio": 20.0,
        }

        scaling_laws_path = analysis_dir / "scaling_laws.json"
        with open(scaling_laws_path, "w") as f:
            json.dump(scaling_laws, f)

        analyzer = ScalingLawAnalyzer(
            results_path=tmp_path / "results.json",
            output_dir=analysis_dir,
        )

        result = analyzer.predict(4.0)

        # expected_loss must be l_inf + k * C^a = 1.5 + 2.0 * 4.0^0.5 = 5.5
        # Current (unfixed) code returns 2.0 * 4.0^0.5 = 4.0 — this assertion fails.
        assert result["expected_loss"] == pytest.approx(5.5)


class TestScalingAnalysis:
    """Test suite for ScalingAnalysis dataclass."""

    def test_predict_optimal_size(self):
        """predict_optimal_size returns correct values from known power law fits."""
        # N_opt = 1.0 * C^0.5
        n_fit = PowerLawFit(name="N_opt", coefficient_k=1.0, exponent_a=0.5, r_squared=0.99)
        d_fit = PowerLawFit(name="D_opt", coefficient_k=1.0, exponent_a=0.5, r_squared=0.99)
        l_fit = PowerLawFit(name="L_opt", coefficient_k=1.0, exponent_a=0.5, r_squared=0.99)

        analysis = ScalingAnalysis(
            n_opt_fit=n_fit,
            d_opt_fit=d_fit,
            l_opt_fit=l_fit,
            optimal_points=[],
            optimal_ratio=1.0,
        )

        result = analysis.predict_optimal_size(1e20)

        # N_opt = 1.0 * (1e20)^0.5 = 1e10
        assert result["optimal_params"] == int(1e10)
        assert result["optimal_tokens"] == int(1e10)
        assert result["target_compute"] == 1e20

    def test_predict_optimal_size_uses_l_inf_for_loss(self):
        """predict_optimal_size() propagates l_inf through to expected_loss."""
        l_fit = PowerLawFit(name="L_opt", coefficient_k=2.0, exponent_a=0.5, r_squared=0.99, l_inf=1.0)
        n_fit = PowerLawFit(name="N_opt", coefficient_k=1.0, exponent_a=0.5, r_squared=0.99)
        d_fit = PowerLawFit(name="D_opt", coefficient_k=1.0, exponent_a=0.5, r_squared=0.99)
        analysis = ScalingAnalysis(
            n_opt_fit=n_fit,
            d_opt_fit=d_fit,
            l_opt_fit=l_fit,
            optimal_points=[],
            optimal_ratio=1.0,
        )
        result = analysis.predict_optimal_size(4.0)
        # l_inf=1.0 + k=2.0 * 4.0^0.5 = 1.0 + 4.0 = 5.0
        assert result["expected_loss"] == pytest.approx(5.0)

    def test_to_dict(self):
        """to_dict returns correct structure with all expected keys."""
        n_fit = PowerLawFit(name="N_opt", coefficient_k=1.0, exponent_a=0.5, r_squared=0.99)
        d_fit = PowerLawFit(name="D_opt", coefficient_k=2.0, exponent_a=0.6, r_squared=0.98)
        l_fit = PowerLawFit(name="L_opt", coefficient_k=3.0, exponent_a=-0.1, r_squared=0.95)

        analysis = ScalingAnalysis(
            n_opt_fit=n_fit,
            d_opt_fit=d_fit,
            l_opt_fit=l_fit,
            optimal_points=[{"compute_budget": 1e18, "model_size": 1e6}],
            optimal_ratio=20.0,
        )

        d = analysis.to_dict()

        assert "n_opt_fit" in d
        assert "d_opt_fit" in d
        assert "l_opt_fit" in d
        assert "optimal_points" in d
        assert "optimal_ratio" in d
        assert d["optimal_ratio"] == 20.0
        assert len(d["optimal_points"]) == 1


class TestPowerLawFitLinearSpace:
    """Test suite for linear-space power law fitting with irreducible loss (l_inf)."""

    def test_fit_power_law_with_irreducible_loss(self, tmp_path):
        """fit_power_law recovers l_inf and exponent from data with a known baseline.

        Uses x in [10, 1e5] so that l_inf=1.5 is a significant fraction of y_min
        (~83%), making it detectable by nonlinear least squares.
        """
        analyzer = ScalingLawAnalyzer(
            results_path=tmp_path / "r.json",
            output_dir=tmp_path / "a",
        )
        np.random.seed(0)
        x = np.logspace(1, 5, 20)
        y = 1.5 + 0.1 * np.power(x, 0.5) * np.random.uniform(0.95, 1.05, 20)
        fit = analyzer.fit_power_law(x, y, "L_opt")
        assert fit.l_inf is not None
        assert fit.l_inf == pytest.approx(1.5, abs=0.5)
        assert fit.exponent_a == pytest.approx(0.5, abs=0.15)
        assert fit.r_squared > 0.95

    def test_fit_power_law_l_inf_stored_in_result(self, tmp_path):
        """fit_power_law always returns a PowerLawFit with l_inf attribute set.

        Uses x in [1e2, 1e5] so l_inf=2.0 is detectable against the trend.
        """
        analyzer = ScalingLawAnalyzer(
            results_path=tmp_path / "r.json",
            output_dir=tmp_path / "a",
        )
        np.random.seed(1)
        x = np.logspace(2, 5, 15)
        y = 2.0 + 0.05 * np.power(x, 0.4)
        fit = analyzer.fit_power_law(x, y, "test")
        assert hasattr(fit, "l_inf")
        assert fit.l_inf is not None
        assert fit.l_inf >= 0

    def test_power_law_fit_predict_with_l_inf(self):
        """predict() adds l_inf when l_inf is set on the fit."""
        fit = PowerLawFit(
            name="L_opt",
            coefficient_k=2.0,
            exponent_a=0.5,
            r_squared=0.99,
            l_inf=1.0,
        )
        result = fit.predict(np.array([4.0]))
        # 1.0 + 2.0 * sqrt(4) = 1.0 + 4.0 = 5.0
        assert result[0] == pytest.approx(5.0)

    def test_power_law_fit_predict_without_l_inf_backward_compat(self):
        """predict() is unchanged when l_inf is None (backward compatibility)."""
        fit = PowerLawFit(
            name="N_opt",
            coefficient_k=2.0,
            exponent_a=0.5,
            r_squared=0.99,
        )
        result = fit.predict(np.array([4.0]))
        # 2.0 * sqrt(4) = 4.0
        assert result[0] == pytest.approx(4.0)

    def test_power_law_fit_to_dict_includes_l_inf(self):
        """to_dict() includes l_inf key when l_inf is set."""
        fit = PowerLawFit(
            name="L_opt",
            coefficient_k=1.0,
            exponent_a=-0.1,
            r_squared=0.95,
            l_inf=2.5,
        )
        d = fit.to_dict()
        assert "l_inf" in d
        assert d["l_inf"] == pytest.approx(2.5)


class TestChinchillaTable:
    """Test suite for ScalingAnalysis.chinchilla_table()."""

    def _make_analysis(self) -> ScalingAnalysis:
        n_fit = PowerLawFit(name="N_opt", coefficient_k=1e6, exponent_a=0.5, r_squared=0.99)
        d_fit = PowerLawFit(name="D_opt", coefficient_k=2e7, exponent_a=0.5, r_squared=0.99)
        l_fit = PowerLawFit(name="L_opt", coefficient_k=0.1, exponent_a=-0.05, r_squared=0.99, l_inf=1.5)
        return ScalingAnalysis(
            n_opt_fit=n_fit,
            d_opt_fit=d_fit,
            l_opt_fit=l_fit,
            optimal_points=[],
            optimal_ratio=20.0,
        )

    def test_chinchilla_table_returns_string(self):
        """chinchilla_table() returns a string."""
        analysis = self._make_analysis()
        table = analysis.chinchilla_table()
        assert isinstance(table, str)

    def test_chinchilla_table_default_has_9_rows(self):
        """chinchilla_table() default produces 11 lines (header + separator + 9 data rows)."""
        analysis = self._make_analysis()
        table = analysis.chinchilla_table()
        lines = table.strip().split("\n")
        # header + separator + 9 data rows = 11 lines
        assert len(lines) == 11

    def test_chinchilla_table_custom_budgets(self):
        """chinchilla_table() with custom budgets produces correct number of data rows."""
        analysis = self._make_analysis()
        table = analysis.chinchilla_table(compute_budgets=[1e18, 1e20, 1e22])
        lines = table.strip().split("\n")
        # header + separator + 3 data rows = 5 lines
        assert len(lines) == 5

    def test_chinchilla_table_contains_header(self):
        """chinchilla_table() output contains all expected column headers."""
        analysis = self._make_analysis()
        table = analysis.chinchilla_table()
        assert "Compute Budget" in table
        assert "Optimal N" in table
        assert "Optimal D" in table
        assert "Predicted Loss" in table

    def test_chinchilla_table_values_from_predict(self):
        """chinchilla_table() row values match predict_optimal_size() output."""
        analysis = self._make_analysis()
        table = analysis.chinchilla_table(compute_budgets=[1e20])
        pred = analysis.predict_optimal_size(1e20)
        # Expected loss should appear in table (formatted to 4 decimals)
        assert f"{pred['expected_loss']:.4f}" in table

    def test_chinchilla_table_end_to_end(self, tmp_path):
        """End-to-end: planner -> trainer (mock) -> analyzer -> chinchilla_table()."""
        planner = SweepPlanner(
            min_flops=1e17,
            max_flops=1e21,
            num_compute_budgets=5,
            num_model_sizes=7,
        )
        sweep_path = tmp_path / "sweep.json"
        planner.save_sweep(sweep_path)

        runner = TrainingRunner(
            mode="mock",
            sweep_path=sweep_path,
            output_dir=tmp_path,
        )
        np.random.seed(42)
        runner.run_sweep(resume=False)

        analyzer = ScalingLawAnalyzer(
            results_path=tmp_path / "results.json",
            output_dir=tmp_path / "analysis",
        )
        analysis = analyzer.analyze()
        table = analysis.chinchilla_table(compute_budgets=[1e18, 1e19, 1e20])
        assert isinstance(table, str)
        assert "Compute Budget" in table
        assert len(table.split("\n")) >= 5


class TestOutlierDetection:
    """Test suite for IQR-based outlier detection in fit_power_law()."""

    def test_outlier_detection_excludes_anomalous_points(self, tmp_path):
        """Fitting with exclude_outliers=True achieves higher R² than without it on contaminated data."""
        analyzer = ScalingLawAnalyzer(
            results_path=tmp_path / "r.json",
            output_dir=tmp_path / "a",
        )
        np.random.seed(42)
        x = np.logspace(10, 20, 20)
        y_clean = 1.5 + 0.1 * np.power(x, 0.5) * np.random.uniform(0.97, 1.03, 20)
        y_with_outliers = y_clean.copy()
        y_with_outliers[5] = y_clean[5] * 10
        y_with_outliers[12] = y_clean[12] * 8

        fit_all = analyzer.fit_power_law(x, y_with_outliers, "test", exclude_outliers=False)
        fit_clean = analyzer.fit_power_law(x, y_with_outliers, "test", exclude_outliers=True)

        # Outlier removal should improve fit quality
        assert fit_clean.r_squared > fit_all.r_squared

    def test_outlier_detection_disabled_when_exclude_outliers_false(self, tmp_path):
        """fit_all (exclude_outliers=False) should have lower R² than fit_clean on contaminated data."""
        analyzer = ScalingLawAnalyzer(
            results_path=tmp_path / "r.json",
            output_dir=tmp_path / "a",
        )
        np.random.seed(42)
        x = np.logspace(10, 20, 20)
        y_clean = 1.5 + 0.1 * np.power(x, 0.5) * np.random.uniform(0.97, 1.03, 20)
        y_with_outliers = y_clean.copy()
        y_with_outliers[5] = y_clean[5] * 10
        y_with_outliers[12] = y_clean[12] * 8

        fit_all = analyzer.fit_power_law(x, y_with_outliers, "test", exclude_outliers=False)
        fit_clean = analyzer.fit_power_law(x, y_with_outliers, "test", exclude_outliers=True)

        assert fit_all.r_squared < fit_clean.r_squared

    def test_outlier_detection_skipped_when_fewer_than_5_points(self, tmp_path):
        """fit_power_law with fewer than 5 points skips outlier detection and succeeds."""
        analyzer = ScalingLawAnalyzer(
            results_path=tmp_path / "r.json",
            output_dir=tmp_path / "a",
        )
        x = np.logspace(10, 14, 4)
        y = 1.5 + 0.1 * np.power(x, 0.5)
        # Should not raise -- just fits without outlier removal
        fit = analyzer.fit_power_law(x, y, "test", exclude_outliers=True)
        assert fit is not None
