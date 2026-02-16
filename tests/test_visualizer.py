"""Tests for the scaling law visualizer module."""

import matplotlib
matplotlib.use("Agg")

import json

import pytest
import numpy as np
import matplotlib.pyplot as plt

from flops_fit.visualizer import ScalingVisualizer


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Reset matplotlib state after each test."""
    yield
    plt.close("all")
    plt.rcdefaults()


class TestScalingVisualizer:
    """Test suite for ScalingVisualizer."""

    def test_plot_isoflops_creates_file(self, tmp_path, results_json, analysis_json):
        """plot_isoflops saves a PNG file to disk."""
        viz = ScalingVisualizer(
            results_path=results_json,
            analysis_path=analysis_json,
            output_dir=tmp_path / "plots",
        )
        viz.plot_isoflops(save=True)

        png = tmp_path / "plots" / "isoflops_curves.png"
        assert png.exists()
        assert png.stat().st_size > 0

    def test_plot_scaling_laws_creates_file(self, tmp_path, results_json, analysis_json):
        """plot_scaling_laws saves a PNG file to disk."""
        viz = ScalingVisualizer(
            results_path=results_json,
            analysis_path=analysis_json,
            output_dir=tmp_path / "plots",
        )
        viz.plot_scaling_laws(save=True)

        png = tmp_path / "plots" / "scaling_laws.png"
        assert png.exists()
        assert png.stat().st_size > 0

    def test_plot_tokens_per_param_creates_file(self, tmp_path, results_json, analysis_json):
        """plot_tokens_per_param saves a PNG file to disk."""
        viz = ScalingVisualizer(
            results_path=results_json,
            analysis_path=analysis_json,
            output_dir=tmp_path / "plots",
        )
        viz.plot_tokens_per_param(save=True)

        png = tmp_path / "plots" / "tokens_per_param.png"
        assert png.exists()
        assert png.stat().st_size > 0

    def test_plot_all_creates_three_files(self, tmp_path, results_json, analysis_json):
        """plot_all returns 3 figures and creates 3 PNG files."""
        viz = ScalingVisualizer(
            results_path=results_json,
            analysis_path=analysis_json,
            output_dir=tmp_path / "plots",
        )
        figures = viz.plot_all(save=True)

        assert len(figures) == 3
        assert (tmp_path / "plots" / "isoflops_curves.png").exists()
        assert (tmp_path / "plots" / "scaling_laws.png").exists()
        assert (tmp_path / "plots" / "tokens_per_param.png").exists()

    def test_plot_isoflops_uses_1_decimal_buckets(self, tmp_path, results_json, analysis_json):
        """Characterization: visualizer uses 1-decimal bucket rounding vs analyzer's 2-decimal.

        Known inconsistency between visualizer and analyzer bucketing.
        """
        viz = ScalingVisualizer(
            results_path=results_json,
            analysis_path=analysis_json,
            output_dir=tmp_path / "plots",
        )
        df, _ = viz.load_data()

        # Visualizer uses 1-decimal rounding
        buckets_1dec = np.round(np.log10(df["compute_budget"]), 1)
        unique_1dec = buckets_1dec.nunique()

        # With budgets 1e17, 1e18, 1e19 -> log10 = 17.0, 18.0, 19.0 -> 3 buckets
        assert unique_1dec == 3

    def test_load_data_filters_completed(self, tmp_path):
        """load_data only returns completed experiments."""
        results = [
            {"experiment_id": "exp_0001", "compute_budget": 1e18, "model_size": 1e6,
             "num_tokens": 1e7, "final_loss": 3.0, "status": "completed"},
            {"experiment_id": "exp_0002", "compute_budget": 1e18, "model_size": 2e6,
             "num_tokens": 5e6, "final_loss": 2.8, "status": "completed"},
            {"experiment_id": "exp_0003", "compute_budget": 1e18, "model_size": 3e6,
             "num_tokens": 3e6, "final_loss": float("nan"), "status": "failed"},
        ]
        results_path = tmp_path / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f)

        viz = ScalingVisualizer(
            results_path=results_path,
            analysis_path=tmp_path / "nonexistent.json",
            output_dir=tmp_path / "plots",
        )
        df, _ = viz.load_data()
        assert len(df) == 2

    def test_load_data_loads_analysis(self, tmp_path, results_json, analysis_json):
        """load_data returns analysis dict when file exists."""
        viz = ScalingVisualizer(
            results_path=results_json,
            analysis_path=analysis_json,
            output_dir=tmp_path / "plots",
        )
        _, analysis = viz.load_data()

        assert analysis is not None
        assert "n_opt_fit" in analysis

    def test_load_data_analysis_optional(self, tmp_path, results_json):
        """load_data returns None for analysis when file doesn't exist."""
        viz = ScalingVisualizer(
            results_path=results_json,
            analysis_path=tmp_path / "nonexistent.json",
            output_dir=tmp_path / "plots",
        )
        _, analysis = viz.load_data()
        assert analysis is None

    def test_paper_style_sets_rcparams(self, tmp_path, results_json):
        """Paper style sets serif font and 300 dpi for saving."""
        ScalingVisualizer(
            results_path=results_json,
            analysis_path=tmp_path / "a.json",
            output_dir=tmp_path / "plots",
            style="paper",
        )
        assert plt.rcParams["font.family"] == ["serif"]
        assert plt.rcParams["savefig.dpi"] == 300.0

    def test_notebook_style_sets_rcparams(self, tmp_path, results_json):
        """Notebook style sets larger figure size."""
        ScalingVisualizer(
            results_path=results_json,
            analysis_path=tmp_path / "a.json",
            output_dir=tmp_path / "plots",
            style="notebook",
        )
        assert plt.rcParams["figure.figsize"] == [10.0, 7.0]
