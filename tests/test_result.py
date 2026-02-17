import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import numpy as np
from flops_fit.result import Result
from flops_fit.analyzer import ScalingAnalysis, PowerLawFit
from flops_fit.visualizer import ScalingVisualizer


@pytest.fixture
def scaling_analysis():
    n_fit = PowerLawFit(name="N_opt", coefficient_k=1e-3, exponent_a=0.5, r_squared=0.99, l_inf=0.0)
    d_fit = PowerLawFit(name="D_opt", coefficient_k=2e-3, exponent_a=0.5, r_squared=0.99, l_inf=0.0)
    l_fit = PowerLawFit(name="L_opt", coefficient_k=5.0, exponent_a=-0.1, r_squared=0.95, l_inf=1.5)
    return ScalingAnalysis(
        n_opt_fit=n_fit,
        d_opt_fit=d_fit,
        l_opt_fit=l_fit,
        optimal_points=[],
        optimal_ratio=20.0,
    )


@pytest.fixture
def scaling_visualizer(tmp_path):
    import json
    results = [
        {"status": "completed", "compute_budget": 1e18, "model_size": 1000,
         "num_tokens": 20000, "final_loss": 2.5, "experiment_id": "e1",
         "actual_flops": 1e18, "wall_time_seconds": 1.0}
    ]
    (tmp_path / "results.json").write_text(json.dumps(results))
    analysis = {
        "n_opt_fit": {"coefficient_k": 1e-3, "exponent_a": 0.5},
        "d_opt_fit": {"coefficient_k": 2e-3, "exponent_a": 0.5},
        "l_opt_fit": {"coefficient_k": 5.0, "exponent_a": -0.1},
    }
    (tmp_path / "scaling_laws.json").write_text(json.dumps(analysis))
    return ScalingVisualizer(
        results_path=tmp_path / "results.json",
        analysis_path=tmp_path / "scaling_laws.json",
        output_dir=tmp_path / "plots",
    )


@pytest.fixture
def result(scaling_analysis, scaling_visualizer, tmp_path):
    return Result(
        analysis=scaling_analysis,
        visualizer=scaling_visualizer,
        output_dir=str(tmp_path),
        compute_budgets=[1e18, 1e19],
    )


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    yield
    plt.close("all")
    plt.rcdefaults()


class TestResult:
    def test_result_is_importable(self):
        from flops_fit.result import Result
        assert Result

    def test_chinchilla_table_returns_string(self, result):
        table = result.chinchilla_table()
        assert isinstance(table, str)
        assert "Compute Budget" in table
        assert "|" in table

    def test_chinchilla_table_with_custom_budgets(self, result):
        table = result.chinchilla_table([1e18, 1e20])
        assert table.count("|---") == 1  # one separator row = 2 data rows

    def test_predict_returns_dict(self, result):
        pred = result.predict(1e18)
        assert isinstance(pred, dict)
        assert set(pred.keys()) >= {"optimal_params", "optimal_tokens", "expected_loss", "tokens_per_param"}

    def test_predict_returns_numeric_values(self, result):
        pred = result.predict(1e18)
        assert pred["optimal_params"] > 0
        assert pred["optimal_tokens"] > 0
        assert pred["expected_loss"] > 0

    def test_plot_returns_figures(self, result):
        figs = result.plot()
        assert isinstance(figs, list)
        assert len(figs) > 0
        import matplotlib.pyplot as plt
        assert all(isinstance(f, plt.Figure) for f in figs)

    def test_plot_saves_to_output_dir(self, result):
        from pathlib import Path
        result.plot()
        plots_dir = Path(result.output_dir) / "plots"
        assert any(plots_dir.glob("*.png"))

    def test_result_stores_compute_budgets(self, result):
        assert result.compute_budgets == [1e18, 1e19]

    def test_result_stores_output_dir(self, result):
        assert result.output_dir is not None
