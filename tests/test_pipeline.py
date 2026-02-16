"""End-to-end pipeline integration tests and Hydra config tests."""

import matplotlib
matplotlib.use("Agg")

import json

import pytest
import numpy as np
import matplotlib.pyplot as plt

from flops_fit.planner import SweepPlanner
from flops_fit.trainer import TrainingRunner
from flops_fit.analyzer import ScalingLawAnalyzer
from flops_fit.visualizer import ScalingVisualizer


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Reset matplotlib state after each test."""
    yield
    plt.close("all")
    plt.rcdefaults()


class TestFullPipeline:
    """End-to-end pipeline integration tests."""

    def test_full_pipeline_mock_mode(self, tmp_path):
        """Full plan -> train -> analyze -> visualize pipeline in mock mode."""
        # Step 1: Plan
        planner = SweepPlanner(
            min_flops=1e17,
            max_flops=1e21,
            num_compute_budgets=5,
            num_model_sizes=7,
        )
        sweep_path = tmp_path / "sweep.json"
        configs = planner.save_sweep(sweep_path)
        assert len(configs) > 0
        assert sweep_path.exists()

        # Step 2: Train
        np.random.seed(42)
        runner = TrainingRunner(
            mode="mock",
            sweep_path=sweep_path,
            output_dir=tmp_path,
        )
        results = runner.run_sweep(resume=False)
        assert all(r["status"] == "completed" for r in results)
        assert (tmp_path / "results.json").exists()

        # Step 3: Analyze
        analyzer = ScalingLawAnalyzer(
            results_path=tmp_path / "results.json",
            output_dir=tmp_path / "analysis",
        )
        analysis = analyzer.analyze()
        assert analysis.n_opt_fit.r_squared > 0
        assert analysis.d_opt_fit.r_squared > 0
        assert analysis.l_opt_fit.r_squared > 0
        assert (tmp_path / "analysis" / "scaling_laws.json").exists()

        # Step 4: Visualize
        viz = ScalingVisualizer(
            results_path=tmp_path / "results.json",
            analysis_path=tmp_path / "analysis" / "scaling_laws.json",
            output_dir=tmp_path / "plots",
        )
        figures = viz.plot_all(save=True)
        assert len(figures) == 3
        assert (tmp_path / "plots" / "isoflops_curves.png").exists()
        assert (tmp_path / "plots" / "scaling_laws.png").exists()
        assert (tmp_path / "plots" / "tokens_per_param.png").exists()

    def test_pipeline_resume_from_partial(self, tmp_path):
        """Resume training completes the full sweep without duplicates."""
        planner = SweepPlanner(
            min_flops=1e17,
            max_flops=1e19,
            num_compute_budgets=3,
            num_model_sizes=4,
        )
        sweep_path = tmp_path / "sweep.json"
        planner.save_sweep(sweep_path)

        # First run: complete all
        np.random.seed(42)
        runner = TrainingRunner(mode="mock", sweep_path=sweep_path, output_dir=tmp_path)
        first_results = runner.run_sweep(resume=False)
        first_count = len(first_results)

        # Second run with resume: should not add duplicate experiments
        np.random.seed(42)
        runner2 = TrainingRunner(mode="mock", sweep_path=sweep_path, output_dir=tmp_path)
        resumed_results = runner2.run_sweep(resume=True)

        assert len(resumed_results) == first_count


class TestHydraConfigs:
    """Smoke tests for Hydra config composition."""

    CONF_DIR = str(
        (pytest.importorskip("flops_fit").__spec__.submodule_search_locations[0]
         if hasattr(pytest.importorskip("flops_fit").__spec__, "submodule_search_locations")
         else __import__("pathlib").Path(pytest.importorskip("flops_fit").__file__).parent)
        / __import__("pathlib").Path("conf")
    )

    @pytest.fixture(autouse=True)
    def _conf_dir(self):
        """Resolve config directory once."""
        from pathlib import Path
        import flops_fit
        self.conf_dir = str(Path(flops_fit.__file__).parent / "conf")

    def test_planner_config_loads(self):
        """Planner config loads with expected defaults."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=self.conf_dir, version_base=None):
            cfg = compose(config_name="planner")
            assert cfg.compute.min_flops > 0
            assert cfg.compute.num_budgets > 0
            assert cfg.output.sweep_path == "outputs/sweep.json"

    def test_planner_config_overrides(self):
        """Planner config accepts overrides."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=self.conf_dir, version_base=None):
            cfg = compose(config_name="planner", overrides=["compute.min_flops=1e15", "compute.num_budgets=3"])
            assert cfg.compute.min_flops == 1e15
            assert cfg.compute.num_budgets == 3

    def test_trainer_config_loads(self):
        """Trainer config loads with expected defaults."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=self.conf_dir, version_base=None):
            cfg = compose(config_name="trainer")
            assert cfg.mode == "mock"
            assert cfg.paths.sweep == "outputs/sweep.json"
            assert cfg.resume == True

    def test_analyzer_config_loads(self):
        """Analyzer config loads with expected defaults."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=self.conf_dir, version_base=None):
            cfg = compose(config_name="analyzer")
            assert cfg.paths.results == "outputs/results.json"

    def test_visualizer_config_loads(self):
        """Visualizer config loads with expected defaults."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=self.conf_dir, version_base=None):
            cfg = compose(config_name="visualizer")
            assert cfg.plots.style == "paper"
            assert cfg.plots.show == False

    def test_all_configs_disable_hydra_output_dir(self):
        """All configs set hydra.run.dir='.' and hydra.output_subdir=null.

        This characterizes the important behavior that Hydra doesn't change
        the working directory or create output subdirectories.
        Reads YAML directly since hydra section isn't accessible via compose().
        """
        from pathlib import Path
        import yaml

        for config_name in ["planner", "trainer", "analyzer", "visualizer"]:
            config_path = Path(self.conf_dir) / f"{config_name}.yaml"
            with open(config_path) as f:
                raw = yaml.safe_load(f)
            assert raw["hydra"]["run"]["dir"] == ".", f"{config_name}: hydra.run.dir != '.'"
            assert raw["hydra"]["output_subdir"] is None, f"{config_name}: hydra.output_subdir not null"
