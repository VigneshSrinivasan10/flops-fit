"""Tests for the training runner module."""

import json

import pytest
import numpy as np

from flops_fit.trainer import TrainingRunner, TrainingResult


class TestTrainingRunner:
    """Test suite for TrainingRunner."""

    def test_load_sweep(self, sweep_json):
        """load_sweep returns list of experiment config dicts."""
        runner = TrainingRunner(
            mode="mock",
            sweep_path=sweep_json,
            output_dir=sweep_json.parent,
        )
        configs = runner.load_sweep()

        assert isinstance(configs, list)
        assert len(configs) == 12
        required_keys = {"experiment_id", "compute_budget", "model_size", "num_tokens"}
        for cfg in configs:
            assert required_keys.issubset(cfg.keys())

    def test_load_sweep_missing_file(self, tmp_path):
        """load_sweep raises FileNotFoundError with message mentioning 'sl-plan'."""
        runner = TrainingRunner(
            mode="mock",
            sweep_path=tmp_path / "nonexistent.json",
            output_dir=tmp_path,
        )
        # Characterization: error message references old command name 'sl-plan'
        with pytest.raises(FileNotFoundError, match="sl-plan"):
            runner.load_sweep()

    def test_mock_train_produces_reasonable_loss(self, tmp_path):
        """_mock_train returns loss in a reasonable range with positive flops and time."""
        # Characterization: _mock_train uses unseeded np.random internally;
        # we seed externally for determinism.
        np.random.seed(42)
        runner = TrainingRunner(mode="mock", sweep_path=tmp_path / "s.json", output_dir=tmp_path)
        loss, actual_flops, wall_time = runner._mock_train(
            model_size=10_000_000,
            num_tokens=100_000_000,
            compute_budget=6e15,
        )

        assert 1.5 < loss < 15.0
        assert actual_flops > 0
        assert wall_time > 0

    def test_mock_train_loss_scaling(self, tmp_path):
        """Larger models with more tokens achieve lower loss (scaling law behavior)."""
        runner = TrainingRunner(mode="mock", sweep_path=tmp_path / "s.json", output_dir=tmp_path)

        np.random.seed(42)
        loss_small, _, _ = runner._mock_train(
            model_size=1_000_000, num_tokens=10_000_000, compute_budget=6e13
        )

        np.random.seed(42)
        loss_large, _, _ = runner._mock_train(
            model_size=100_000_000, num_tokens=1_000_000_000, compute_budget=6e17
        )

        assert loss_large < loss_small

    def test_run_sweep_completes_all_experiments(self, tmp_path, sweep_json):
        """run_sweep in mock mode completes all 12 experiments and writes results.json."""
        np.random.seed(42)
        runner = TrainingRunner(
            mode="mock",
            sweep_path=sweep_json,
            output_dir=tmp_path,
        )
        results = runner.run_sweep(resume=False)

        assert len(results) == 12
        assert all(r["status"] == "completed" for r in results)
        assert (tmp_path / "results.json").exists()

    def test_run_sweep_resume_skips_completed(self, tmp_path, sweep_json, sample_results):
        """resume=True preserves existing results and only runs remaining experiments."""
        # Pre-populate results.json with first 3 experiments completed
        existing = sample_results[:3]
        results_path = tmp_path / "results.json"
        with open(results_path, "w") as f:
            json.dump(existing, f)

        np.random.seed(42)
        runner = TrainingRunner(
            mode="mock",
            sweep_path=sweep_json,
            output_dir=tmp_path,
        )
        results = runner.run_sweep(resume=True)

        assert len(results) == 12
        # First 3 results should have original loss values (not overwritten)
        for i in range(3):
            assert results[i]["final_loss"] == pytest.approx(existing[i]["final_loss"])

    def test_run_sweep_resume_false_reruns_all(self, tmp_path, sweep_json, sample_results):
        """resume=False ignores existing results and re-runs everything."""
        # Pre-populate results.json with 3 completed results
        existing = sample_results[:3]
        results_path = tmp_path / "results.json"
        with open(results_path, "w") as f:
            json.dump(existing, f)

        np.random.seed(42)
        runner = TrainingRunner(
            mode="mock",
            sweep_path=sweep_json,
            output_dir=tmp_path,
        )
        results = runner.run_sweep(resume=False)

        # Should rerun all 12, not load existing
        assert len(results) == 12

    def test_run_experiment_handles_failure(self, tmp_path):
        """run_experiment returns failed result when _train_fn is not set."""
        runner = TrainingRunner(
            mode="nonexistent",
            sweep_path=tmp_path / "s.json",
            output_dir=tmp_path,
        )
        config = {
            "experiment_id": "exp_0000",
            "compute_budget": 1e18,
            "model_size": 1_000_000,
            "num_tokens": 10_000_000,
        }
        result = runner.run_experiment(config)

        assert result.status == "failed"
        assert result.error_message is not None


class TestTrainingResult:
    """Test suite for TrainingResult."""

    def test_to_dict(self):
        """to_dict returns all 10 expected keys."""
        result = TrainingResult(
            experiment_id="exp_0001",
            compute_budget=1e18,
            model_size=100_000_000,
            num_tokens=1_000_000_000,
            final_loss=3.5,
            actual_flops=1.01e18,
            wall_time_seconds=120.0,
        )
        d = result.to_dict()

        expected_keys = {
            "experiment_id", "compute_budget", "model_size", "num_tokens",
            "final_loss", "actual_flops", "wall_time_seconds",
            "timestamp", "status", "error_message",
        }
        assert expected_keys == set(d.keys())
