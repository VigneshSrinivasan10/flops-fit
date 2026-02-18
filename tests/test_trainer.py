"""Tests for the training runner module."""

import json

import pytest
import numpy as np

from flops_fit.trainer import TrainingRunner, TrainingResult


# --- Module-level fixtures shared across test classes ---


import torch
import torch.nn as nn
from torch.utils.data import Dataset


class _TinyModel(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.linear = nn.Linear(4, 1)
        self.width = width

    def forward(self, x):
        return self.linear(x)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class _TinyDataset(Dataset):
    def __init__(self, n=64):
        self.x = torch.randn(n, 4)
        self.y = torch.randn(n, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


@pytest.fixture
def tiny_model_cls():
    """A minimal nn.Module that satisfies the flops_fit contract."""
    return _TinyModel


@pytest.fixture
def tiny_dataset():
    """A minimal Dataset returning (input, target) pairs."""
    return _TinyDataset()


@pytest.fixture
def tiny_experiment():
    """A minimal Experiment dataclass for testing."""
    from flops_fit.sweep import Experiment
    return Experiment(
        experiment_id="exp_0000",
        compute_budget=1e10,
        size_param_value=8,
        num_params=5,  # approximate; overridden by actual model
        num_tokens=1000,
        tokens_per_param=200.0,
    )


class TestLocalTraining:
    """Tests for TrainingRunner mode='local' with real PyTorch."""

    def test_local_train_returns_loss_flops_walltime(
        self, tmp_path, tiny_model_cls, tiny_dataset, tiny_experiment
    ):
        """_local_train() returns (loss, actual_flops, wall_time) with sensible values."""
        import torch
        runner = TrainingRunner(mode="local", output_dir=tmp_path)
        loss, actual_flops, wall_time = runner._local_train(
            experiment=tiny_experiment,
            model_cls=tiny_model_cls,
            size_param="width",
            model_kwargs={},
            dataset_or_loader=tiny_dataset,
            loss_fn=torch.nn.MSELoss(),
        )
        assert isinstance(loss, float)
        assert not (loss != loss)  # not NaN
        assert actual_flops > 0
        assert wall_time > 0

    def test_local_train_actual_flops_uses_chinchilla_formula(
        self, tmp_path, tiny_model_cls, tiny_dataset, tiny_experiment
    ):
        """actual_flops returned by _local_train() equals 6 * num_params * num_tokens."""
        import torch
        runner = TrainingRunner(mode="local", output_dir=tmp_path)
        _, actual_flops, _ = runner._local_train(
            experiment=tiny_experiment,
            model_cls=tiny_model_cls,
            size_param="width",
            model_kwargs={},
            dataset_or_loader=tiny_dataset,
            loss_fn=torch.nn.MSELoss(),
        )
        # Formula: C = 6 * N * D; N from actual model, D from experiment.num_tokens
        model = tiny_model_cls(width=tiny_experiment.size_param_value)
        expected_n = model.num_params()
        expected_flops = 6 * expected_n * tiny_experiment.num_tokens
        assert actual_flops == pytest.approx(expected_flops, rel=0.01)

    def test_local_train_uses_device_placement(
        self, tmp_path, tiny_model_cls, tiny_dataset, tiny_experiment
    ):
        """_local_train() runs without errors (device placement happens internally)."""
        import torch
        runner = TrainingRunner(mode="local", output_dir=tmp_path)
        # Should not raise regardless of whether CUDA is available
        loss, _, _ = runner._local_train(
            experiment=tiny_experiment,
            model_cls=tiny_model_cls,
            size_param="width",
            model_kwargs={},
            dataset_or_loader=tiny_dataset,
            loss_fn=torch.nn.MSELoss(),
        )
        assert isinstance(loss, float)

    def test_run_experiment_from_sweep_returns_training_result(
        self, tmp_path, tiny_model_cls, tiny_dataset, tiny_experiment
    ):
        """run_experiment_from_sweep() returns a TrainingResult with status='completed'."""
        import torch
        runner = TrainingRunner(mode="local", output_dir=tmp_path)
        result = runner.run_experiment_from_sweep(
            experiment=tiny_experiment,
            model_cls=tiny_model_cls,
            size_param="width",
            model_kwargs={},
            dataset_or_loader=tiny_dataset,
            loss_fn=torch.nn.MSELoss(),
        )
        assert isinstance(result, TrainingResult)
        assert result.status == "completed"
        assert result.experiment_id == tiny_experiment.experiment_id
        assert result.compute_budget == tiny_experiment.compute_budget
        assert not (result.final_loss != result.final_loss)  # not NaN
        assert result.actual_flops > 0
        assert result.wall_time_seconds > 0

    def test_run_experiment_from_sweep_handles_failure(
        self, tmp_path, tiny_experiment
    ):
        """run_experiment_from_sweep() returns status='failed' when model errors."""
        import torch

        class BrokenModel:
            def __init__(self, width=8):
                pass
            def num_params(self):
                return 5
            def forward(self, x):
                raise RuntimeError("deliberate failure")
            def parameters(self):
                return iter([])
            def train(self):
                pass
            def to(self, device):
                return self

        runner = TrainingRunner(mode="local", output_dir=tmp_path)
        result = runner.run_experiment_from_sweep(
            experiment=tiny_experiment,
            model_cls=BrokenModel,
            size_param="width",
            model_kwargs={},
            dataset_or_loader=torch.utils.data.TensorDataset(
                torch.randn(16, 4), torch.randn(16, 1)
            ),
            loss_fn=torch.nn.MSELoss(),
        )
        assert result.status == "failed"
        assert result.error_message is not None

    def test_resume_sweep_with_experiments_skips_completed(
        self, tmp_path, tiny_model_cls, tiny_dataset
    ):
        """run_sweep_from_plan() skips experiments whose IDs are already in results.json."""
        import json
        import torch
        from flops_fit.sweep import Experiment, SweepPlan

        experiments = [
            Experiment(
                experiment_id=f"exp_{i:04d}",
                compute_budget=1e10,
                size_param_value=8,
                num_params=5,
                num_tokens=100,
                tokens_per_param=20.0,
            )
            for i in range(3)
        ]
        plan = SweepPlan(
            experiments=experiments,
            model_cls_name="TinyModel",
            size_param="width",
            compute_budgets=[1e10],
        )

        # Pre-populate results.json with first experiment completed
        existing = [
            {
                "experiment_id": "exp_0000",
                "compute_budget": 1e10,
                "model_size": 5,
                "num_tokens": 100,
                "final_loss": 1.23,
                "actual_flops": 3000.0,
                "wall_time_seconds": 0.01,
                "timestamp": "2026-01-01T00:00:00",
                "status": "completed",
                "error_message": None,
            }
        ]
        results_path = tmp_path / "results.json"
        with open(results_path, "w") as f:
            json.dump(existing, f)

        runner = TrainingRunner(mode="local", output_dir=tmp_path)
        results = runner.run_sweep_from_plan(
            plan=plan,
            model_cls=tiny_model_cls,
            size_param="width",
            model_kwargs={},
            dataset_or_loader=tiny_dataset,
            loss_fn=torch.nn.MSELoss(),
            resume=True,
        )

        assert len(results) == 3
        # First result is the pre-existing one (not re-run)
        assert results[0]["final_loss"] == pytest.approx(1.23)
        # All results are completed
        assert all(r["status"] == "completed" for r in results)


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


class TestAccelerateIntegration:
    """Tests verifying Accelerate integration in single-process mode."""

    def test_unwrap_model_num_params_correct(
        self, tmp_path, tiny_model_cls, tiny_dataset, tiny_experiment
    ):
        """unwrap_model().num_params() returns correct value after Accelerate prepare()."""
        runner = TrainingRunner(mode="local", output_dir=tmp_path)
        _, actual_flops, _ = runner._local_train(
            experiment=tiny_experiment,
            model_cls=tiny_model_cls,
            size_param="width",
            model_kwargs={},
            dataset_or_loader=tiny_dataset,
            loss_fn=torch.nn.MSELoss(),
        )
        # Verify FLOPs = 6 * N * D, proving unwrap_model().num_params() is correct
        model = tiny_model_cls(width=tiny_experiment.size_param_value)
        expected_n = model.num_params()
        expected_flops = 6 * expected_n * tiny_experiment.num_tokens
        assert actual_flops == pytest.approx(expected_flops, rel=0.01)

    def test_local_train_works_on_cpu_without_cuda(
        self, tmp_path, tiny_model_cls, tiny_dataset, tiny_experiment
    ):
        """_local_train() completes on CPU (Accelerate handles device placement)."""
        runner = TrainingRunner(mode="local", output_dir=tmp_path)
        loss, actual_flops, wall_time = runner._local_train(
            experiment=tiny_experiment,
            model_cls=tiny_model_cls,
            size_param="width",
            model_kwargs={},
            dataset_or_loader=tiny_dataset,
            loss_fn=torch.nn.MSELoss(),
        )
        # Proves Accelerate falls back to CPU gracefully
        assert isinstance(loss, float)
        assert not (loss != loss)  # not NaN
        assert actual_flops > 0
        assert wall_time > 0

    def test_sweep_results_json_written_once(
        self, tmp_path, tiny_model_cls, tiny_dataset
    ):
        """run_sweep_from_plan() writes results.json with no duplicate experiment_ids."""
        from flops_fit.sweep import Experiment, SweepPlan

        experiments = [
            Experiment(
                experiment_id=f"accel_{i:04d}",
                compute_budget=1e10,
                size_param_value=8,
                num_params=5,
                num_tokens=100,
                tokens_per_param=20.0,
            )
            for i in range(2)
        ]
        plan = SweepPlan(
            experiments=experiments,
            model_cls_name="TinyModel",
            size_param="width",
            compute_budgets=[1e10],
        )

        runner = TrainingRunner(mode="local", output_dir=tmp_path)
        results = runner.run_sweep_from_plan(
            plan=plan,
            model_cls=tiny_model_cls,
            size_param="width",
            model_kwargs={},
            dataset_or_loader=tiny_dataset,
            loss_fn=torch.nn.MSELoss(),
            resume=False,
        )

        # Verify results.json written with exactly 2 results
        results_path = tmp_path / "results.json"
        assert results_path.exists()
        with open(results_path) as f:
            saved = json.load(f)
        assert len(saved) == 2
        # No duplicate experiment IDs
        ids = [r["experiment_id"] for r in saved]
        assert len(ids) == len(set(ids))

    def test_accelerate_backward_compatibility(
        self, tmp_path, tiny_model_cls, tiny_dataset, tiny_experiment
    ):
        """Return types and value ranges unchanged after Accelerate integration."""
        runner = TrainingRunner(mode="local", output_dir=tmp_path)
        loss, actual_flops, wall_time = runner._local_train(
            experiment=tiny_experiment,
            model_cls=tiny_model_cls,
            size_param="width",
            model_kwargs={},
            dataset_or_loader=tiny_dataset,
            loss_fn=torch.nn.MSELoss(),
        )
        # Same types as pre-Accelerate
        assert isinstance(loss, float)
        assert isinstance(actual_flops, float)
        assert isinstance(wall_time, float)
        # Same value constraints
        assert not (loss != loss)  # not NaN
        assert loss > 0
        assert actual_flops > 0
        assert actual_flops == 6 * _TinyModel(width=8).num_params() * 1000
        assert wall_time > 0
