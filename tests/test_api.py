"""Tests for flops_fit.api: find_optimal() entry point."""

import pytest
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset

from flops_fit import find_optimal
from flops_fit.sweep import SweepPlan


class MockModel:
    """A simple mock model that meets the flops_fit contract."""

    def __init__(self, width=64, num_layers=4):
        self.width = width
        self.num_layers = num_layers

    def num_params(self):
        return self.width * self.num_layers * 100


class BadModel:
    """A model missing num_params()."""

    def __init__(self, width=64):
        self.width = width


def test_find_optimal_importable():
    from flops_fit import find_optimal as fo

    assert callable(fo)


def test_find_optimal_validates_then_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="model validation passed"):
        find_optimal(MockModel, "width", model_kwargs={"num_layers": 4})


def test_find_optimal_rejects_bad_model():
    with pytest.raises(TypeError, match="num_params"):
        find_optimal(BadModel, "width")


def test_find_optimal_default_kwargs():
    """Calling with model_kwargs=None (the default) should still work."""
    with pytest.raises(NotImplementedError, match="model validation passed"):
        find_optimal(MockModel, "width")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class TinyDataset(torch.utils.data.Dataset):
    """Minimal torch Dataset for testing."""

    def __init__(self):
        self.data = torch.randn(10, 4)

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return self.data[idx]


# ---------------------------------------------------------------------------
# Dataset validation through find_optimal()
# ---------------------------------------------------------------------------


class TestFindOptimalDatasetValidation:
    """Integration tests: dataset validation via find_optimal()."""

    def test_accepts_valid_dataset(self):
        """Valid Dataset + valid loss_fn -> passes validation, hits NotImplementedError."""
        with pytest.raises(NotImplementedError, match="model validation passed"):
            find_optimal(
                MockModel, "width", dataset=TinyDataset(), loss_fn=torch.nn.MSELoss()
            )

    def test_accepts_none_dataset(self):
        """dataset=None preserves backward compat (Phase 1 behavior)."""
        with pytest.raises(NotImplementedError, match="model validation passed"):
            find_optimal(MockModel, "width", dataset=None, loss_fn=None)

    def test_rejects_invalid_dataset(self):
        """Non-Dataset object raises TypeError, not NotImplementedError."""
        with pytest.raises(TypeError, match="Expected a torch.utils.data.Dataset"):
            find_optimal(MockModel, "width", dataset=[1, 2, 3])

    def test_model_validated_before_dataset(self):
        """Bad model + bad dataset -> error about MODEL (validated first)."""
        with pytest.raises(TypeError, match="num_params"):
            find_optimal(BadModel, "width", dataset=[1, 2, 3])


# ---------------------------------------------------------------------------
# Loss validation through find_optimal()
# ---------------------------------------------------------------------------


class TestFindOptimalLossValidation:
    """Integration tests: loss_fn validation via find_optimal()."""

    def test_accepts_valid_loss(self):
        """Valid loss_fn -> passes validation, hits NotImplementedError."""
        with pytest.raises(NotImplementedError, match="model validation passed"):
            find_optimal(
                MockModel, "width", dataset=TinyDataset(), loss_fn=torch.nn.MSELoss()
            )

    def test_accepts_none_loss(self):
        """loss_fn=None preserves backward compat."""
        with pytest.raises(NotImplementedError, match="model validation passed"):
            find_optimal(MockModel, "width", dataset=TinyDataset(), loss_fn=None)

    def test_rejects_non_callable_loss(self):
        """Non-callable loss_fn raises TypeError."""
        with pytest.raises(TypeError, match="loss_fn must be callable"):
            find_optimal(MockModel, "width", loss_fn=42)

    def test_dataset_validated_before_loss(self):
        """Bad dataset + bad loss_fn -> error about DATASET (validated before loss)."""
        with pytest.raises(TypeError, match="Expected a torch.utils.data.Dataset"):
            find_optimal(MockModel, "width", dataset="not_a_dataset", loss_fn=42)


# ---------------------------------------------------------------------------
# Sweep planning through find_optimal()
# ---------------------------------------------------------------------------


class TestFindOptimalSweepPlanning:
    """Integration tests: sweep planning via find_optimal()."""

    def test_returns_sweep_plan(self):
        """compute_budgets triggers sweep planning, returns SweepPlan (not exception)."""
        result = find_optimal(
            MockModel,
            "width",
            model_kwargs={"num_layers": 4},
            compute_budgets=[1e15, 1e16],
        )
        assert isinstance(result, SweepPlan)
        assert result.num_experiments > 0
        assert result.compute_budgets == [1e15, 1e16]

    def test_no_compute_budgets_raises_not_implemented(self):
        """Without compute_budgets, still raises NotImplementedError (backward compat)."""
        with pytest.raises(NotImplementedError, match="model validation passed"):
            find_optimal(MockModel, "width")

    def test_compute_budgets_none_raises_not_implemented(self):
        """Explicit compute_budgets=None behaves same as omitting it."""
        with pytest.raises(NotImplementedError, match="model validation passed"):
            find_optimal(MockModel, "width", compute_budgets=None)

    def test_model_validated_before_sweep(self):
        """Bad model + compute_budgets -> error about MODEL (validated first)."""
        with pytest.raises(TypeError, match="num_params"):
            find_optimal(BadModel, "width", compute_budgets=[1e15])

    def test_dataset_validated_before_sweep(self):
        """Bad dataset + compute_budgets -> error about DATASET (validated before sweep)."""
        with pytest.raises(TypeError, match="Expected a torch.utils.data.Dataset"):
            find_optimal(
                MockModel, "width", dataset="not_a_dataset", compute_budgets=[1e15]
            )

    def test_loss_validated_before_sweep(self):
        """Bad loss_fn + compute_budgets -> error about LOSS (validated before sweep)."""
        with pytest.raises(TypeError, match="loss_fn must be callable"):
            find_optimal(
                MockModel,
                "width",
                dataset=TinyDataset(),
                loss_fn=42,
                compute_budgets=[1e15],
            )


# ---------------------------------------------------------------------------
# Training execution through find_optimal()
# ---------------------------------------------------------------------------


class TestFindOptimalTraining:
    """Integration tests for find_optimal() training execution path."""

    @pytest.fixture
    def tiny_model_cls(self):
        class TinyLinear(nn.Module):
            def __init__(self, width=8):
                super().__init__()
                self.net = nn.Linear(4, 1)
                self.width = width

            def forward(self, x):
                return self.net(x)

            def num_params(self):
                return sum(p.numel() for p in self.parameters())

        return TinyLinear

    @pytest.fixture
    def tiny_dataset(self):
        class DS(Dataset):
            def __init__(self):
                self.x = torch.randn(64, 4)
                self.y = torch.randn(64, 1)

            def __len__(self):
                return 64

            def __getitem__(self, i):
                return self.x[i], self.y[i]

        return DS()

    def test_find_optimal_executes_training_returns_results(
        self, tmp_path, tiny_model_cls, tiny_dataset
    ):
        """find_optimal() returns list of result dicts when dataset+loss_fn provided."""
        results = find_optimal(
            model_cls=tiny_model_cls,
            model_size_param="width",
            dataset=tiny_dataset,
            loss_fn=nn.MSELoss(),
            compute_budgets=[1e8],
            train=True,
            output_dir=str(tmp_path),
        )
        assert isinstance(results, list)
        assert len(results) > 0
        required_keys = {"experiment_id", "final_loss", "actual_flops", "wall_time_seconds", "status"}
        for r in results:
            assert required_keys.issubset(r.keys())
            assert r["status"] == "completed"
            assert not (r["final_loss"] != r["final_loss"])  # not NaN

    def test_find_optimal_train_false_returns_sweep_plan(
        self, tiny_model_cls, tiny_dataset
    ):
        """find_optimal(train=False) returns SweepPlan even when dataset provided."""
        plan = find_optimal(
            model_cls=tiny_model_cls,
            model_size_param="width",
            dataset=tiny_dataset,
            loss_fn=nn.MSELoss(),
            compute_budgets=[1e8],
            train=False,
        )
        assert isinstance(plan, SweepPlan)

    def test_find_optimal_no_dataset_returns_sweep_plan(self, tiny_model_cls):
        """find_optimal() without dataset still returns SweepPlan (existing behavior)."""
        plan = find_optimal(
            model_cls=tiny_model_cls,
            model_size_param="width",
            compute_budgets=[1e8],
        )
        assert isinstance(plan, SweepPlan)

    def test_find_optimal_writes_results_json(
        self, tmp_path, tiny_model_cls, tiny_dataset
    ):
        """find_optimal() with output_dir writes results.json after training."""
        find_optimal(
            model_cls=tiny_model_cls,
            model_size_param="width",
            dataset=tiny_dataset,
            loss_fn=nn.MSELoss(),
            compute_budgets=[1e8],
            train=True,
            output_dir=str(tmp_path),
        )
        results_file = tmp_path / "results.json"
        assert results_file.exists()
        import json
        data = json.loads(results_file.read_text())
        assert isinstance(data, list)
        assert len(data) > 0

    def test_find_optimal_resume_skips_completed(
        self, tmp_path, tiny_model_cls, tiny_dataset
    ):
        """find_optimal() with resume=True skips experiments already in results.json."""
        import json
        # Run once to get results
        results1 = find_optimal(
            model_cls=tiny_model_cls,
            model_size_param="width",
            dataset=tiny_dataset,
            loss_fn=nn.MSELoss(),
            compute_budgets=[1e8],
            train=True,
            output_dir=str(tmp_path),
        )
        first_loss = results1[0]["final_loss"]

        # Run again with resume=True — completed experiments must not be re-run
        results2 = find_optimal(
            model_cls=tiny_model_cls,
            model_size_param="width",
            dataset=tiny_dataset,
            loss_fn=nn.MSELoss(),
            compute_budgets=[1e8],
            train=True,
            output_dir=str(tmp_path),
            resume=True,
        )
        # First result must have the same loss (not re-computed)
        assert results2[0]["final_loss"] == pytest.approx(first_loss)
