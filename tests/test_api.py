"""Tests for flops_fit.api: find_optimal() entry point."""

import pytest
import torch
import torch.utils.data

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
