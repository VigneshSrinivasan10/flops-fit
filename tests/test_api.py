"""Tests for flops_fit.api: find_optimal() entry point."""

import pytest

from flops_fit import find_optimal


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
