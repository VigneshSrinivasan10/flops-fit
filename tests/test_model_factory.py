"""Tests for flops_fit.model_factory: creation and contract validation."""

import pytest

from flops_fit.model_factory import (
    create_model,
    create_models_at_sizes,
    validate_model_contract,
)


class MockModel:
    """A simple mock model that meets the flops_fit contract."""

    def __init__(self, width=64, num_layers=4):
        self.width = width
        self.num_layers = num_layers

    def num_params(self):
        return self.width * self.num_layers * 100  # fake but deterministic


class ModelWithoutNumParams:
    """A model missing the num_params() method."""

    def __init__(self, width=64):
        self.width = width


class ModelWithBadNumParams:
    """A model where num_params() returns a non-integer."""

    def __init__(self, width=64):
        self.width = width

    def num_params(self):
        return "not_a_number"


class ModelWithNegativeNumParams:
    """A model where num_params() returns a negative integer."""

    def __init__(self, width=64):
        self.width = width

    def num_params(self):
        return -100


# --- create_model tests ---


def test_create_model_basic():
    instance = create_model(MockModel, "width", 128, {"num_layers": 4})
    assert instance.width == 128
    assert instance.num_layers == 4


# --- create_models_at_sizes tests ---


def test_create_models_at_sizes():
    results = create_models_at_sizes(MockModel, "width", [64, 128, 256], {"num_layers": 4})
    assert len(results) == 3

    sizes_seen = []
    params_seen = []
    for size_value, model in results:
        assert isinstance(model, MockModel)
        assert model.width == size_value
        sizes_seen.append(size_value)
        params_seen.append(model.num_params())

    assert sizes_seen == [64, 128, 256]
    # num_params should differ across sizes
    assert len(set(params_seen)) == 3


# --- validate_model_contract tests ---


def test_validate_contract_passes():
    probe = validate_model_contract(MockModel, "width", {"num_layers": 4})
    assert isinstance(probe, MockModel)
    assert probe.num_params() > 0


def test_validate_missing_num_params():
    with pytest.raises(TypeError, match="num_params"):
        validate_model_contract(ModelWithoutNumParams, "width", {})


def test_validate_bad_num_params_return():
    with pytest.raises(ValueError, match="num_params"):
        validate_model_contract(ModelWithBadNumParams, "width", {})


def test_validate_negative_num_params_return():
    with pytest.raises(ValueError, match="num_params"):
        validate_model_contract(ModelWithNegativeNumParams, "width", {})


def test_validate_wrong_size_param():
    with pytest.raises(TypeError, match="nonexistent"):
        validate_model_contract(MockModel, "nonexistent", {})


def test_validate_size_param_in_kwargs_warns():
    with pytest.warns(UserWarning, match="width"):
        probe = validate_model_contract(MockModel, "width", {"width": 32, "num_layers": 4})
    assert isinstance(probe, MockModel)
