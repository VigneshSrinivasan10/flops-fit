"""Tests for loss function validation."""

from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from flops_fit.loss import _get_name, validate_loss_fn


class CustomLoss(nn.Module):
    """A custom loss module with forward(self, pred, target)."""

    def forward(self, pred, target):
        return (pred - target).pow(2).mean()


def my_loss_with_kwargs(pred, target, weight=1.0):
    """Loss function with extra keyword argument."""
    return weight * (pred - target).mean()


def my_loss_varargs(*args):
    """Loss function with *args."""
    return args[0].mean()


class TestValidateLossFn:
    """Test suite for validate_loss_fn."""

    def test_accepts_plain_function(self):
        """validate_loss_fn accepts a lambda with 2 args."""
        validate_loss_fn(lambda pred, target: (pred - target).mean())

    def test_accepts_nn_module_instance(self):
        """validate_loss_fn accepts an nn.Module instance like CrossEntropyLoss()."""
        validate_loss_fn(nn.CrossEntropyLoss())

    def test_accepts_custom_nn_module(self):
        """validate_loss_fn accepts a custom nn.Module with forward(self, pred, target)."""
        validate_loss_fn(CustomLoss())

    def test_accepts_function_with_extra_kwargs(self):
        """validate_loss_fn accepts a function with 2+ positional args plus kwargs."""
        validate_loss_fn(my_loss_with_kwargs)

    def test_accepts_var_args(self):
        """validate_loss_fn accepts a function with *args."""
        validate_loss_fn(my_loss_varargs)

    def test_rejects_none(self):
        """validate_loss_fn rejects None with TypeError mentioning 'loss_fn is required'."""
        with pytest.raises(TypeError, match="loss_fn is required"):
            validate_loss_fn(None)

    def test_rejects_non_callable(self):
        """validate_loss_fn rejects non-callable with TypeError mentioning 'callable'."""
        with pytest.raises(TypeError, match="callable"):
            validate_loss_fn(42)

    def test_rejects_zero_arg_callable(self):
        """validate_loss_fn rejects a zero-arg callable."""
        with pytest.raises(TypeError, match="at least 2 positional"):
            validate_loss_fn(lambda: 0)

    def test_rejects_one_arg_callable(self):
        """validate_loss_fn rejects a one-arg callable."""
        with pytest.raises(TypeError):
            validate_loss_fn(lambda x: x)

    def test_warns_class_not_instance(self):
        """validate_loss_fn rejects nn.Module class (not instance) with helpful message."""
        with pytest.raises(TypeError, match="Did you mean") as exc_info:
            validate_loss_fn(nn.CrossEntropyLoss)
        assert "with parentheses" in str(exc_info.value).lower() or "parentheses" in str(
            exc_info.value
        )

    def test_handles_uninspectable_callable(self):
        """validate_loss_fn gracefully handles callables where inspect.signature fails."""
        mock = MagicMock()
        # MagicMock is callable but inspect.signature raises ValueError
        validate_loss_fn(mock)  # Should NOT raise


class TestGetName:
    """Test suite for _get_name helper."""

    def test_function_name(self):
        """_get_name returns __name__ for a lambda."""
        assert _get_name(lambda: 0) == "<lambda>"

    def test_module_name(self):
        """_get_name returns class name for nn.Module instance."""
        assert _get_name(nn.CrossEntropyLoss()) == "CrossEntropyLoss"
