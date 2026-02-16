"""Loss Function Validation

Validates user-provided loss functions before training begins. Checks that
the loss function is callable, accepts at least 2 positional arguments
(predictions and targets), and detects common mistakes like passing a class
instead of an instance.

Gracefully handles C-extension callables where inspect.signature may fail.
"""

import inspect
import logging
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)


def validate_loss_fn(loss_fn: nn.Module | Any) -> None:
    """Validate that loss_fn is a callable accepting (predictions, targets).

    Args:
        loss_fn: The loss function to validate. Can be a callable, nn.Module
            instance, or any object with a __call__ method.

    Raises:
        TypeError: If loss_fn is None, not callable, a class instead of an
            instance, or does not accept at least 2 positional arguments.
    """
    if loss_fn is None:
        raise TypeError(
            "loss_fn is required. Expected a callable like "
            "torch.nn.CrossEntropyLoss(), a lambda, or a custom callable."
        )

    if inspect.isclass(loss_fn) and issubclass(loss_fn, nn.Module):
        class_name = loss_fn.__name__
        raise TypeError(
            f"loss_fn appears to be a class, not an instance. "
            f"Did you mean `loss_fn={class_name}()` (with parentheses)?"
        )

    if not callable(loss_fn):
        type_name = type(loss_fn).__name__
        raise TypeError(
            f"loss_fn must be callable, got {type_name}. "
            f"Expected a callable like torch.nn.CrossEntropyLoss(), "
            f"a lambda, or a custom callable."
        )

    # Inspect signature to check parameter count
    try:
        # For nn.Module instances, inspect forward() instead of __call__
        target = loss_fn.forward if isinstance(loss_fn, nn.Module) else loss_fn
        sig = inspect.signature(target)
    except (ValueError, TypeError):
        # inspect.signature can fail on C extensions, builtins, etc.
        # Gracefully skip the signature check in these cases.
        logger.debug(
            "Could not inspect signature of %s, skipping arity check.",
            _get_name(loss_fn),
        )
        return

    positional_count = 0
    has_var_positional = False

    for param in sig.parameters.values():
        if param.name == "self":
            continue
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_count += 1
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True

    if positional_count < 2 and not has_var_positional:
        name = _get_name(loss_fn)
        raise TypeError(
            f"loss_fn '{name}' must accept at least 2 positional arguments "
            f"(predictions, targets), but accepts {positional_count}."
        )


def _get_name(loss_fn: nn.Module | Any) -> str:
    """Get a human-readable name for a loss function.

    Args:
        loss_fn: The loss function to name.

    Returns:
        The function name or class name.
    """
    if hasattr(loss_fn, "__name__"):
        return loss_fn.__name__
    return type(loss_fn).__name__
