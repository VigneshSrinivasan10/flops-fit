"""Model factory: instantiation and contract validation for flops_fit.

Provides functions to create model instances at different sizes and validate
that a model class meets the flops_fit contract (must expose num_params() -> int).
"""

import warnings


def create_model(model_cls, size_param, size_value, model_kwargs):
    """Create a single model instance at a specific size.

    Args:
        model_cls: The model class to instantiate.
        size_param: Name of the constructor parameter that controls model size.
        size_value: Value for the size parameter.
        model_kwargs: Other constructor keyword arguments.

    Returns:
        An instance of model_cls.
    """
    kwargs = {**model_kwargs, size_param: size_value}
    return model_cls(**kwargs)


def create_models_at_sizes(model_cls, size_param, size_values, model_kwargs):
    """Create model instances at multiple sizes.

    Args:
        model_cls: The model class to instantiate.
        size_param: Name of the constructor parameter that controls model size.
        size_values: List of values for the size parameter.
        model_kwargs: Other constructor keyword arguments.

    Returns:
        List of (size_value, model_instance) tuples.
    """
    models = []
    for size_value in size_values:
        model = create_model(model_cls, size_param, size_value, model_kwargs)
        models.append((size_value, model))
    return models


def validate_model_contract(model_cls, size_param, model_kwargs):
    """Validate that a model class meets the flops_fit contract.

    Creates a probe instance and checks:
    1. model_cls is callable (can be instantiated)
    2. size_param is accepted as a constructor argument
    3. Instance has num_params() method
    4. num_params() returns a positive integer

    If size_param is present in model_kwargs, a warning is issued and
    the key is removed from kwargs before proceeding.

    Args:
        model_cls: The model class to validate.
        size_param: Name of the constructor parameter that controls model size.
        model_kwargs: Other constructor keyword arguments.

    Returns:
        The probe model instance.

    Raises:
        TypeError: If model_cls cannot be instantiated, size_param is not
            accepted, or num_params() is missing.
        ValueError: If num_params() returns a non-positive value.
    """
    # Handle size_param appearing in model_kwargs
    if size_param in model_kwargs:
        warnings.warn(
            f"'{size_param}' found in model_kwargs and will be overridden "
            f"by the library during sweeps. Removing it from model_kwargs.",
            UserWarning,
            stacklevel=2,
        )
        model_kwargs = {k: v for k, v in model_kwargs.items() if k != size_param}

    # Pick a small probe value
    probe_value = 64

    try:
        probe = create_model(model_cls, size_param, probe_value, model_kwargs)
    except TypeError as e:
        raise TypeError(
            f"Cannot create {model_cls.__name__} with "
            f"{size_param}={probe_value} and kwargs={model_kwargs}. "
            f"Error: {e}"
        ) from e

    if not hasattr(probe, "num_params"):
        raise TypeError(
            f"{model_cls.__name__} does not have a num_params() method. "
            f"flops_fit requires model classes to expose num_params() -> int "
            f"so the library can measure model size at different scales. "
            f"Did you mean count_parameters()?"
        )

    try:
        n = probe.num_params()
    except Exception as e:
        raise TypeError(
            f"{model_cls.__name__}.num_params() raised an error: {e}. "
            f"num_params() must return a positive integer."
        ) from e

    if not isinstance(n, int) or n <= 0:
        raise ValueError(
            f"{model_cls.__name__}.num_params() returned {n!r}. "
            f"Expected a positive integer."
        )

    return probe
