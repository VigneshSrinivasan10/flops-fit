"""Public API for flops_fit.

Provides the find_optimal() entry point for computing-optimal model sizing.
"""

from flops_fit.data import validate_dataset
from flops_fit.loss import validate_loss_fn
from flops_fit.model_factory import validate_model_contract
from flops_fit.sweep import plan_sweep


def find_optimal(
    model_cls,
    model_size_param,
    model_kwargs=None,
    dataset=None,
    loss_fn=None,
    compute_budgets=None,
    **kwargs,
):
    """Find compute-optimal model size using scaling law experiments.

    Validates that the provided model class meets the flops_fit contract
    (must accept ``model_size_param`` as a constructor argument and expose
    a ``num_params() -> int`` method), then runs IsoFLOPs sweeps to fit
    scaling laws and predict optimal model sizes for given compute budgets.

    Args:
        model_cls: A model class that accepts ``model_size_param`` as a
            constructor argument and exposes a ``num_params()`` method.
        model_size_param: Name of the constructor parameter that controls
            model size (e.g., ``"d_model"``, ``"hidden_size"``, ``"width"``).
        model_kwargs: Additional keyword arguments passed to model_cls
            constructor (everything except the size parameter). Defaults
            to ``{}``.
        dataset: Training dataset (Phase 2).
        loss_fn: Loss function (Phase 2).
        compute_budgets: List of compute budgets in FLOPs (Phase 3).
        **kwargs: Additional configuration options for future phases.

    Returns:
        SweepPlan: When ``compute_budgets`` is provided, returns an
            inspectable experiment grid (IsoFLOPs sweep plan) that
            the user can review before committing to training.

    Raises:
        TypeError: If model_cls doesn't meet the model contract.
        ValueError: If model parameters are invalid.
        NotImplementedError: When ``compute_budgets`` is not provided
            (full training pipeline not yet implemented).
    """
    if model_kwargs is None:
        model_kwargs = {}

    # Validate model contract up front
    validate_model_contract(model_cls, model_size_param, model_kwargs)

    # Validate dataset interface (Phase 2)
    if dataset is not None:
        validate_dataset(dataset)

    # Validate loss function (Phase 2)
    if loss_fn is not None:
        validate_loss_fn(loss_fn)

    # Phase 3: Generate and return sweep plan
    if compute_budgets is not None:
        plan = plan_sweep(
            model_cls=model_cls,
            size_param=model_size_param,
            model_kwargs=model_kwargs,
            compute_budgets=compute_budgets,
        )
        return plan

    # Full training pipeline not yet implemented
    raise NotImplementedError(
        "find_optimal() model validation passed. "
        "Full pipeline not yet implemented."
    )
