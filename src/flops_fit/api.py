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
    train: bool = True,
    output_dir: str = "outputs",
    resume: bool = True,
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
        dataset: Training dataset (torch Dataset or DataLoader). Required for
            training execution.
        loss_fn: Loss function callable (outputs, targets) -> scalar tensor.
            Required for training execution.
        compute_budgets: List of compute budgets in FLOPs (Phase 3).
        train: If True (default) and ``dataset`` + ``loss_fn`` are both
            provided, executes training and returns results. If False, returns
            the SweepPlan for inspection without running training.
        output_dir: Directory to write ``results.json`` and experiment
            artifacts. Only used when ``train=True``. Defaults to
            ``"outputs"``.
        resume: If True (default), skip experiments already recorded as
            completed in ``{output_dir}/results.json`` from a prior run.
        **kwargs: Additional configuration options for future phases.

    Returns:
        list[dict]: When ``train=True`` and both ``dataset`` and ``loss_fn``
            are provided, returns a list of result dicts (one per experiment)
            with keys: ``experiment_id``, ``final_loss``, ``actual_flops``,
            ``wall_time_seconds``, ``status``, and metadata.
        SweepPlan: When ``train=False``, or when ``dataset`` / ``loss_fn``
            is omitted, returns an inspectable experiment grid that the user
            can review before committing to training.

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

    # Phase 3 + 4: Generate sweep plan and optionally execute training
    if compute_budgets is not None:
        plan = plan_sweep(
            model_cls=model_cls,
            size_param=model_size_param,
            model_kwargs=model_kwargs,
            compute_budgets=compute_budgets,
        )

        # Phase 4: execute training if dataset + loss_fn both provided and train=True
        if train and dataset is not None and loss_fn is not None:
            from flops_fit.trainer import TrainingRunner
            runner = TrainingRunner(mode="local", output_dir=output_dir)
            return runner.run_sweep_from_plan(
                plan=plan,
                model_cls=model_cls,
                size_param=model_size_param,
                model_kwargs=model_kwargs,
                dataset_or_loader=dataset,
                loss_fn=loss_fn,
                resume=resume,
            )

        # Inspection mode: just return the plan
        return plan

    # Full training pipeline not yet implemented
    raise NotImplementedError(
        "find_optimal() model validation passed. "
        "Full pipeline not yet implemented."
    )
