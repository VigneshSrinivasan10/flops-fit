"""Sweep planning module for flops_fit.

Generates IsoFLOP experiment grids by probing actual model instances at
different size parameter values. Returns inspectable SweepPlan data structures
that the training engine consumes.

This module is the library-API equivalent of planner.py (which serves the
Hydra CLI). It works with model classes instead of raw parameter counts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from flops_fit.model_factory import create_model

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """A single planned experiment in an IsoFLOP sweep."""

    experiment_id: str
    compute_budget: float  # Target FLOPs (C)
    size_param_value: int  # Value for the model's size parameter
    num_params: int  # Actual parameter count from probe model
    num_tokens: int  # Training tokens (D = C / (flops_per_param_per_token * N))
    tokens_per_param: float  # D / N ratio


@dataclass
class SweepPlan:
    """Complete IsoFLOP sweep plan, inspectable before training."""

    experiments: list[Experiment]
    model_cls_name: str  # For display/logging
    size_param: str  # Name of size parameter
    compute_budgets: list[float]  # Original user-requested budgets
    model_kwargs: dict = field(default_factory=dict)

    @property
    def total_flops(self) -> float:
        """Total estimated FLOPs across all planned experiments."""
        return sum(e.compute_budget for e in self.experiments)

    @property
    def num_experiments(self) -> int:
        return len(self.experiments)

    def __repr__(self) -> str:
        return (
            f"SweepPlan({self.num_experiments} experiments, "
            f"{len(self.compute_budgets)} budgets, "
            f"total_flops={self.total_flops:.2e})"
        )


def _generate_size_values(min_size: int, max_size: int, num_values: int) -> list[int]:
    """Generate log-spaced integer size values from min to max.

    Values are rounded to multiples of 64 for GPU efficiency and
    deduplicated after rounding.

    Args:
        min_size: Minimum size parameter value.
        max_size: Maximum size parameter value.
        num_values: Number of values to generate before deduplication.

    Returns:
        Sorted list of unique integer size values (multiples of 64).
    """
    if min_size >= max_size:
        return [min_size]

    raw = np.logspace(
        np.log10(min_size),
        np.log10(max_size),
        num_values,
    )

    # Round to multiples of 64 for GPU efficiency
    rounded = [max(64, int(round(v / 64)) * 64) for v in raw]

    # Deduplicate while preserving order
    seen = set()
    result = []
    for v in rounded:
        if v not in seen:
            seen.add(v)
            result.append(v)

    return sorted(result)


def _probe_model_sizes(
    model_cls,
    size_param: str,
    model_kwargs: dict,
    size_values: list[int],
) -> list[tuple[int, int]]:
    """Create probe models at each size value to measure actual param counts.

    For each size value, instantiates the model via the model factory, reads
    ``num_params()``, and deletes the probe. Invalid sizes (those that raise
    exceptions during construction) are skipped with a warning.

    Args:
        model_cls: The model class.
        size_param: Name of the size parameter.
        model_kwargs: Other constructor kwargs.
        size_values: List of size parameter values to probe.

    Returns:
        List of (size_value, num_params) tuples, sorted by num_params ascending.
    """
    results = []
    for sv in size_values:
        try:
            model = create_model(model_cls, size_param, sv, model_kwargs)
            n = model.num_params()
            results.append((sv, n))
            del model
        except Exception as exc:
            logger.debug(
                "Skipping %s=%d: %s: %s", size_param, sv, type(exc).__name__, exc
            )
            continue

    return sorted(results, key=lambda x: x[1])


def plan_sweep(
    model_cls,
    size_param: str,
    model_kwargs: dict | None = None,
    compute_budgets: list[float] | None = None,
    *,
    num_sizes_per_budget: int = 7,
    min_size: int = 64,
    max_size: int = 8192,
    flops_per_param_per_token: int = 6,
) -> SweepPlan:
    """Generate an IsoFLOP experiment grid by probing actual model instances.

    Creates probe models at log-spaced size parameter values, measures their
    actual parameter counts, then generates experiment entries for each
    (compute_budget, model_size) pair that passes feasibility filtering.

    Args:
        model_cls: The model class to sweep over.
        size_param: Name of the constructor parameter that controls model size.
        model_kwargs: Other constructor keyword arguments (shared across all sizes).
        compute_budgets: List of compute budgets in FLOPs.
        num_sizes_per_budget: Maximum number of model sizes to try per budget.
        min_size: Minimum value for the size parameter.
        max_size: Maximum value for the size parameter.
        flops_per_param_per_token: FLOPs per parameter per token (default 6 for
            dense transformers: forward 2ND + backward 4ND).

    Returns:
        A SweepPlan containing all feasible Experiment entries.
    """
    if model_kwargs is None:
        model_kwargs = {}
    if compute_budgets is None:
        compute_budgets = []

    # 1. Generate size values via log-spacing
    # Use more probe points than needed to account for dedup and invalid sizes
    num_probe_values = max(num_sizes_per_budget * 3, 20)
    size_values = _generate_size_values(min_size, max_size, num_probe_values)

    # 2. Probe all sizes to get (size_value, num_params) mapping
    size_param_map = _probe_model_sizes(model_cls, size_param, model_kwargs, size_values)

    # 3. For each compute budget, generate feasible experiments
    experiments: list[Experiment] = []
    exp_idx = 0

    for budget in sorted(compute_budgets):
        # Filter to feasible sizes: tokens = C / (flops * N), need tokens >= N/10
        feasible = [
            (sv, n_params)
            for sv, n_params in size_param_map
            if budget / (flops_per_param_per_token * n_params) >= n_params / 10
        ]

        if not feasible:
            logger.warning(
                "No feasible model sizes for compute budget %.2e", budget
            )
            continue

        # 4. If more feasible sizes than requested, select evenly-spaced subset
        if len(feasible) > num_sizes_per_budget:
            indices = np.linspace(0, len(feasible) - 1, num_sizes_per_budget, dtype=int)
            feasible = [feasible[i] for i in indices]

        # 5. Create Experiment entries
        for size_value, n_params in feasible:
            num_tokens = int(budget / (flops_per_param_per_token * n_params))
            tokens_per_param = num_tokens / n_params
            experiments.append(
                Experiment(
                    experiment_id=f"exp_{exp_idx:04d}",
                    compute_budget=budget,
                    size_param_value=size_value,
                    num_params=n_params,
                    num_tokens=num_tokens,
                    tokens_per_param=tokens_per_param,
                )
            )
            exp_idx += 1

    # 6. Return SweepPlan
    return SweepPlan(
        experiments=experiments,
        model_cls_name=model_cls.__name__,
        size_param=size_param,
        compute_budgets=list(compute_budgets),
        model_kwargs=model_kwargs,
    )
