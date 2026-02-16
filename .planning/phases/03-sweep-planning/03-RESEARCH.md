# Phase 3: Sweep Planning - Research

**Researched:** 2026-02-16
**Domain:** IsoFLOP experiment grid generation, compute budgeting, model-factory-to-planner bridge
**Confidence:** HIGH

## Summary

Phase 3 bridges the existing `SweepPlanner` (which generates experiment grids from raw parameter counts and compute budgets) with the new library API where users pass a model class, size parameter, and kwargs. The core challenge is: given a compute budget and a model class, how does the library decide what size parameter values to try, and how does it estimate FLOPs for each experiment?

The existing `SweepPlanner` already implements the IsoFLOP methodology correctly -- it generates logarithmically spaced compute budgets, creates model sizes per budget, calculates token counts via `D = C / (6N)`, and filters infeasible configurations. What it lacks is integration with the model factory: it generates raw parameter counts (e.g., 10M, 100M, 1B) but doesn't know how to translate those into model-class-specific size parameter values (e.g., `d_model=256`).

Phase 3 needs to: (1) accept `compute_budgets` in `find_optimal()` and generate an experiment grid as a data structure, (2) use the model factory to create probe models at different sizes to get actual parameter counts (since the `12*L*d^2` approximation overshoots by ~7.5x for small models), and (3) provide a `total_flops()` cost estimation method so users can preview compute cost before committing GPU hours. The sweep plan must be inspectable as a Python data structure, not just printed output.

No new external libraries are needed. This phase uses only the existing model factory, numpy for log-spacing, and dataclasses for the plan data structure.

**Primary recommendation:** Create a new `SweepPlan` dataclass (list of `Experiment` entries) returned by a `plan_sweep()` function that takes model class, size param, kwargs, and compute budgets. The function creates probe models at multiple size values to build a mapping from size parameter values to actual parameter counts, then generates the IsoFLOP grid. Wire `plan_sweep()` into `find_optimal()` so the sweep plan is created before training begins. Expose `total_flops` as a property on the `SweepPlan` object.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python dataclasses | stdlib | `SweepPlan`, `Experiment` data structures | Already used for `ExperimentConfig`, `TrainingResult`; consistent pattern |
| numpy | >=1.26.0 | Logarithmic spacing of compute budgets and size values | Already a dependency; used in existing `SweepPlanner` |
| model_factory.py | (internal) | `create_model()` to probe actual parameter counts | Already implemented in Phase 1 |

### Supporting

No additional libraries needed. This phase uses existing project infrastructure.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Probing actual models for param counts | Analytical formula (12*L*d^2) | Formula overshoots by ~7.5x at small scales due to SwiGLU + d_model rounding (documented prior decision). Probing is slower but accurate. For typical sweep sizes (20-50 experiments), probing adds seconds, not minutes. |
| New `SweepPlan` dataclass | Reuse existing `ExperimentConfig` list | `ExperimentConfig` lacks size_param_value and model-class context. A new dataclass wrapping a list of experiments provides `total_flops` and other summary methods. |
| Generating size values by binary search for target param counts | Linearly spacing size_param values and measuring what you get | Binary search is more precise but complex. Log-spacing size values and recording actual params is simpler and sufficient -- the IsoFLOP method doesn't require exact parameter targets, just a spread across the compute budget. |

**Installation:**
```bash
# No new dependencies
```

## Architecture Patterns

### Recommended Project Structure Changes

```
src/flops_fit/
    __init__.py          # MODIFY: export SweepPlan
    api.py               # MODIFY: wire plan_sweep() into find_optimal()
    sweep.py             # NEW: SweepPlan, Experiment, plan_sweep()
    model_factory.py     # UNCHANGED (from Phase 1)
    data.py              # UNCHANGED (from Phase 2)
    loss.py              # UNCHANGED (from Phase 2)
    planner.py           # UNCHANGED (existing CLI planner stays as-is)
    model.py             # UNCHANGED
    trainer.py           # UNCHANGED
    analyzer.py          # UNCHANGED
    visualizer.py        # UNCHANGED
```

**Rationale:** A new `sweep.py` module keeps the library-API sweep planning separate from the existing CLI-oriented `planner.py`. The existing `SweepPlanner` continues to work for the Hydra CLI pipeline. The new `sweep.py` provides the library-API equivalent that works with model classes instead of raw parameter counts. In Phase 7, the existing `planner.py` CLI can be refactored to use `sweep.py` internally, but that is out of scope for Phase 3.

**Why not modify planner.py?** The existing `SweepPlanner` is tightly coupled to the Hydra CLI (it has `@hydra.main`, reads from config, writes JSON files). The new sweep planning must work purely in Python with no file I/O, no Hydra, and returns data structures. Modifying `planner.py` to serve both purposes would create a muddled interface. Cleaner to have a fresh module that the library API uses, leaving the CLI planner untouched.

### Pattern 1: Experiment Grid as Data Structure

**What:** A `SweepPlan` dataclass containing a list of `Experiment` entries, with summary properties like `total_flops`, `num_experiments`, and human-readable `__repr__`.

**When to use:** Returned by `plan_sweep()`, consumed by the training engine (Phase 4), inspectable by users.

**Example:**
```python
# sweep.py
from dataclasses import dataclass, field


@dataclass
class Experiment:
    """A single planned experiment in an IsoFLOP sweep."""
    experiment_id: str
    compute_budget: float       # Target FLOPs (C)
    size_param_value: int       # Value for the model's size parameter (e.g., d_model=256)
    num_params: int             # Actual parameter count from probe model
    num_tokens: int             # Training tokens (D = C / (6*N))
    tokens_per_param: float     # D / N ratio


@dataclass
class SweepPlan:
    """Complete IsoFLOP sweep plan, inspectable before training."""
    experiments: list[Experiment]
    model_cls_name: str         # For display/logging
    size_param: str             # Name of size parameter
    compute_budgets: list[float]  # Original user-requested budgets

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
```

### Pattern 2: Size-Parameter-to-Param-Count Mapping via Probing

**What:** Generate a range of size parameter values, create a probe model at each value, call `num_params()` to get the actual parameter count. This builds a mapping from size_param_value to actual_params that the grid generator uses.

**When to use:** Inside `plan_sweep()`, before generating the experiment grid.

**Why probing over formula:** The prior decision notes that `12*L*d^2` overshoots by ~7.5x at small scales. Probing is architecture-agnostic -- it works for any model class, not just transformers.

**Example:**
```python
def _probe_model_sizes(model_cls, size_param, model_kwargs, size_values):
    """Create probe models at each size value to measure actual param counts.

    Args:
        model_cls: The model class.
        size_param: Name of the size parameter.
        model_kwargs: Other constructor kwargs.
        size_values: List of size parameter values to probe.

    Returns:
        List of (size_value, num_params) tuples, sorted by num_params.
    """
    from flops_fit.model_factory import create_model

    results = []
    for sv in size_values:
        model = create_model(model_cls, size_param, sv, model_kwargs)
        n = model.num_params()
        results.append((sv, n))

    return sorted(results, key=lambda x: x[1])
```

### Pattern 3: IsoFLOP Grid Generation from Probed Sizes

**What:** For each compute budget, select size values whose parameter counts span a reasonable range for that budget (not too few tokens, not too many). Calculate token count as `D = C / (6*N)` and filter infeasible entries.

**When to use:** Core of `plan_sweep()`.

**Example:**
```python
def _generate_grid(compute_budgets, size_param_map, num_sizes_per_budget=7):
    """Generate IsoFLOP experiment grid.

    Args:
        compute_budgets: List of compute budgets in FLOPs.
        size_param_map: List of (size_value, num_params) from probing.
        num_sizes_per_budget: How many model sizes to try per budget.

    Returns:
        List of Experiment entries.
    """
    experiments = []
    exp_idx = 0

    for budget in sorted(compute_budgets):
        # Filter to feasible sizes: need at least some minimum tokens
        feasible = [
            (sv, np) for sv, np in size_param_map
            if budget / (6 * np) >= np / 10  # at least N/10 tokens
        ]

        if not feasible:
            continue

        # Select evenly-spaced subset if we have more than needed
        if len(feasible) > num_sizes_per_budget:
            indices = np.linspace(0, len(feasible) - 1, num_sizes_per_budget, dtype=int)
            feasible = [feasible[i] for i in indices]

        for size_value, num_params in feasible:
            num_tokens = int(budget / (6 * num_params))
            experiments.append(Experiment(
                experiment_id=f"exp_{exp_idx:04d}",
                compute_budget=budget,
                size_param_value=size_value,
                num_params=num_params,
                num_tokens=num_tokens,
                tokens_per_param=num_tokens / num_params,
            ))
            exp_idx += 1

    return experiments
```

### Pattern 4: Wiring into find_optimal()

**What:** In `find_optimal()`, after validation passes, call `plan_sweep()` with the compute_budgets to create a `SweepPlan`. In Phase 3, this is the last step before `NotImplementedError` -- Phase 4 will consume the plan for training.

**Example:**
```python
# api.py (updated for Phase 3)
def find_optimal(
    model_cls,
    model_size_param,
    model_kwargs=None,
    dataset=None,
    loss_fn=None,
    compute_budgets=None,
    **kwargs,
):
    if model_kwargs is None:
        model_kwargs = {}

    validate_model_contract(model_cls, model_size_param, model_kwargs)

    if dataset is not None:
        validate_dataset(dataset)

    if loss_fn is not None:
        validate_loss_fn(loss_fn)

    # Phase 3: Generate sweep plan
    if compute_budgets is not None:
        plan = plan_sweep(
            model_cls=model_cls,
            size_param=model_size_param,
            model_kwargs=model_kwargs,
            compute_budgets=compute_budgets,
        )
        # Phase 4 will consume `plan` for training

    raise NotImplementedError(
        "find_optimal() validation and planning passed. "
        "Training engine not yet implemented."
    )
```

### Anti-Patterns to Avoid

- **Modifying the existing `planner.py`:** The existing SweepPlanner serves the Hydra CLI. The library API needs a different interface. Keep them separate until Phase 7 reconciles them.

- **Using analytical formulas to estimate param counts:** The `12*L*d^2` approximation is known to be inaccurate at small scales (~7.5x overshoot). Always probe actual models.

- **Generating model sizes as raw parameter counts and then searching for the right size_param value:** This is backwards. Generate size_param values directly (log-spaced), probe to get param counts, then use those param counts for the IsoFLOP calculation. The size_param values are what the model factory needs, not parameter targets.

- **Writing the sweep plan to a JSON file:** The library API returns data structures, not files. The existing planner.py writes JSON for the CLI pipeline. The library API's `SweepPlan` is a Python object that Phase 4 consumes directly in memory.

- **Making compute_budgets required in Phase 3:** Keep it optional in `find_optimal()`. Phase 3 generates the plan IF budgets are provided. Phase 4 will handle the case where budgets are None by either requiring them or providing defaults.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Logarithmic spacing of values | Manual loop with math.log | `numpy.logspace` | Already used in existing planner; handles edge cases |
| Model instantiation at different sizes | Direct `model_cls(**kwargs)` | `model_factory.create_model()` | Handles size_param injection, already validated in Phase 1 |
| Serialization of experiment plan | Custom JSON encoder | Dataclass `to_dict()` / `__repr__` | Consistent with existing ExperimentConfig, TrainingResult patterns |
| FLOP calculation | Custom formula per architecture | `C = 6 * N * D` approximation | Standard Chinchilla approximation; works for all dense architectures. Non-dense models (MoE, sparse) are v2 |

**Key insight:** Most of the hard work (model factory, validation, IsoFLOP math) already exists. Phase 3's job is to wire existing pieces together into a user-facing data structure and integrate with `find_optimal()`.

## Common Pitfalls

### Pitfall 1: Memory Pressure from Probing Many Model Sizes

**What goes wrong:** Probing 50+ model sizes by instantiating full models can consume significant memory, especially for large size_param values where models have millions/billions of parameters.

**Why it happens:** `create_model()` builds a full PyTorch model in memory. If the user's model allocates GPU memory or downloads pretrained weights in `__init__`, probing is expensive.

**How to avoid:** (a) Probe on CPU, not GPU -- the factory should not move models to GPU. (b) Delete each probe model after reading `num_params()` -- do not keep all probes alive simultaneously. (c) Limit the number of probe values to a reasonable range (20-30 sizes). (d) Use `torch.no_grad()` context if model __init__ does any computation.

**Warning signs:** OOM errors during `plan_sweep()` before any training starts.

### Pitfall 2: Size Parameter Values That Don't Make Sense

**What goes wrong:** Logarithmically spacing `d_model` from 64 to 4096 might produce values like 143 or 287 that aren't multiples of the number of attention heads, causing model construction errors.

**Why it happens:** Not all integer values are valid for all size parameters. `d_model` must typically be divisible by `num_heads`. Hidden sizes often need to be multiples of 64 for GPU efficiency.

**How to avoid:** After generating log-spaced size values, round them to sensible boundaries (multiples of 64 or powers of 2). Wrap each probe in try/except -- if a size value causes a construction error, skip it and log a warning. The grid doesn't need every value to work; it needs enough to cover the parameter range.

**Warning signs:** Many `TypeError` or `ValueError` from model construction during probing.

### Pitfall 3: Compute Budgets That Are Infeasible for the Model

**What goes wrong:** User passes `compute_budgets=[1e25]` but the model's maximum practical size is 1B params. The grid generation produces configs with unreasonable token counts (trillions of tokens for a small model).

**Why it happens:** IsoFLOP math says `D = C / (6*N)`. For small N and large C, D becomes astronomical.

**How to avoid:** Filter experiments where token counts are unreasonable. The existing planner uses `num_tokens < model_size // 10` as a lower bound. Add an upper bound too -- if `tokens_per_param > 1000`, the experiment is likely infeasible (most real experiments use 5-200 tokens per parameter). Warn the user if a budget produces no feasible experiments.

**Warning signs:** Experiments with millions of tokens per parameter, or entire budgets being filtered out.

### Pitfall 4: Duplicate or Overlapping Experiments Across Budgets

**What goes wrong:** Different compute budgets select the same size_param values, leading to very similar experiments that waste compute.

**Why it happens:** If the range of size values is small relative to the number of budgets, the same models appear in multiple budget levels.

**How to avoid:** This is actually correct IsoFLOP behavior -- the same model size trained for different amounts of data IS the experiment. The duplicates are only in model size, not in token count. Do NOT de-duplicate across budgets. However, within a single budget, avoid duplicate size values (which can happen with rounding).

**Warning signs:** User confused by seeing the same `d_model` values across budgets. Add a note in docs that this is by design.

### Pitfall 5: The `num_params()` Return Value Changes with Size

**What goes wrong:** Some model classes might have `num_params()` return values that don't monotonically increase with the size parameter, or that include non-trainable parameters.

**Why it happens:** `num_params()` is user-defined. Some models count embeddings, some don't. Some count buffers. The relationship between size_param and param count may not be perfectly monotonic due to rounding.

**How to avoid:** Sort probed (size_value, num_params) by num_params after probing. If the mapping is highly non-monotonic (e.g., param count decreases as size increases), warn the user. The IsoFLOP grid uses actual probed param counts, not assumed monotonic relationships.

**Warning signs:** Grid has experiments where smaller size_param values have more parameters than larger ones.

## Code Examples

### Example 1: User Creates a Sweep Plan

```python
import flops_fit
from flops_fit.sweep import plan_sweep

# User defines their model
class MyTransformer:
    def __init__(self, d_model=256, num_layers=6, vocab_size=50257):
        ...
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

# Create sweep plan
plan = plan_sweep(
    model_cls=MyTransformer,
    size_param="d_model",
    model_kwargs={"num_layers": 6, "vocab_size": 50257},
    compute_budgets=[1e17, 1e18, 1e19],
)

# Inspect the plan
print(plan)
# SweepPlan(35 experiments, 3 budgets, total_flops=1.11e+20)

print(plan.total_flops)
# 1.11e+20

print(plan.num_experiments)
# 35

# Look at individual experiments
for exp in plan.experiments[:3]:
    print(f"  C={exp.compute_budget:.1e}  d_model={exp.size_param_value}  "
          f"N={exp.num_params:,}  D={exp.num_tokens:,}")
```

### Example 2: Cost Estimation Before Training

```python
plan = plan_sweep(
    model_cls=MyTransformer,
    size_param="d_model",
    model_kwargs={"num_layers": 6},
    compute_budgets=[1e17, 1e18, 1e19],
)

# Check if compute cost is acceptable
total = plan.total_flops
print(f"Total compute: {total:.2e} FLOPs")

# Rough GPU-hours estimate (A100 ~312 TFLOPS)
gpu_hours = total / (312e12 * 3600)
print(f"Estimated GPU-hours: {gpu_hours:.1f}")

if gpu_hours > 100:
    print("Too expensive! Reducing budgets...")
    plan = plan_sweep(
        model_cls=MyTransformer,
        size_param="d_model",
        model_kwargs={"num_layers": 6},
        compute_budgets=[1e17, 1e18],  # Removed the big one
    )
```

### Example 3: Sweep Plan as Data Structure

```python
# The plan is a plain Python data structure, not opaque
plan = plan_sweep(...)

# Access as list
experiments = plan.experiments
assert isinstance(experiments, list)

# Each experiment is a dataclass
exp = experiments[0]
assert hasattr(exp, 'compute_budget')
assert hasattr(exp, 'size_param_value')
assert hasattr(exp, 'num_params')
assert hasattr(exp, 'num_tokens')

# Convert to dict for serialization
exp_dict = exp.to_dict() if hasattr(exp, 'to_dict') else dataclasses.asdict(exp)

# Filter experiments programmatically
big_exps = [e for e in plan.experiments if e.compute_budget >= 1e18]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `SweepPlanner` with hardcoded min/max model sizes | Library API: probe actual models at different size_param values | Phase 3 (this phase) | Works with any model class, not just parameter-count-based sizing |
| Sweep plan written to JSON file | Sweep plan returned as Python data structure | Phase 3 (this phase) | No file I/O needed; plan is inspectable in memory; consumed directly by training engine |
| No cost estimation | `SweepPlan.total_flops` property | Phase 3 (this phase) | Users can preview compute cost before committing GPU hours |
| Raw parameter counts in experiment grid | Size_param_value + actual_param_count per experiment | Phase 3 (this phase) | Training engine knows exactly what size_param value to pass to model factory |

## Open Questions

1. **What range of size_param values should be probed?**
   - What we know: The user specifies `model_size_param` (e.g., "d_model") but not the range. The existing planner uses `min_model_size` and `max_model_size` in parameter counts.
   - What's unclear: Should the library auto-detect a reasonable range by doing a small initial probe, or should users specify `min_size` / `max_size` for the size parameter?
   - Recommendation: Accept optional `min_size` and `max_size` kwargs in `plan_sweep()`. If not provided, use a heuristic: start from the probe value (64, from validation), double repeatedly to find the range where param counts span the feasible region for the given compute budgets. This auto-detection is useful for most cases; power users can override.

2. **Should `plan_sweep()` be a standalone function or a method on a class?**
   - What we know: The existing pattern is standalone functions (`validate_model_contract()`, `validate_dataset()`) and classes (`SweepPlanner`).
   - What's unclear: Whether sweep planning needs state that would benefit from a class.
   - Recommendation: Standalone function `plan_sweep()` returning a `SweepPlan` dataclass. No persistent state needed -- the function takes all inputs and returns all outputs. This matches the library-first design philosophy (simple function calls, not object lifecycle management).

3. **How does the `6*N*D` FLOP formula interact with non-transformer models?**
   - What we know: `C = 6*N*D` is specific to dense transformers (forward + backward = 6 multiply-accumulate ops per parameter per token). Other architectures may have different ratios.
   - What's unclear: Whether Phase 3 should allow users to override the FLOP formula.
   - Recommendation: Use `6*N*D` as the default. This is the standard Chinchilla approximation and is correct for the primary use case (transformer scaling laws). Add an optional `flops_per_param_per_token` parameter (default=6) to `plan_sweep()` for users who know their architecture's ratio. Non-dense models (MoE, sparse attention) are a v2 concern.

4. **Should the sweep plan include the model_kwargs for reproducibility?**
   - What we know: The plan needs `model_cls`, `size_param`, and `size_param_value` per experiment to recreate models.
   - What's unclear: Whether `model_kwargs` should be stored in the plan for full reproducibility.
   - Recommendation: Yes, store `model_kwargs` in the `SweepPlan` (not per experiment, since kwargs are the same for all experiments). This enables Phase 4's training engine to recreate models without needing to pass kwargs separately.

## Sources

### Primary (HIGH confidence)
- Existing codebase: `src/flops_fit/planner.py` (existing SweepPlanner implementation) -- direct inspection
- Existing codebase: `src/flops_fit/model_factory.py` (create_model, validate_model_contract) -- direct inspection
- Existing codebase: `src/flops_fit/api.py` (find_optimal stub) -- direct inspection
- Existing codebase: `src/flops_fit/trainer.py` (TrainingResult schema, experiment execution) -- direct inspection
- Existing codebase: `src/flops_fit/analyzer.py` (expected input format: compute_budget, model_size, num_tokens) -- direct inspection
- Existing codebase: `tests/conftest.py` (sample_sweep_configs schema) -- direct inspection
- `.planning/ROADMAP.md` -- Phase 3 success criteria and requirements
- `.planning/REQUIREMENTS.md` -- SWEEP-01, SWEEP-02 definitions

### Secondary (MEDIUM confidence)
- `.planning/research/ARCHITECTURE.md` -- overall system design context (partially superseded by library-first pivot)
- `.planning/phases/01-library-skeleton-and-model-interface/01-RESEARCH.md` -- model factory patterns, probe validation
- `.planning/phases/02-dataset-and-loss-interfaces/02-RESEARCH.md` -- validation integration pattern

### Tertiary (LOW confidence)
- None. This phase is pure Python using existing internal infrastructure.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, uses existing model factory and numpy
- Architecture: HIGH -- simple data structure + function pattern, consistent with existing codebase conventions
- Pitfalls: HIGH -- based on direct analysis of existing planner behavior and known prior decisions (7.5x overshoot, SwiGLU rounding)

**Research date:** 2026-02-16
**Valid until:** Indefinite (no version-sensitive dependencies in this phase)
