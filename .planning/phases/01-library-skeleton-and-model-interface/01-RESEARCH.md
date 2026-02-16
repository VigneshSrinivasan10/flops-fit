# Phase 1: Library Skeleton and Model Interface - Research

**Researched:** 2026-02-16
**Domain:** Python package restructuring, model factory pattern, interface validation
**Confidence:** HIGH

## Summary

Phase 1 transforms `flops_fit` from a Hydra-driven CLI pipeline into a library with a callable `find_optimal()` entry point. The core technical challenges are: (1) restructuring the package so `import flops_fit` exposes `find_optimal` without breaking existing imports, (2) building a model factory that creates model instances at different sizes by varying a single named parameter, and (3) validating the model contract (the `num_params()` method) at call time with clear errors.

This phase does NOT involve Hydra plugin systems, training loops, or GPU support. It is purely about the public API surface and the model instantiation machinery. The existing GPT model already has `count_parameters()` (not `num_params()`) and `from_config()`, so the model contract needs to be defined and the existing model will need adaptation in a later phase (Phase 7) to conform.

The technical domain is well-understood Python packaging and factory patterns. No external library research is needed -- this phase uses only stdlib and the existing project structure.

**Primary recommendation:** Define `find_optimal()` as a stub that validates inputs and builds model instances. Use a simple factory function (not a class or registry) that takes `model_cls`, `size_param`, and `kwargs`, then calls `model_cls(**{size_param: value, **kwargs})` for each size value. Validate `num_params()` exists on a probe instance.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | >=3.11 | Language runtime | Already in pyproject.toml |
| setuptools | (bundled) | Package build backend | Already configured in pyproject.toml with `src/` layout |
| pytest | >=8.0.0 | Testing | Already in dev dependencies |

### Supporting

No additional libraries needed for this phase. The model factory and validation are pure Python.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Simple factory function | `typing.Protocol` with `runtime_checkable` | Protocol adds formality but the phase requirements only ask for `num_params()` validation. Protocol is better suited for Phase 2+ when the full model interface (forward, configure_optimizers) is needed. Overkill here. |
| Direct class instantiation | ABC base class users must inherit | Forces inheritance on users. The project decision is "duck typing" -- any class with `num_params()` works. |
| `hasattr()` check | `isinstance()` with Protocol | `hasattr()` is simpler and sufficient for checking a single method. Protocol check is better when checking multiple methods. |

**Installation:**
```bash
pip install -e .
# or
uv pip install -e .
```

Already works. No changes to build system needed.

## Architecture Patterns

### Recommended Project Structure Changes

```
src/flops_fit/
    __init__.py          # ADD: export find_optimal
    api.py               # NEW: find_optimal() implementation
    model_factory.py     # NEW: model instantiation + validation
    model.py             # UNCHANGED (existing GPT)
    planner.py           # UNCHANGED
    trainer.py           # UNCHANGED
    analyzer.py          # UNCHANGED
    visualizer.py        # UNCHANGED
    conf/                # UNCHANGED
```

**Rationale:** Two new files, no moves, no renames. The existing code continues to work. `api.py` is the public entry point; `model_factory.py` is the internal machinery. This is the minimal change that satisfies all four success criteria.

### Pattern 1: Model Factory via Parameter Injection

**What:** A function that creates model instances by injecting a size parameter value into the model class constructor kwargs.

**When to use:** Every time the library needs to create a model at a specific size (sweep planning, training).

**Example:**
```python
# model_factory.py

def create_model(model_cls, size_param, size_value, model_kwargs):
    """Create a single model instance at a specific size.

    Args:
        model_cls: The model class to instantiate
        size_param: Name of the constructor parameter that controls model size
        size_value: Value for the size parameter
        model_kwargs: Other constructor keyword arguments

    Returns:
        An instance of model_cls
    """
    kwargs = {**model_kwargs, size_param: size_value}
    return model_cls(**kwargs)


def create_models_at_sizes(model_cls, size_param, size_values, model_kwargs):
    """Create model instances at multiple sizes.

    Args:
        model_cls: The model class to instantiate
        size_param: Name of the constructor parameter that controls model size
        size_values: List of values for the size parameter
        model_kwargs: Other constructor keyword arguments

    Returns:
        List of (size_value, model_instance) tuples
    """
    models = []
    for size_value in size_values:
        model = create_model(model_cls, size_param, size_value, model_kwargs)
        models.append((size_value, model))
    return models
```

### Pattern 2: Contract Validation via Probe Instance

**What:** Create a single "probe" model instance to validate the model contract before committing to a full sweep. Check that the required methods exist and return sensible values.

**When to use:** At the start of `find_optimal()`, before any expensive work.

**Example:**
```python
def validate_model_contract(model_cls, size_param, model_kwargs):
    """Validate that a model class meets the flops_fit contract.

    Creates a probe instance and checks:
    1. model_cls is callable (can be instantiated)
    2. size_param is accepted as a constructor argument
    3. Instance has num_params() method
    4. num_params() returns a positive integer

    Raises:
        TypeError: If model_cls is not callable
        TypeError: If size_param is not accepted
        TypeError: If num_params() is missing
        ValueError: If num_params() returns non-positive value
    """
    # Pick a small probe value for the size param
    # Use the smallest value from model_kwargs if size_param is there,
    # otherwise use a reasonable default
    probe_value = model_kwargs.get(size_param, 64)

    try:
        probe = create_model(model_cls, size_param, probe_value, model_kwargs)
    except TypeError as e:
        raise TypeError(
            f"Cannot create {model_cls.__name__} with "
            f"{size_param}={probe_value} and kwargs={model_kwargs}. "
            f"Error: {e}"
        ) from e

    if not hasattr(probe, 'num_params'):
        raise TypeError(
            f"{model_cls.__name__} does not have a num_params() method. "
            f"flops_fit requires model classes to expose num_params() -> int "
            f"so the library can measure model size at different scales."
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

    return probe  # Return probe for caller's convenience
```

### Pattern 3: Stub API with Full Signature

**What:** Define `find_optimal()` with its complete signature but implement only the model-related parts. The function validates inputs, creates probe models, but raises `NotImplementedError` for the actual sweep/training/analysis.

**When to use:** Phase 1 only. Later phases fill in the implementation.

**Example:**
```python
# api.py

def find_optimal(
    model_cls,
    model_size_param,
    model_kwargs=None,
    dataset=None,         # Phase 2
    loss_fn=None,         # Phase 2
    compute_budgets=None, # Phase 3
    **kwargs,
):
    """Find compute-optimal model size using scaling law experiments.

    Args:
        model_cls: A model class that accepts `model_size_param` as a
            constructor argument and exposes a `num_params()` method.
        model_size_param: Name of the constructor parameter that controls
            model size (e.g., "d_model", "hidden_size", "width").
        model_kwargs: Additional keyword arguments passed to model_cls
            constructor (everything except the size parameter).
        dataset: Training dataset (Phase 2).
        loss_fn: Loss function (Phase 2).
        compute_budgets: List of compute budgets in FLOPs (Phase 3).

    Returns:
        Result object with scaling law predictions (Phase 6).

    Raises:
        TypeError: If model_cls doesn't meet the model contract.
        ValueError: If model parameters are invalid.
    """
    if model_kwargs is None:
        model_kwargs = {}

    # Validate model contract
    validate_model_contract(model_cls, model_size_param, model_kwargs)

    # Phase 1: Model creation works
    # Phase 2+: Actual sweep execution
    raise NotImplementedError(
        "find_optimal() model validation passed. "
        "Full pipeline not yet implemented."
    )
```

### Anti-Patterns to Avoid

- **Importing the existing GPT in the public API:** The `__init__.py` currently imports `GPT`, `GPTConfig`, etc. The new `find_optimal` should NOT depend on the GPT model. It should work with ANY class that meets the contract.

- **Requiring model inheritance:** Do NOT create a base class that users must inherit from. The contract is "has `num_params()` method" -- duck typing, not inheritance.

- **Validating during the sweep instead of up front:** If a model class is missing `num_params()`, the user should learn this IMMEDIATELY when calling `find_optimal()`, not 30 minutes into a training sweep.

- **Breaking existing imports:** The existing `from flops_fit import GPT` must continue to work. Add new exports alongside existing ones.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Package structure with `src/` layout | Custom sys.path manipulation | setuptools `package-dir = {"" = "src"}` | Already configured in pyproject.toml |
| Checking if a method exists on an object | Custom metaclass or ABC | `hasattr(obj, 'num_params')` | Simple, Pythonic, no dependencies |
| Type checking constructor parameters | Manual inspect.signature parsing | Try/except on instantiation with clear error message | inspect.signature has edge cases with `__init_subclass__`, decorators, etc. Just try creating the object. |

**Key insight:** This phase is intentionally simple. The complexity is in defining the RIGHT interface, not in building complex machinery. The factory is 10 lines of code. The validation is 30 lines. The API stub is 20 lines. The risk is over-engineering, not under-engineering.

## Common Pitfalls

### Pitfall 1: Breaking Existing Imports

**What goes wrong:** Adding `find_optimal` to `__init__.py` alongside existing imports. If any of the existing imports fail (e.g., torch not installed), the new API also becomes inaccessible.

**Why it happens:** The current `__init__.py` eagerly imports `GPT`, `SweepPlanner`, etc., all of which require PyTorch and other dependencies. If those imports are at module level, any import error breaks everything.

**How to avoid:** Currently the existing imports DO work (the package installs fine). So the immediate fix is just to add `find_optimal` to the existing `__init__.py` exports. However, be aware that if a user has a broken torch installation, `import flops_fit` will fail entirely -- but that is already the case today. No regression.

**Warning signs:** `import flops_fit` raises `ModuleNotFoundError` for torch.

### Pitfall 2: Size Parameter Name Collision

**What goes wrong:** The `size_param` name collides with one of the keys in `model_kwargs`. For example, user passes `model_size_param="d_model"` and also `model_kwargs={"d_model": 256}`.

**Why it happens:** The factory merges `{size_param: value, **model_kwargs}`, so the size_param value should override. But if the user accidentally includes the size param in kwargs, it is ambiguous whether they want the factory to vary it or keep it fixed.

**How to avoid:** If `size_param` is present in `model_kwargs`, either: (a) warn and remove it from kwargs (it will be overridden anyway), or (b) raise a clear error. Option (a) is friendlier -- the user probably just forgot to remove it.

**Warning signs:** User gets confused because their size param value in kwargs is being ignored.

### Pitfall 3: num_params() Naming Confusion

**What goes wrong:** The existing GPT model has `count_parameters()`, not `num_params()`. Users of other frameworks may have `parameter_count()`, `n_parameters`, etc.

**Why it happens:** There is no universal standard for this method name in the PyTorch ecosystem.

**How to avoid:** The error message when `num_params()` is missing must be crystal clear about what is expected. Include the exact method signature. Consider mentioning common alternatives: "Did you mean count_parameters()? flops_fit requires num_params() -> int."

**Warning signs:** Users file bugs saying "my model has count_parameters but flops_fit doesn't recognize it."

### Pitfall 4: Model Instantiation Side Effects

**What goes wrong:** Creating a probe model instance triggers heavy side effects -- downloading pretrained weights, allocating GPU memory, running expensive initialization.

**Why it happens:** The validation pattern creates a real model instance to check the contract. If the model class does expensive work in `__init__`, the probe is costly.

**How to avoid:** This is acceptable for Phase 1. The probe is created once, not per size value. In future phases, consider allowing users to pass a pre-instantiated model for validation instead of creating a new one. For now, document that validation creates a temporary model instance.

**Warning signs:** `find_optimal()` takes unexpectedly long before any actual training starts.

## Code Examples

### Example 1: User-Facing API Usage

```python
import flops_fit

# User defines their model class
class MyTransformer:
    def __init__(self, d_model=256, num_layers=6, vocab_size=50257):
        self.d_model = d_model
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        # ... build model ...

    def num_params(self):
        # Return parameter count
        return sum(p.numel() for p in self.parameters())

# User calls the library
result = flops_fit.find_optimal(
    model_cls=MyTransformer,
    model_size_param="d_model",
    model_kwargs={"num_layers": 6, "vocab_size": 50257},
)
```

### Example 2: Model Factory Creating Multiple Sizes

```python
# Internal usage: create models at sweep sizes
sizes = [128, 256, 512, 1024]
models = create_models_at_sizes(
    model_cls=MyTransformer,
    size_param="d_model",
    size_values=sizes,
    model_kwargs={"num_layers": 6, "vocab_size": 50257},
)

for size_value, model in models:
    n_params = model.num_params()
    print(f"d_model={size_value}: {n_params:,} parameters")
```

### Example 3: Validation Error Messages

```python
# Missing num_params()
class BadModel:
    def __init__(self, width=64):
        pass

flops_fit.find_optimal(
    model_cls=BadModel,
    model_size_param="width",
)
# TypeError: BadModel does not have a num_params() method.
# flops_fit requires model classes to expose num_params() -> int
# so the library can measure model size at different scales.

# Wrong size parameter name
flops_fit.find_optimal(
    model_cls=MyTransformer,
    model_size_param="hidden_size",  # MyTransformer uses "d_model"
    model_kwargs={"num_layers": 6},
)
# TypeError: Cannot create MyTransformer with hidden_size=64 and
# kwargs={'num_layers': 6}. Error: __init__() got an unexpected
# keyword argument 'hidden_size'
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hydra `_target_` plugin resolution | Library-first: pass Python class directly | Project decision 2026-02-16 | No YAML needed for basic usage; Hydra becomes optional CLI wrapper |
| `count_parameters()` on existing GPT | `num_params()` as the contract method | Phase 1 decision | Existing GPT model will need a `num_params()` alias in Phase 7 |

**Note on prior architecture research:** The architecture research (`.planning/research/ARCHITECTURE.md`) recommended a Hydra-based plugin system with `typing.Protocol` and `_target_` resolution. The project has since pivoted to a library-first approach where users pass Python objects directly. The Hydra/Protocol patterns from that research are still relevant for the CLI wrapper (Phase 7) but are NOT the primary API pattern for Phase 1.

## Open Questions

1. **Should `num_params()` accept arguments?**
   - What we know: The existing GPT has `count_parameters(non_embedding=True)`. The Chinchilla paper uses non-embedding parameter count. Some users may want to report total params.
   - What's unclear: Whether the contract should be `num_params() -> int` (simplest) or `num_params(non_embedding: bool = True) -> int` (more flexible).
   - Recommendation: Start with `num_params() -> int` (no arguments). Users decide what to count. If the community consistently needs the flag, add it later. Simpler contract = lower adoption barrier.

2. **Should `find_optimal()` accept size values or generate them?**
   - What we know: Phase 3 (Sweep Planning) generates size values from compute budgets. Phase 1 only needs to verify that models CAN be created at different sizes.
   - What's unclear: Whether `find_optimal()` should accept explicit size values in Phase 1 for testing, or whether validation is sufficient.
   - Recommendation: Phase 1 `find_optimal()` does NOT accept size values. It validates the model contract and raises `NotImplementedError`. The model factory is available as an internal function for testing. Size value generation comes in Phase 3.

3. **What happens if size_param is also in model_kwargs?**
   - What we know: The factory merges `{size_param: value, **model_kwargs}`. If size_param is in kwargs, it gets overridden.
   - What's unclear: Should we warn, error, or silently override?
   - Recommendation: Issue a warning and remove it from kwargs. The user's intent is clear -- they want the library to vary this parameter.

## Sources

### Primary (HIGH confidence)
- Existing codebase analysis: `src/flops_fit/__init__.py`, `model.py`, `planner.py`, `trainer.py` -- direct inspection
- `pyproject.toml` -- build configuration, dependencies, entry points
- `.planning/PROJECT.md` -- project decisions including library-first pivot
- `.planning/ROADMAP.md` -- Phase 1 success criteria and requirements
- `.planning/REQUIREMENTS.md` -- API-01, API-02 requirement definitions

### Secondary (MEDIUM confidence)
- `.planning/research/ARCHITECTURE.md` -- prior architecture research (useful context but partially superseded by library-first pivot)
- `.planning/research/SUMMARY.md` -- project research summary

### Tertiary (LOW confidence)
- None. This phase is pure Python with no external dependencies to verify.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new dependencies, existing package structure
- Architecture: HIGH - simple factory pattern, well-understood Python packaging
- Pitfalls: HIGH - based on direct codebase inspection and common Python patterns

**Research date:** 2026-02-16
**Valid until:** Indefinite (no version-sensitive dependencies in this phase)
