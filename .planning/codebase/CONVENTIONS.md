# Coding Conventions

**Analysis Date:** 2026-02-15

## Naming Patterns

**Files:**
- `snake_case.py` for all modules: `analyzer.py`, `planner.py`, `trainer.py`, `visualizer.py`, `model.py`
- Test files: `test_<module>.py` mirroring the source module name

**Classes:**
- `PascalCase` throughout: `ScalingLawAnalyzer`, `SweepPlanner`, `TrainingRunner`, `ScalingVisualizer`, `GPT`, `GPTConfig`
- Dataclasses use the same `PascalCase` convention: `PowerLawFit`, `ScalingAnalysis`, `ExperimentConfig`, `TrainingResult`

**Functions and Methods:**
- `snake_case` for all functions and methods: `fit_power_law`, `generate_sweep`, `run_experiment`, `plot_isoflops`
- Private methods prefixed with `_`: `_mock_train`, `_setup_style`, `_build_cache`
- Class method named `from_config` for alternative constructors (see `GPT.from_config`)

**Variables:**
- `snake_case` for local variables: `compute_budget`, `model_size`, `optimal_df`
- Single uppercase letters used for mathematical quantities: `C` (compute), `N` (params), `D` (tokens), `L` (loss)
- Constants in `UPPER_SNAKE_CASE`: `COMPUTE_COLORS`

**Type Annotations:**
- All function signatures include type hints
- Return types always annotated
- Use `str | Path` union syntax (Python 3.10+ style), not `Union[str, Path]`
- Use `Optional[T]` in `model.py` but `T | None` in other modules (slight inconsistency)
- `Literal` type used for constrained string params: `Literal["completed", "failed", "skipped"]`, `Literal["paper", "notebook"]`

## Code Style

**Formatting:**
- Ruff enforcer, configured in `pyproject.toml`
- Line length: 100 characters
- Target: Python 3.11

**Linting:**
- Ruff lint with rule sets: `E` (pycodestyle errors), `F` (pyflakes), `I` (isort), `W` (pycodestyle warnings)
- No `type: ignore` or `noqa` suppressions found in codebase

## Import Organization

**Order (enforced by ruff isort):**
1. Standard library: `dataclasses`, `pathlib`, `typing`, `json`, `logging`, `math`
2. Third-party: `hydra`, `omegaconf`, `numpy`, `scipy`, `pandas`, `torch`, `matplotlib`, `tqdm`
3. Local: `from flops_fit.<module> import ...`

**Pattern observed in every module:**
```python
# stdlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import json
import logging

# third-party
import hydra
from omegaconf import DictConfig
import numpy as np

# local (only in __init__.py)
from flops_fit.model import GPT, GPTConfig
```

**Path Aliases:**
- None. Imports use the full package path `flops_fit.<module>`.

## Module-Level Logger Pattern

Every module defines a module-level logger immediately after imports:
```python
logger = logging.getLogger(__name__)
```

This logger is used throughout the module for `logger.info(...)`, `logger.debug(...)`, `logger.warning(...)`, `logger.error(...)`.

## Docstrings

**Module docstrings:**
- Every module has a top-level docstring describing purpose, key concepts, and CLI usage examples:
```python
"""
Sweep Configuration Planner

Generates configurations for scaling law experiments...

Usage:
    uv run sl-plan
    uv run sl-plan compute.min_flops=1e18 compute.max_flops=1e22
"""
```

**Class docstrings:**
- Describe the class purpose, how it works (numbered steps), and list `Attributes:` with types

**Method docstrings:**
- All public methods have docstrings with `Args:` and `Returns:` sections
- Single-line summary followed by longer explanation where needed

## Dataclasses

Dataclasses are used for data containers (not business logic):
- `@dataclass` decorator without `frozen=True` (mutable)
- `field(default_factory=...)` for mutable defaults
- `field(init=False)` for derived attributes computed in `__post_init__`
- Every dataclass has a `to_dict(self) -> dict` method for JSON serialization

Example pattern from `src/flops_fit/planner.py`:
```python
@dataclass
class ExperimentConfig:
    """Configuration for a single training experiment."""
    experiment_id: str
    compute_budget: float
    model_size: int
    num_tokens: int
    tokens_per_param: float = field(init=False)

    def __post_init__(self):
        self.tokens_per_param = self.num_tokens / self.model_size if self.model_size > 0 else 0

    def to_dict(self) -> dict:
        ...
```

## Class Design (Service Classes)

Service classes (`ScalingLawAnalyzer`, `SweepPlanner`, `TrainingRunner`, `ScalingVisualizer`) follow a consistent pattern:
- `__init__` accepts path arguments as `str | Path`, immediately converts to `Path` via `Path(arg)`
- `__init__` creates output directories: `self.output_dir.mkdir(parents=True, exist_ok=True)`
- Methods are small and focused on a single concern
- `analyze()` / `run_sweep()` / `plot_all()` are the high-level orchestration methods
- `main()` at module level is the Hydra entry point, constructing the service class and calling the orchestrator

## Error Handling

**Strategy:** Fail fast with meaningful messages for expected errors; catch-and-record for experiment failures.

**Patterns:**
- `FileNotFoundError` raised with instructive messages guiding the user to the prior step:
```python
raise FileNotFoundError(
    f"Results file not found: {self.results_path}. "
    "Run 'sl-train' first to generate results."
)
```
- Experiment failures are caught and recorded as `status="failed"` with `error_message` rather than aborting the sweep:
```python
except Exception as e:
    logger.error(f"Experiment {config['experiment_id']} failed: {e}")
    return TrainingResult(..., status="failed", error_message=str(e))
```
- `plot_all()` uses `try/except` with `logger.warning` to produce partial output rather than failing entirely
- `ValueError` raised for invalid configuration (e.g., unsupported parametrization, insufficient data for fitting)

## Serialization

All data objects use `to_dict() -> dict` for JSON serialization. JSON files are written with `indent=2`. Files are opened with `open(path)` (no explicit encoding) and `json.load` / `json.dump`.

## Configuration

Hydra is used for all CLI configuration:
- `@hydra.main(version_base=None, config_path="conf", config_name="<module>")` decorates `main(cfg: DictConfig)`
- Config files live in `src/flops_fit/conf/`
- Each service class accepts plain Python types in `__init__`, keeping it testable without Hydra

## Logging vs Printing

- `logger.*` for operational messages (progress, file paths saved, counts)
- `print()` used only in `main()` functions for summary output to the user (results tables, formatted reports)

---

*Convention analysis: 2026-02-15*
