# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-16)

**Core value:** Given a compute budget, tell the user exactly how big their model should be and how much data to train on -- for their specific architecture and dataset.
**Current focus:** Phase 5 in progress. Linear-space NLS fitting with l_inf and IQR outlier detection complete.

## Current Position

Phase: 5 of 9 (Analysis and Fitting) -- IN PROGRESS
Plan: 1/? complete
Status: fit_power_law() refactored to linear-space NLS with l_inf and IQR outlier detection. 152 tests passing.
Last activity: 2026-02-17 -- 05-01 complete: linear-space NLS fitting with l_inf and IQR outlier detection

Progress: [█████░░░░░] 50%

## Performance Metrics

**Velocity:**
- Total plans completed: 10
- Average duration: ~8min
- Total execution time: ~79min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-baseline | 3/3 | ~50min | ~17min |
| 01-skeleton | 1/1 | ~2min | ~2min |
| 02-dataset-and-loss | 2/2 | ~4min | ~2min |
| 03-sweep-planning | 2/2 | ~5min | ~2.5min |
| 04-training-engine | 2/2 | ~4min | ~2min |
| 05-analysis-and-fitting | 1/? | ~18min | ~18min |

**Recent Trend:**
- Last 3 plans: 04-01 (~2min), 04-02 (~2min), 05-01 (~18min)
- Trend: 05-01 longer due to TDD + debugging initial guess bounds issue

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Library-first pivot: Python objects as input, not YAML/config
- Existing CLI/Hydra becomes example, not core
- Model contract: class + size parameter name + `num_params()`
- Mock train loss range is 1.5-15.0 (small models produce legitimately high loss ~9.6)
- Analyzer integration tests need 5+ compute budgets (3 produces identical optimal N, r²=0)
- create_model_for_scaling 12*L*d² approximation overshoots by ~7.5x at small scales due to SwiGLU + d_model rounding
- Known inconsistency: analyzer uses 2-decimal bucket rounding, visualizer uses 1-decimal
- Hydra config tests use initialize_config_dir (not config_module) since conf/ has no __init__.py
- Duck typing for model contract: num_params() -> int, no base class required
- Probe-based validation creates small instance (size=64) to verify contract up front
- Warning (not error) when size_param appears in model_kwargs
- IterableDataset wrapping forces shuffle=False (torch requirement)
- nn.Module signature inspection targets .forward not __call__ for accurate param counts
- Uninspectable callables (C extensions) pass validation silently
- drop_last=True on all wrapped DataLoaders for consistent batch sizes
- Standalone plan_sweep() function (not class) -- matches library-first design
- Log-spaced size values rounded to multiples of 64 for GPU efficiency
- Probe models via create_model() with try/except to skip invalid sizes gracefully
- Feasibility filter: tokens >= num_params/10 (matching existing planner.py)
- Configurable flops_per_param_per_token (default 6) for non-transformer architectures
- _local_train() uses actual model.num_params() post-creation for FLOPs (not experiment.num_params)
- SGD optimizer with lr=0.01 default for local training (scaling law experiments need loss signal, not optimized convergence)
- Model cleanup: del model + torch.cuda.empty_cache() after each experiment to prevent GPU OOM across sweeps
- run_sweep_from_plan returns list[dict] (matching existing run_sweep() API) for consistency
- Lazy-import TrainingRunner inside training branch to avoid circular imports and keep startup fast
- train=True default makes training the happy path when dataset+loss_fn provided (explicit opt-out via train=False)
- output_dir defaults to 'outputs' (string not Path) so TrainingRunner handles Path conversion internally
- Linear-space NLS replaces log-space regression: unbiased when loss has additive baseline (irreducible entropy)
- fit_power_law() parametrizes optimization as [log10(k), a, l_inf] with bounds [-10,5] x [-1,2] x [0,inf]
- Test x range for l_inf recovery must be small enough that l_inf is a significant fraction of y_min (logspace(1,5) not logspace(10,20))

### Pending Todos

None yet.

### Blockers/Concerns

- Accelerate version pin (>=1.0.0) not verified -- validate before adding to pyproject.toml
- Hydra + torchrun conflict needs Compose API workaround -- relevant for Phase 9

## Session Continuity

Last session: 2026-02-17
Stopped at: Completed 05-01-PLAN.md (linear-space NLS fitting with l_inf and IQR outlier detection). 152 tests passing.
Resume file: None
