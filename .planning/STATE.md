# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-16)

**Core value:** Given a compute budget, tell the user exactly how big their model should be and how much data to train on -- for their specific architecture and dataset.
**Current focus:** Phase 3 complete. Sweep planning wired into find_optimal() API. Ready for Phase 4.

## Current Position

Phase: 3 of 9 (Sweep Planning) -- COMPLETE
Plan: 2/2 complete
Status: find_optimal() returns SweepPlan when compute_budgets provided. 133 tests passing.
Last activity: 2026-02-17 -- 03-02 complete: API integration with 6 new integration tests

Progress: [███░░░░░░░] 33%

## Performance Metrics

**Velocity:**
- Total plans completed: 8
- Average duration: ~8min
- Total execution time: ~61min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-baseline | 3/3 | ~50min | ~17min |
| 01-skeleton | 1/1 | ~2min | ~2min |
| 02-dataset-and-loss | 2/2 | ~4min | ~2min |
| 03-sweep-planning | 2/2 | ~5min | ~2.5min |

**Recent Trend:**
- Last 3 plans: 02-02 (~2min), 03-01 (~3min), 03-02 (~2min)
- Trend: Consistent

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

### Pending Todos

None yet.

### Blockers/Concerns

- Accelerate version pin (>=1.0.0) not verified -- validate before adding to pyproject.toml
- Hydra + torchrun conflict needs Compose API workaround -- relevant for Phase 9

## Session Continuity

Last session: 2026-02-17
Stopped at: Completed 03-02-PLAN.md (API integration). Phase 3 complete. Ready for Phase 4.
Resume file: None
