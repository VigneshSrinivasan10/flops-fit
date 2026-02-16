# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-16)

**Core value:** Given a compute budget, tell the user exactly how big their model should be and how much data to train on -- for their specific architecture and dataset.
**Current focus:** Phase 2 complete. Ready for Phase 3.

## Current Position

Phase: 2 of 9 (Dataset and Loss Interfaces) -- COMPLETE
Plan: 2/2 complete
Status: Phase 2 complete. find_optimal() validates model, dataset, and loss_fn at call time.
Last activity: 2026-02-16 -- 02-02 complete: API integration with 8 new tests (12 total in test_api.py)

Progress: [██░░░░░░░░] 18%

## Performance Metrics

**Velocity:**
- Total plans completed: 6
- Average duration: ~9min
- Total execution time: ~56min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-baseline | 3/3 | ~50min | ~17min |
| 01-skeleton | 1/1 | ~2min | ~2min |
| 02-dataset-and-loss | 2/2 | ~4min | ~2min |

**Recent Trend:**
- Last 3 plans: 01-01-skeleton (~2min), 02-01 (~2min), 02-02 (~2min)
- Trend: Accelerating

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

### Pending Todos

None yet.

### Blockers/Concerns

- Accelerate version pin (>=1.0.0) not verified -- validate before adding to pyproject.toml
- Hydra + torchrun conflict needs Compose API workaround -- relevant for Phase 9

## Session Continuity

Last session: 2026-02-16
Stopped at: Completed 02-02-PLAN.md (API integration). Phase 2 complete. Ready for Phase 3.
Resume file: None
