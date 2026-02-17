---
phase: 03-sweep-planning
plan: 01
subsystem: api
tags: [dataclass, isoflop, sweep, probe, model-factory, numpy]

# Dependency graph
requires:
  - phase: 01-baseline
    provides: "model_factory.create_model() for probing param counts"
provides:
  - "Experiment dataclass with compute_budget, size_param_value, num_params, num_tokens"
  - "SweepPlan dataclass with total_flops property and custom repr"
  - "plan_sweep() function for IsoFLOP grid generation"
  - "_probe_model_sizes() for architecture-agnostic param count measurement"
  - "_generate_size_values() for log-spaced GPU-friendly size values"
affects: [03-02-api-integration, 04-training-engine]

# Tech tracking
tech-stack:
  added: []
  patterns: [probe-based-sizing, feasibility-filtering, log-spaced-grid]

key-files:
  created:
    - src/flops_fit/sweep.py
    - tests/test_sweep.py
  modified: []

key-decisions:
  - "Standalone plan_sweep() function (not class method) -- matches library-first design"
  - "Log-spaced size values rounded to multiples of 64 for GPU efficiency"
  - "Probe models via create_model() with try/except to skip invalid sizes gracefully"
  - "Feasibility filter: tokens >= num_params/10, matching existing planner.py logic"
  - "configurable flops_per_param_per_token (default 6) for non-transformer architectures"

patterns-established:
  - "Probe-based sizing: instantiate model at each size, measure num_params(), delete"
  - "IsoFLOP grid: for each budget, select feasible sizes, compute tokens = C/(f*N)"

# Metrics
duration: 3min
completed: 2026-02-17
---

# Phase 3 Plan 1: Sweep Planning Summary

**Probe-based IsoFLOP grid generation with Experiment/SweepPlan dataclasses and feasibility filtering**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-17T06:22:25Z
- **Completed:** 2026-02-17T06:25:30Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files created:** 2

## Accomplishments
- Experiment dataclass capturing compute_budget, size_param_value, num_params, num_tokens, tokens_per_param
- SweepPlan dataclass with total_flops property, num_experiments, model_kwargs storage, and informative repr
- plan_sweep() function: generates log-spaced size values, probes actual models for param counts, builds IsoFLOP grid with feasibility filtering
- Graceful handling of invalid size values (try/except in probing) and infeasible configs (tokens < N/10 filtered)
- 16 new tests covering all 8 test categories from the plan

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for sweep module** - `557e7e4` (test)
2. **Task 1 GREEN: Implement sweep.py** - `66c9e28` (feat)

## Files Created/Modified
- `src/flops_fit/sweep.py` - Experiment, SweepPlan dataclasses; plan_sweep(), _probe_model_sizes(), _generate_size_values() functions
- `tests/test_sweep.py` - 16 tests across 8 test classes: dataclass creation, properties, basic sweep, probing accuracy, feasibility filtering, invalid sizes, optional params, repr

## Decisions Made
- Standalone `plan_sweep()` function rather than a class -- consistent with library-first API design (no object lifecycle)
- Size values rounded to multiples of 64 for GPU memory alignment efficiency
- Probe models individually with try/except -- skip invalid sizes with debug logging rather than failing
- Store model_kwargs in SweepPlan for reproducibility (training engine can recreate models)
- Default 20+ probe points (3x num_sizes_per_budget) to account for dedup and invalid sizes

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- sweep.py is ready for 03-02 (API integration: wire plan_sweep into find_optimal)
- SweepPlan data structure ready for Phase 4 training engine consumption
- 127 total tests passing, no regressions

---
*Phase: 03-sweep-planning*
*Completed: 2026-02-17*
