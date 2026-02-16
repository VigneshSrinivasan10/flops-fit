---
phase: 01-library-skeleton-and-model-interface
plan: 01
subsystem: api
tags: [factory-pattern, duck-typing, model-contract, validation]

# Dependency graph
requires: []
provides:
  - "find_optimal() public API entry point"
  - "Model factory: create_model, create_models_at_sizes"
  - "Contract validation: validate_model_contract (num_params check)"
affects: [02-dataset-and-loss-interfaces, 03-sweep-planning, 07-cli-wrapper]

# Tech tracking
tech-stack:
  added: []
  patterns: [factory-via-parameter-injection, probe-instance-validation, stub-api-with-full-signature]

key-files:
  created:
    - src/flops_fit/model_factory.py
    - src/flops_fit/api.py
    - tests/test_model_factory.py
    - tests/test_api.py
  modified:
    - src/flops_fit/__init__.py

key-decisions:
  - "Duck typing for model contract: any class with num_params() -> int, no base class required"
  - "Probe-based validation: create a small model instance to verify contract before expensive sweeps"
  - "Warning on size_param in model_kwargs: warn and remove rather than error"

patterns-established:
  - "Model contract: classes must expose num_params() -> int"
  - "Factory pattern: model_cls + size_param + size_value + model_kwargs"
  - "Up-front validation with actionable error messages including 'Did you mean count_parameters()?' hint"

# Metrics
duration: 2min
completed: 2026-02-16
---

# Phase 01 Plan 01: Library Skeleton Summary

**Model factory with parameter-injection pattern, num_params() contract validation, and find_optimal() stub API**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-16T20:04:02Z
- **Completed:** 2026-02-16T20:06:12Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Model factory creates instances at arbitrary sizes by injecting a named size parameter
- Contract validation catches missing num_params(), bad return values, and wrong constructor parameters with clear error messages
- find_optimal() validates model contract up front then raises NotImplementedError (stub for later phases)
- 12 tests covering creation, multi-size, contract pass/fail, API import and behavior

## Task Commits

Each task was committed atomically:

1. **Task 1: Create model factory, API stub, and package exports** - `dd2f61f` (feat)
2. **Task 2: Add tests for model factory and API** - `178bc45` (test)

## Files Created/Modified
- `src/flops_fit/model_factory.py` - Factory functions: create_model, create_models_at_sizes, validate_model_contract
- `src/flops_fit/api.py` - Public find_optimal() entry point with full signature
- `src/flops_fit/__init__.py` - Added find_optimal export alongside existing GPT/Planner exports
- `tests/test_model_factory.py` - 8 tests for factory creation and contract validation
- `tests/test_api.py` - 4 tests for API import, validation, and error handling

## Decisions Made
- Duck typing for model contract: any class with num_params() -> int, no ABC or Protocol required
- Probe instance uses size_value=64 as default for validation
- Warning (not error) when size_param appears in model_kwargs -- friendlier UX

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- find_optimal() is ready for Phase 2 to add dataset and loss_fn parameters
- Model factory is ready for Phase 3 sweep planning to call create_models_at_sizes
- Existing GPT model will need num_params() alias added in Phase 7

---
*Phase: 01-library-skeleton-and-model-interface*
*Completed: 2026-02-16*
