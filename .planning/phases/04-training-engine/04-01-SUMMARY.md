---
phase: 04-training-engine
plan: 01
subsystem: training
tags: [pytorch, training-loop, chinchilla, device-placement, sgd, isoflop]

# Dependency graph
requires:
  - phase: 03-sweep-planning
    provides: "Experiment and SweepPlan dataclasses from sweep.py"
  - phase: 02-dataset-and-loss
    provides: "wrap_dataset() from data.py"
  - phase: 01-baseline
    provides: "create_model() from model_factory.py, TrainingRunner skeleton"
provides:
  - "_local_train(): real PyTorch training loop returning (loss, actual_flops, wall_time)"
  - "run_experiment_from_sweep(): bridges Experiment dataclass to training with error handling"
  - "run_sweep_from_plan(): iterates SweepPlan, resume-aware, writes results.json incrementally"
  - "_get_device(): module-level helper selecting cuda:0 or cpu at runtime"
affects: [04-02-training-engine, find_optimal]

# Tech tracking
tech-stack:
  added: [torch, time]
  patterns:
    - "Chinchilla FLOPs formula: C = 6 * actual_N * experiment.num_tokens"
    - "Device placement via torch.cuda.is_available() at runtime, no user config"
    - "Model cleanup: del model + torch.cuda.empty_cache() after each experiment"
    - "TDD: RED commit then GREEN commit per feature"

key-files:
  created: []
  modified:
    - src/flops_fit/trainer.py
    - tests/test_trainer.py

key-decisions:
  - "actual_flops uses actual model N from model.num_params() post-creation, not experiment.num_params (plan spec)"
  - "SGD optimizer with lr=0.01 default (simple, no scheduler needed for scaling law experiments)"
  - "Model cleanup with del + cuda.empty_cache() after each experiment to prevent GPU OOM across sweep"
  - "run_sweep_from_plan returns list[dict] (not list[TrainingResult]) matching existing run_sweep() API"

patterns-established:
  - "Experiment dataclass bridges sweep planning to training runner"
  - "run_experiment_from_sweep wraps _local_train with try/except for graceful failure handling"
  - "Incremental results.json write after each experiment enables crash-safe resume"

# Metrics
duration: 2min
completed: 2026-02-17
---

# Phase 4 Plan 01: Training Engine - Local Training Loop Summary

**Real PyTorch training loop (SGD, device-aware) wired to Experiment/SweepPlan dataclasses with Chinchilla FLOPs formula and crash-safe resume via incremental results.json writes**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-02-17T11:27:45Z
- **Completed:** 2026-02-17T11:29:43Z
- **Tasks:** 2 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments

- Implemented `_local_train()` with full PyTorch training loop: create_model, wrap_dataset, device placement (cuda:0 or cpu), SGD optimizer, epoch/batch loop, loss accumulation, Chinchilla FLOPs calculation, model cleanup
- Implemented `run_experiment_from_sweep()` wrapping `_local_train()` with try/except, returning TrainingResult with status='completed' or 'failed'
- Implemented `run_sweep_from_plan()` iterating a SweepPlan with resume support (skip completed experiment_ids), incremental results.json writes after each experiment
- Added `_get_device()` module-level helper: returns `cuda:0` if available, `cpu` otherwise
- All 139 tests pass (133 pre-existing + 6 new TestLocalTraining tests)

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Write failing tests for _local_train() and run_experiment_from_sweep()** - `17cb341` (test)
2. **Task 2 (GREEN): Implement _local_train(), run_experiment_from_sweep(), and run_sweep_from_plan()** - `2806791` (feat)

_Note: TDD tasks have separate RED (test) and GREEN (feat) commits_

## Files Created/Modified

- `src/flops_fit/trainer.py` - Added `_get_device()` helper, `_local_train()` method, `run_experiment_from_sweep()` method, `run_sweep_from_plan()` method; added imports for torch, time, create_model, wrap_dataset, SweepPlan, Experiment
- `tests/test_trainer.py` - Added `TestLocalTraining` class with 6 tests covering return types, Chinchilla FLOPs formula, device placement, failure handling, resume/skip behavior

## Implementation Details

**FLOPs formula:** `actual_flops = 6 * model.num_params() * experiment.num_tokens`
- Uses the actual N from the instantiated model (not experiment.num_params), measured after create_model() returns

**Device placement:** `_get_device()` calls `torch.cuda.is_available()` at runtime
- Returns `torch.device("cuda:0")` or `torch.device("cpu")`
- Model, inputs, and targets all moved to device before training

**Optimizer:** SGD with configurable learning_rate (default 0.01)
- No scheduler; scaling law experiments need only final loss, not convergence

**Cleanup:** `del model` then `torch.cuda.empty_cache()` (if CUDA available) after each experiment
- Prevents GPU OOM accumulation across multi-experiment sweeps

## Decisions Made

- Used actual `model.num_params()` (post-creation) for FLOPs calculation instead of `experiment.num_params` — ensures accuracy if model constructor snaps to valid sizes
- SGD chosen over Adam for simplicity (scaling law experiments need loss signal, not optimized training)
- `run_sweep_from_plan()` returns `list[dict]` (matching existing `run_sweep()` API) rather than `list[TrainingResult]` for consistency

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `TrainingRunner(mode="local")._local_train()` is fully implemented and tested
- `find_optimal()` can now call `run_sweep_from_plan()` to execute real training
- `run_experiment_from_sweep()` handles failures gracefully (status='failed' with error_message)
- Ready for Phase 4 Plan 02: wiring training engine into find_optimal() end-to-end flow

---
*Phase: 04-training-engine*
*Completed: 2026-02-17*

## Self-Check: PASSED

- FOUND: src/flops_fit/trainer.py
- FOUND: tests/test_trainer.py
- FOUND: .planning/phases/04-training-engine/04-01-SUMMARY.md
- FOUND commit 17cb341 (test(04-01): add failing tests for local training)
- FOUND commit 2806791 (feat(04-01): implement local training loop with device placement)
