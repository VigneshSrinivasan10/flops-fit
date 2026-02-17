---
phase: 04-training-engine
plan: "02"
subsystem: api
tags: [training, find_optimal, sweep, integration-tests, pytorch]

# Dependency graph
requires:
  - phase: 04-01
    provides: TrainingRunner.run_sweep_from_plan() implementation with _local_train() and resume support
  - phase: 03-02
    provides: plan_sweep() and SweepPlan integrated into find_optimal()
provides:
  - find_optimal() with training execution path (train=True + dataset + loss_fn -> list[dict])
  - Integration tests covering training path and resume behavior
  - results.json written incrementally to output_dir during sweep
affects: [05-scaling-laws, 06-analyzer, 09-cli]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy import of TrainingRunner inside if-block to avoid circular imports and keep startup fast"
    - "train=True gate: training only executes when all three (train, dataset, loss_fn) are present"
    - "Return type polymorphism: list[dict] vs SweepPlan based on execution context"

key-files:
  created: []
  modified:
    - src/flops_fit/api.py
    - tests/test_api.py

key-decisions:
  - "Lazy-import TrainingRunner inside training branch to avoid circular imports and keep import-time overhead low"
  - "train=True default makes training the happy path when dataset+loss_fn provided (explicit opt-out via train=False)"
  - "output_dir defaults to 'outputs' (string not Path) so TrainingRunner handles Path conversion internally"

patterns-established:
  - "find_optimal() is the single user-facing entry point — all execution modes flow through it"
  - "SweepPlan-only mode preserved: omitting dataset or passing train=False returns plan for inspection"

# Metrics
duration: 2min
completed: 2026-02-17
---

# Phase 4 Plan 02: Training Execution Wired into find_optimal() Summary

**find_optimal() now executes real PyTorch training when dataset + loss_fn + compute_budgets provided, returning list[dict] results; SweepPlan-only path preserved via train=False or omitting dataset**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-17T11:31:50Z
- **Completed:** 2026-02-17T11:33:49Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Extended find_optimal() signature with `train`, `output_dir`, and `resume` parameters
- Training execution path calls TrainingRunner.run_sweep_from_plan() when train=True + dataset + loss_fn all provided
- SweepPlan-only path preserved: train=False or omitting dataset/loss_fn returns SweepPlan unchanged
- results.json written incrementally to output_dir after each experiment
- resume=True skips already-completed experiments from prior runs (loss values preserved)
- 5 new integration tests covering all execution paths; full 144-test suite passing

## Return Type Contract

| Condition | Return Type |
|-----------|-------------|
| `train=True` and `dataset is not None` and `loss_fn is not None` | `list[dict]` with training results |
| `train=False` | `SweepPlan` |
| `dataset is None` or `loss_fn is None` | `SweepPlan` |
| `compute_budgets is None` | `NotImplementedError` |

Each result dict contains: `experiment_id`, `final_loss`, `actual_flops`, `wall_time_seconds`, `status`, `compute_budget`, `model_size`, `num_tokens`, `timestamp`, `error_message`.

## Training Trigger Logic

```python
if train and dataset is not None and loss_fn is not None:
    from flops_fit.trainer import TrainingRunner  # lazy import
    runner = TrainingRunner(mode="local", output_dir=output_dir)
    return runner.run_sweep_from_plan(
        plan=plan, model_cls=model_cls, size_param=model_size_param,
        model_kwargs=model_kwargs, dataset_or_loader=dataset,
        loss_fn=loss_fn, resume=resume,
    )
```

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend find_optimal() to execute training when dataset + loss_fn provided** - `05db163` (feat)
2. **Task 2: Add integration tests for training execution path in find_optimal()** - `f5f8036` (test)

**Plan metadata:** (docs: complete plan — committed separately)

## Files Created/Modified
- `src/flops_fit/api.py` - Added train/output_dir/resume params, training execution branch, updated docstring
- `tests/test_api.py` - Added TestFindOptimalTraining class with 5 integration tests

## Decisions Made
- Lazy-import `TrainingRunner` inside the training branch to avoid circular imports and keep startup fast for non-training use cases
- `train=True` as default makes training the ergonomic happy path; `train=False` is the explicit opt-out for inspection mode
- `output_dir` parameter type is `str` (not `Path`) so TrainingRunner handles Path conversion internally — consistent with the plan spec and avoids adding `from pathlib import Path` to api.py

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. The implementation was straightforward: plan had explicit code to write, tests were pre-specified, and the underlying TrainingRunner.run_sweep_from_plan() (from 04-01) worked correctly end-to-end.

## Edge Cases Discovered

- End-to-end smoke test produced 7 experiments for a single 1e8 FLOPs budget (the sweep planner generates multiple model sizes per budget); this is correct behavior
- `python` command not available in this environment (only `uv run python`); adjusted verify commands accordingly

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `find_optimal()` is now a complete user-facing API: validates model contract, plans sweep, executes training, saves results
- Phase 5 (scaling laws) can now fit power laws to the results list[dict] returned by find_optimal()
- 144 tests passing, no regressions

## Self-Check: PASSED

- src/flops_fit/api.py: FOUND
- tests/test_api.py: FOUND
- 04-02-SUMMARY.md: FOUND
- Commit 05db163 (Task 1): FOUND
- Commit f5f8036 (Task 2): FOUND

---
*Phase: 04-training-engine*
*Completed: 2026-02-17*
