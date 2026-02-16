---
phase: 02-dataset-and-loss-interfaces
plan: 01
subsystem: validation
tags: [torch, dataset, dataloader, loss, inspect, tdd]

# Dependency graph
requires:
  - phase: 01-skeleton
    provides: "Package structure and find_optimal stub"
provides:
  - "validate_dataset: accepts Dataset/DataLoader/IterableDataset, rejects invalid with HuggingFace hint"
  - "wrap_dataset: normalizes Dataset to DataLoader with configurable batch_size/shuffle/num_workers"
  - "validate_loss_fn: validates callable arity, detects class-vs-instance, handles C extensions"
  - "_get_name: human-readable name for loss functions"
affects: [02-02-api-integration, 03-training-loop]

# Tech tracking
tech-stack:
  added: []
  patterns: ["isinstance-chain validation", "inspect.signature arity checking", "graceful fallback for C extensions"]

key-files:
  created:
    - src/flops_fit/data.py
    - src/flops_fit/loss.py
    - tests/test_data.py
    - tests/test_loss.py
  modified: []

key-decisions:
  - "IterableDataset wrapping forces shuffle=False (torch requirement)"
  - "nn.Module signature inspection targets .forward not __call__ for accurate param counts"
  - "Uninspectable callables (C extensions) pass validation silently rather than failing"
  - "drop_last=True on all wrapped DataLoaders for consistent batch sizes"

patterns-established:
  - "Validation functions: raise TypeError with type name + actionable hint"
  - "inspect.signature in try/except separate from arity check to avoid catching own TypeError"

# Metrics
duration: 2min
completed: 2026-02-16
---

# Phase 02 Plan 01: Dataset and Loss Validation Summary

**Dataset validation with HuggingFace hints and loss function arity checking via inspect.signature with C-extension fallback**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-16T20:11:45Z
- **Completed:** 2026-02-16T20:13:08Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- validate_dataset accepts Dataset, DataLoader, IterableDataset; rejects invalid types with type name and HuggingFace `with_format('torch')` hint
- wrap_dataset normalizes any valid dataset type to DataLoader with configurable batch_size, shuffle, num_workers, drop_last=True
- validate_loss_fn checks callable arity via inspect.signature, detects class-vs-instance nn.Module mistakes, gracefully handles C-extension callables
- 27 tests total (14 data, 13 loss) all passing with ruff clean

## Task Commits

Each task was committed atomically:

1. **Task 1: TDD data.py -- dataset validation and wrapping** - `6f6a0e2` (feat)
2. **Task 2: TDD loss.py -- loss function validation** - `73538ec` (feat)

_Note: Task 1 was committed in a prior session. Task 2 required a bug fix (arity TypeError caught by outer except)._

## Files Created/Modified
- `src/flops_fit/data.py` - Dataset validation (validate_dataset) and DataLoader wrapping (wrap_dataset)
- `src/flops_fit/loss.py` - Loss function validation with signature inspection and class detection
- `tests/test_data.py` - 14 tests: acceptance, rejection, wrapping, passthrough, batch size
- `tests/test_loss.py` - 13 tests: acceptance, rejection, arity, class detection, C extension handling

## Decisions Made
- IterableDataset wrapping forces shuffle=False (torch requirement for iterable datasets)
- nn.Module signature inspection targets `.forward` not `__call__` for accurate parameter counts
- Uninspectable callables (C extensions, MagicMock) pass validation silently rather than failing
- drop_last=True on all wrapped DataLoaders for consistent batch sizes during training

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed arity TypeError caught by outer except clause**
- **Found during:** Task 2 (loss.py validation)
- **Issue:** The `raise TypeError` for wrong arity was inside a `try` block that also caught `TypeError` from `inspect.signature`, silently swallowing the arity error
- **Fix:** Moved signature inspection into its own try/except with early return on failure, keeping arity check outside the except scope
- **Files modified:** src/flops_fit/loss.py
- **Verification:** test_rejects_zero_arg_callable and test_rejects_one_arg_callable now pass
- **Committed in:** 73538ec (Task 2 commit)

**2. [Rule 1 - Bug] Removed unused torch import in test_loss.py**
- **Found during:** Task 2 (ruff check)
- **Issue:** `import torch` was unused (only `torch.nn` needed)
- **Fix:** Removed the unused import
- **Files modified:** tests/test_loss.py
- **Verification:** ruff check passes clean
- **Committed in:** 73538ec (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- data.py and loss.py provide the validation layer for Plan 02 (API integration)
- Plan 02 will wire validate_dataset, wrap_dataset, and validate_loss_fn into find_optimal()
- No blockers

## Self-Check: PASSED

- All 4 files exist on disk
- Commit 6f6a0e2 (Task 1) verified in history
- Commit 73538ec (Task 2) verified as HEAD

---
*Phase: 02-dataset-and-loss-interfaces*
*Completed: 2026-02-16*
