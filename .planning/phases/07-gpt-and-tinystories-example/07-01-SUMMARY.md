---
phase: 07-gpt-and-tinystories-example
plan: 01
subsystem: model
tags: [gpt, tinystories, examples, pytorch, huggingface, backward-compat, dataset]

# Dependency graph
requires:
  - phase: 06-results-object-and-api-integration
    provides: "find_optimal() returning Result; 169 tests passing baseline"
provides:
  - "flops_fit.examples package with GPT (num_params() contract), TinyStoriesDataset"
  - "Backward-compat re-export in flops_fit.model"
  - "TinyStoriesDataset lazy-loading PyTorch Dataset wrapper"
  - "flops_fit top-level TinyStoriesDataset export"
affects: [07-02, 07-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Thin re-export module: model.py becomes backward-compat facade over examples/gpt.py"
    - "Lazy-import for heavy dependencies (datasets, transformers) inside prepare_data() not at module level"
    - "examples/ as canonical home for reference implementations"

key-files:
  created:
    - src/flops_fit/examples/__init__.py
    - src/flops_fit/examples/gpt.py
    - src/flops_fit/examples/tinystories.py
  modified:
    - src/flops_fit/model.py
    - src/flops_fit/__init__.py

key-decisions:
  - "GPT implementation lives in examples/gpt.py; model.py is a thin backward-compat re-export wrapper"
  - "num_params() is the primary contract method (not n()); returns sum of all parameter numel()"
  - "tinystories.py imports datasets/transformers lazily inside prepare_data() to avoid import-time side effects"
  - "TinyStoriesDataset uses __getitem__ lazy-load fallback calling prepare_data() on first access"
  - "examples/__init__.py created before tinystories.py existed in plan sequence; created both together atomically"

patterns-established:
  - "examples/ package pattern: reference implementations importable as from flops_fit.examples import X"
  - "Lazy-load dataset pattern: _dataset=None sentinel, prepare_data() downloads, __getitem__/__len__ auto-call prepare_data()"

# Metrics
duration: 4min
completed: 2026-02-17
---

# Phase 7 Plan 01: GPT Examples Package and TinyStories Dataset Summary

**GPT implementation relocated to examples/gpt.py with num_params() contract method; backward-compat model.py re-export; lazy-loading TinyStoriesDataset wrapper for HuggingFace roneneldan/TinyStories**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-02-17T21:05:00Z
- **Completed:** 2026-02-17T21:09:02Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Created `flops_fit.examples` package with GPT reference implementation and `num_params()` library contract method
- Refactored `model.py` to thin backward-compat re-export; all existing imports continue to work unchanged
- Created `TinyStoriesDataset` lazy-loading PyTorch Dataset wrapper; instantiates with zero network I/O
- Updated `flops_fit.__init__` to export `TinyStoriesDataset` at top level
- All 169 pre-existing tests continue to pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Create examples package with GPT (moved + num_params() added)** - `e75f8d9` (feat)
2. **Task 2: Create TinyStoriesDataset in examples/tinystories.py** - included in `e75f8d9` (required for examples/__init__.py import chain)

_Note: Task 2 was created together with Task 1 because examples/__init__.py imports TinyStoriesDataset; splitting them would have caused an ImportError during Task 1 verification._

## Files Created/Modified
- `src/flops_fit/examples/__init__.py` - Package exports: GPT, GPTConfig, create_model_for_scaling, TinyStoriesDataset
- `src/flops_fit/examples/gpt.py` - Full GPT implementation with num_params() contract method added after count_parameters()
- `src/flops_fit/examples/tinystories.py` - Lazy-loading TinyStoriesDataset (datasets/transformers imported only in prepare_data())
- `src/flops_fit/model.py` - Replaced with thin backward-compat re-export wrapper
- `src/flops_fit/__init__.py` - Added TinyStoriesDataset import and export

## Decisions Made
- GPT lives canonically in `examples/gpt.py`; `model.py` becomes a pure re-export facade -- this is cleaner than keeping model.py as the canonical source and importing from it in examples
- `num_params()` returns all parameters (not non-embedding subset) -- matches library contract in model_factory.py
- `datasets` and `transformers` are lazy-imported inside `prepare_data()` only -- prevents HuggingFace startup overhead on every `import flops_fit`
- Created `tinystories.py` as part of Task 1's commit since `examples/__init__.py` needed it to import cleanly

## Deviations from Plan

None - plan executed exactly as written. Task 2's `tinystories.py` was created together with Task 1 files as an atomic unit (necessary since `examples/__init__.py` imports `TinyStoriesDataset`), but all content matches the plan specification exactly.

## Issues Encountered
None. The only implementation note: plan sequence had `__init__.py` import `TinyStoriesDataset` before `tinystories.py` was created. Created both in the same task to avoid import failures during verification. This matched the plan's content exactly, just committed together.

## User Setup Required
None - no external service configuration required. (TinyStories download happens only when `prepare_data()` is called explicitly, not at import time.)

## Next Phase Readiness
- `flops_fit.examples` package ready for Plan 02 (test_examples.py mock-based tests)
- `TinyStoriesDataset` ready for Plan 03 (example script using find_optimal + TinyStories)
- Backward compatibility confirmed: all 169 tests pass; `from flops_fit.model import GPT` still works

---
*Phase: 07-gpt-and-tinystories-example*
*Completed: 2026-02-17*

## Self-Check: PASSED

All expected files exist:
- FOUND: src/flops_fit/examples/__init__.py
- FOUND: src/flops_fit/examples/gpt.py
- FOUND: src/flops_fit/examples/tinystories.py
- FOUND: src/flops_fit/model.py (re-export wrapper)
- FOUND: src/flops_fit/__init__.py (with TinyStoriesDataset)

Commit verified: e75f8d9
