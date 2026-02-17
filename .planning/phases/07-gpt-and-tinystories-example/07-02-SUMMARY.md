---
phase: 07-gpt-and-tinystories-example
plan: 02
subsystem: examples
tags: [gpt, tinystories, examples, pytorch, argparse, mode-parameter, mock, testing]

# Dependency graph
requires:
  - phase: 07-gpt-and-tinystories-example-plan-01
    provides: "flops_fit.examples package with GPT (num_params() contract), TinyStoriesDataset"
  - phase: 06-results-object-and-api-integration
    provides: "find_optimal() returning Result; TrainingRunner with mode parameter"
provides:
  - "find_optimal() accepts mode parameter (default='local'), threads it to TrainingRunner"
  - "example_programmatic.py: runnable demo with mode='mock' by default (no GPU, no network)"
  - "example_cli_wrapper.py: argparse CLI wrapper demo with --real flag for local training"
  - "tests/test_examples.py: 16 tests covering GPT contract, TinyStories lazy-load, mocked HF, example script smoke tests"
  - "188 tests passing"
affects: [07-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "mode='mock' default for demo/CI scripts — real training opt-in via --real flag"
    - "Synthetic TensorDataset as drop-in for TinyStories in mock/CI mode (no mocking needed)"
    - "Direct mock injection (_dataset/_tokenizer) to test TinyStoriesDataset without HuggingFace"

key-files:
  created:
    - src/flops_fit/examples/example_programmatic.py
    - src/flops_fit/examples/example_cli_wrapper.py
    - tests/test_examples.py
  modified:
    - src/flops_fit/api.py
    - tests/test_api.py

key-decisions:
  - "mode='local' default in find_optimal() preserves all existing test behavior (no regression)"
  - "Example scripts default to mode='mock' with synthetic TensorDataset — user must pass --real to download TinyStories"
  - "gpt_loss_fn reshapes (B, T, V) logits to (B*T, V) and labels to (B*T,) for F.cross_entropy compatibility"
  - "TinyStoriesDataset mock tests inject _dataset/_tokenizer directly (no patch of load_dataset) — cleaner and doesn't rely on internal import paths"

patterns-established:
  - "Synthetic dataset pattern: torch.randint TensorDataset as zero-dependency drop-in for real datasets in demos/CI"
  - "Model factory pattern: closure over fixed arch params (num_layers, num_heads) returning (d_model: int) -> Model"

# Metrics
duration: 3min
completed: 2026-02-17
---

# Phase 7 Plan 02: mode Parameter + Example Scripts Summary

**find_optimal() mode parameter threading to TrainingRunner; two runnable example scripts (programmatic + argparse CLI) defaulting to mock mode; 16-test suite covering GPT contract, lazy TinyStories, and example smoke tests**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-02-17T21:11:06Z
- **Completed:** 2026-02-17T21:14:40Z
- **Tasks:** 3 (Task 0, Task 1, Task 2)
- **Files modified:** 5

## Accomplishments
- Added `mode: str = "local"` parameter to `find_optimal()` and threaded it to `TrainingRunner(mode=mode, ...)`
- Created `example_programmatic.py` with model factory, GPT loss function, synthetic dataset, and mock-default workflow
- Created `example_cli_wrapper.py` with argparse CLI exposing all key parameters; `--real` flag enables TinyStories + local training
- Created `tests/test_examples.py` with 16 tests covering all example components without network access
- Full suite grew from 169 to 188 tests passing (16 new test_examples.py + 3 new test_api.py mode tests)

## Task Commits

Each task was committed atomically:

1. **Task 0: Add mode parameter to find_optimal() in api.py** - `1bee9bb` (feat)
2. **Task 1: Create programmatic and CLI wrapper example scripts** - `21c79ed` (feat)
3. **Task 2: Create tests/test_examples.py with mocked HuggingFace** - `0759a30` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `src/flops_fit/api.py` - Added `mode: str = "local"` parameter; updated docstring; changed `TrainingRunner(mode="local", ...)` to `TrainingRunner(mode=mode, ...)`
- `tests/test_api.py` - Appended `TestFindOptimalModeParameter` class with 3 tests for mode parameter signature and behavior
- `src/flops_fit/examples/example_programmatic.py` - Full demo script with make_model_factory(), gpt_loss_fn(), _make_synthetic_dataset(), run(), main()
- `src/flops_fit/examples/example_cli_wrapper.py` - Argparse CLI with build_parser(), make_model_factory(), gpt_loss_fn(), main()
- `tests/test_examples.py` - 16 tests across 5 test classes; no HuggingFace network calls

## Decisions Made
- `mode='local'` default in `find_optimal()` preserves all 169 pre-existing tests unchanged; no regression risk
- Example scripts default to `mode='mock'` with synthetic `TensorDataset` — real training is opt-in via `--real` flag, which downloads TinyStories
- `gpt_loss_fn` uses `.view(-1, VOCAB_SIZE)` and `.view(-1)` reshaping for correct `F.cross_entropy` input shapes from GPT's (B, T, V) output
- TinyStoriesDataset test mock injects `_dataset` and `_tokenizer` directly rather than patching `load_dataset` — simpler and robust to import path changes

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. All components implemented straightforwardly following plan specifications.

## User Setup Required

None - no external service configuration required. Both example scripts run without GPU or network in default (mock) mode. Real training mode requires internet and `--real` flag.

## Next Phase Readiness
- `example_programmatic.py` and `example_cli_wrapper.py` are the primary Phase 7 user-facing deliverables (EX-01, EX-03)
- `test_examples.py` provides CI coverage without network dependencies
- Plan 03 (if any) can use the established mock/real pattern for further example development

---
*Phase: 07-gpt-and-tinystories-example*
*Completed: 2026-02-17*

## Self-Check: PASSED

All expected files exist:
- FOUND: src/flops_fit/api.py
- FOUND: tests/test_api.py
- FOUND: src/flops_fit/examples/example_programmatic.py
- FOUND: src/flops_fit/examples/example_cli_wrapper.py
- FOUND: tests/test_examples.py
- FOUND: .planning/phases/07-gpt-and-tinystories-example/07-02-SUMMARY.md

Commits verified: 1bee9bb, 21c79ed, 0759a30
