---
phase: 08-vit-and-cifar-example
plan: 02
subsystem: examples
tags: [vit, cifar10, pytorch, scaling-law, find_optimal, image-classification]

# Dependency graph
requires:
  - phase: 08-01
    provides: VisionTransformer, vit_loss_fn, CIFAR10Dataset in flops_fit.examples
  - phase: 07-02
    provides: example_programmatic.py pattern to mirror, test_examples.py to extend

provides:
  - example_vit_cifar.py runnable end-to-end demo using flops_fit.find_optimal() with ViT + CIFAR-10
  - make_vit_factory closure for embed_dim-parameterized VisionTransformer creation
  - _make_synthetic_cifar_dataset for mock mode (no network, no GPU)
  - TestViTContract, TestViTLossFunction, TestCIFAR10DatasetLazyLoad, TestViTExampleScript in test_examples.py
  - 201 total passing tests (13 new ViT tests)

affects: [09-packaging-and-docs, any example or demo referencing ViT/CIFAR]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - embed_dim as size parameter (vs d_model in GPT) — same find_optimal() API, different param name
    - Direct logits output (no tuple) — vit_loss_fn receives logits as-is, no unpacking
    - TensorDataset for synthetic CIFAR mock — randn images + randint.long() labels

key-files:
  created:
    - src/flops_fit/examples/example_vit_cifar.py
  modified:
    - tests/test_examples.py

key-decisions:
  - "make_vit_factory default num_layers=4 (not 6) for quick demo — balances expressiveness and speed"
  - "Labels use .long() explicitly in _make_synthetic_cifar_dataset — PyTorch cross_entropy requires LongTensor"
  - "vit_loss_fn imported directly from flops_fit.examples in example_vit_cifar.py — proves architecture-agnostic API"

patterns-established:
  - "ViT example mirrors GPT example structure exactly — same run(), main(), mock/real pattern"
  - "Architecture-agnostic find_optimal(): only model_cls, model_size_param, and loss_fn differ between GPT and ViT demos"

# Metrics
duration: 3min
completed: 2026-02-17
---

# Phase 8 Plan 02: ViT + CIFAR Example Summary

**Architecture-agnostic find_optimal() demo: ViT + CIFAR-10 example script with embed_dim as size param and direct-logits loss fn, plus 13 new ViT contract/smoke tests (201 total passing)**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-17T21:36:25Z
- **Completed:** 2026-02-17T21:39:26Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Created `example_vit_cifar.py` mirroring `example_programmatic.py` structure; runs in mock mode (no GPU/network) via `python -m flops_fit.examples.example_vit_cifar`
- Proves flops_fit.find_optimal() is architecture-agnostic: same API call, embed_dim instead of d_model, direct logits instead of tuple output
- Added 13 new ViT tests across 4 test classes; total test count increased from 188 to 201

## Task Commits

Each task was committed atomically:

1. **Task 1: Create example_vit_cifar.py** - `f43d39f` (feat)
2. **Task 2: Add ViT test classes to test_examples.py** - `a546d6c` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `src/flops_fit/examples/example_vit_cifar.py` - Runnable ViT + CIFAR scaling law demo; make_vit_factory, _make_synthetic_cifar_dataset, run(), main() with --real flag
- `tests/test_examples.py` - Appended TestViTContract, TestViTLossFunction, TestCIFAR10DatasetLazyLoad, TestViTExampleScript (13 new tests)

## Decisions Made

- `make_vit_factory` defaults to `num_layers=4` (not 6 as in module default) for faster demo sweep
- `_make_synthetic_cifar_dataset` explicitly calls `.long()` on labels tensor — PyTorch cross_entropy requires LongTensor dtype
- `vit_loss_fn` imported directly from `flops_fit.examples` (not re-implemented) — single source of truth

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. The benign PyTorch UserWarning about `enable_nested_tensor` (due to `norm_first=True` in VisionTransformer) was documented in STATE.md from Phase 08-01 and expected.

## User Setup Required

None - no external service configuration required. Mock mode (default) requires no GPU or network access.

## Next Phase Readiness

- Phase 8 complete: VisionTransformer + CIFAR10Dataset (08-01) + example script + tests (08-02)
- Both GPT/TinyStories and ViT/CIFAR-10 examples demonstrate architecture-agnostic find_optimal() API
- 201 tests passing; ready for Phase 9 (packaging and docs)

## Self-Check: PASSED

- `src/flops_fit/examples/example_vit_cifar.py` — FOUND
- `tests/test_examples.py` — FOUND
- `.planning/phases/08-vit-and-cifar-example/08-02-SUMMARY.md` — FOUND
- Commit `f43d39f` (Task 1) — FOUND
- Commit `a546d6c` (Task 2) — FOUND

---
*Phase: 08-vit-and-cifar-example*
*Completed: 2026-02-17*
