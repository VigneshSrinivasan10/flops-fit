---
phase: 08-vit-and-cifar-example
plan: 01
subsystem: examples
tags: [vit, vision-transformer, cifar10, torchvision, image-classification, pytorch]

# Dependency graph
requires:
  - phase: 07-gpt-and-tinystories-example
    provides: examples/ package structure (gpt.py, tinystories.py, __init__.py) and model contract pattern (num_params())

provides:
  - VisionTransformer class in examples/vit.py with embed_dim as size param
  - vit_loss_fn (direct logits, not tuple -- structural contrast with gpt_loss_fn)
  - CIFAR10Dataset wrapper with lazy torchvision import
  - Updated examples/__init__.py exporting all 7 names

affects:
  - 08-vit-and-cifar-example phase 02 (example scripts using VisionTransformer + CIFAR10Dataset)
  - Any future phase demonstrating architecture-agnostic scaling

# Tech tracking
tech-stack:
  added:
    - torchvision (lazy-imported for CIFAR10Dataset, already in PyTorch ecosystem)
  patterns:
    - ViT forward() returns direct logits (not tuple) -- image classification pattern
    - Lazy torchvision import inside _prepare_data() (same as TinyStories lazy HF import)
    - nn.TransformerEncoder with norm_first=True (pre-norm ViT variant)
    - Learnable positional embeddings initialized with std=0.02

key-files:
  created:
    - src/flops_fit/examples/vit.py
    - src/flops_fit/examples/cifar.py
  modified:
    - src/flops_fit/examples/__init__.py

key-decisions:
  - "VisionTransformer.forward() returns logits DIRECTLY (not a tuple) -- structurally different from GPT"
  - "vit_loss_fn takes (logits, labels) with no tuple unpacking -- proves library handles both output patterns"
  - "Lazy-import torchvision inside _prepare_data() only (zero import-time overhead, matches tinystories.py)"
  - "norm_first=True (pre-norm transformer) -- better training stability than post-norm"
  - "Assertions for incompatible patch_size and embed_dim/num_heads at __init__ time (fail fast)"

patterns-established:
  - "Image model pattern: forward() returns logits directly, loss_fn takes (logits, labels)"
  - "Lazy dataset loading: _dataset=None until first __len__/__getitem__ call"

# Metrics
duration: 2min
completed: 2026-02-17
---

# Phase 8 Plan 01: VisionTransformer + CIFAR10Dataset Summary

**Custom ViT model class with embed_dim as size parameter + lazy CIFAR-10 dataset wrapper, proving flops_fit handles image classification with direct logit output (no tuple unpacking)**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-17T21:31:35Z
- **Completed:** 2026-02-17T21:34:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- VisionTransformer(embed_dim=N) class with patch projection, learnable CLS token, pre-norm transformer encoder, classification head; forward() returns (B, 10) logits directly
- vit_loss_fn(logits, labels) function taking direct logits with no tuple unpacking -- structural contrast with gpt_loss_fn that unpacks (logits, loss) tuple
- CIFAR10Dataset with lazy torchvision import inside _prepare_data(); _dataset=None until first __len__/__getitem__ call
- Updated examples/__init__.py to export all 7 names; all 188 existing tests still pass (no regressions)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create vit.py -- VisionTransformer class and vit_loss_fn** - `ff24b3f` (feat)
2. **Task 2: Create cifar.py and update examples/__init__.py exports** - `07f5e3b` (feat)

**Plan metadata:** committed with docs commit following

## Files Created/Modified

- `src/flops_fit/examples/vit.py` - VisionTransformer(embed_dim=N) class with num_params() and vit_loss_fn
- `src/flops_fit/examples/cifar.py` - CIFAR10Dataset with lazy torchvision import
- `src/flops_fit/examples/__init__.py` - Updated to export VisionTransformer, vit_loss_fn, CIFAR10Dataset alongside GPT exports

## Decisions Made

- **Direct logit output:** VisionTransformer.forward() returns logits directly (not a tuple), making vit_loss_fn simpler than gpt_loss_fn. This structural contrast between ViT and GPT validates that flops_fit's trainer can handle both output patterns.
- **Pre-norm transformer (norm_first=True):** Used nn.TransformerEncoderLayer with norm_first=True (pre-norm) for better gradient flow at scale. This triggers a benign PyTorch UserWarning about enable_nested_tensor being disabled -- expected and harmless.
- **Assertions at init time:** Both incompatible patch_size (image_size % patch_size != 0) and embed_dim/num_heads mismatch raise AssertionError in __init__ to fail fast before any forward pass.
- **Lazy torchvision import:** torchvision.datasets and torchvision.transforms imported only inside _prepare_data() (not at module top), matching the TinyStoriesDataset pattern for zero import-time overhead.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. The PyTorch UserWarning about `enable_nested_tensor is False because encoder_layer.norm_first was True` is expected behavior when using norm_first=True with TransformerEncoder. It is not an error and does not affect correctness.

## User Setup Required

None - no external service configuration required. CIFAR-10 downloads automatically via torchvision on first dataset access (download=True).

## Next Phase Readiness

- VisionTransformer and CIFAR10Dataset ready for use in example scripts (Phase 8 Plan 02)
- Model contract satisfied: num_params() returns positive int, forward() returns (B, 10) tensor
- Loss function contract satisfied: vit_loss_fn(logits, labels) returns scalar
- All 7 examples exports available: GPT, GPTConfig, create_model_for_scaling, TinyStoriesDataset, VisionTransformer, vit_loss_fn, CIFAR10Dataset

---
*Phase: 08-vit-and-cifar-example*
*Completed: 2026-02-17*

## Self-Check: PASSED

- FOUND: src/flops_fit/examples/vit.py
- FOUND: src/flops_fit/examples/cifar.py
- FOUND: src/flops_fit/examples/__init__.py
- FOUND: .planning/phases/08-vit-and-cifar-example/08-01-SUMMARY.md
- FOUND: ff24b3f (Task 1 commit)
- FOUND: 07f5e3b (Task 2 commit)
