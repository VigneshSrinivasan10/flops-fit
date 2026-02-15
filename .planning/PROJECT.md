# flops-fit

## What This Is

A public, open-source scaling law tool that finds compute-optimal model size (N), data size (D), and predicted loss for any given compute budget. Users plug in their own model architecture, data loader, and loss function — the tool orchestrates the sweep experiments, fits power laws, and produces Chinchilla-style predictions with visualizations. Supports text and image data.

## Core Value

Given a compute budget, tell the user exactly how big their model should be and how much data to train on — for their specific architecture and dataset.

## Requirements

### Validated

- ✓ IsoFLOP sweep planning with configurable compute budgets — existing
- ✓ Training execution with mock/local modes and resume support — existing
- ✓ Power law fitting (N_opt, D_opt, L_opt vs compute) via scipy — existing
- ✓ Scaling law visualization (IsoFLOP curves, scaling law plots) — existing
- ✓ Hydra-based configuration with YAML overrides and presets — existing
- ✓ CLI pipeline (ff-plan → ff-train → ff-analyze → ff-visualize) — existing
- ✓ GPT model with u-mup parameterization for scaling experiments — existing

### Active

- [ ] Plugin architecture: users provide model, data loader, and loss function as Python modules referenced by YAML config
- [ ] Model plugin interface: `forward()` + `num_params()`, config maps size parameters
- [ ] Dataset plugin interface: user-provided data loader for text and image data
- [ ] Loss plugin interface: user-provided loss function referenced by config
- [ ] Multi-GPU training support via data parallelism
- [ ] Existing GPT + TinyStories refactored into a built-in example plugin
- [ ] Full Chinchilla table output: optimal N, D, and loss for a range of compute budgets

### Out of Scope

- Non-text/non-image modalities (point clouds, audio, etc.) — defer to future
- Composable architecture blocks (config-wired layers) — users write Python classes instead
- FLOP estimation in model interface — tool handles this or derives from training
- Distributed training beyond single-node data parallelism — too complex for v1
- API training mode — focus on local training with user-provided components
- Config-driven architecture generation — user writes the model class, config only sets size params

## Context

- Existing codebase is a working 4-stage pipeline (plan → train → analyze → visualize) hardcoded to GPT + TinyStories
- Uses Hydra for configuration, PyTorch for models, scipy for curve fitting, matplotlib for plots
- File-based data handoff between stages via JSON in `outputs/`
- The GPT model uses u-mup parameterization for hyperparameter transfer across scales
- Target hardware: CPU or single/multi-GPU; optimized config exists for AMD Ryzen 7 5825U
- Python 3.11, uv package manager, pytest + ruff for testing/linting

## Constraints

- **Stack**: Python 3.11, PyTorch, Hydra, scipy, matplotlib — existing stack preserved
- **Modalities**: Text and image data only for v1
- **Model interface**: Users provide a Python class with `forward()` and `num_params()`
- **Plugin packaging**: YAML config references Python module paths for model, data, loss

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Plugin via Python modules + YAML config refs | Balances flexibility (any arch) with simplicity (no DSL) | — Pending |
| User provides model + data loader + loss | Tool can't know how to train arbitrary modalities — user knows best | — Pending |
| Multi-GPU via data parallelism | Covers most scaling law experiments without distributed complexity | — Pending |
| Existing GPT becomes example, not default | Clean separation — no privileged built-in architecture | — Pending |
| Text + image modalities for v1 | Covers most ML research; other modalities deferred | — Pending |

---
*Last updated: 2026-02-15 after initialization*
