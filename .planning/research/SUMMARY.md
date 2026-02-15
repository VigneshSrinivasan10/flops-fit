# Project Research Summary

**Project:** flops-fit
**Domain:** Pluggable scaling law / compute-optimal ML experiment framework
**Researched:** 2026-02-15
**Confidence:** MEDIUM

## Executive Summary

flops-fit is an ML research tool that automates Chinchilla-style scaling law experiments: given a compute budget, it determines the optimal model size, data size, and expected loss. The existing codebase is a working 4-stage pipeline (plan, train, analyze, visualize) hardcoded to GPT + TinyStories. The next milestone transforms it into a pluggable framework where users bring their own model architecture, data loader, and loss function. Experts build this kind of framework using protocol-based plugin interfaces loaded via configuration (Hydra `_target_` instantiation), a modality-agnostic training loop that communicates via `Dict[str, Tensor]` batch conventions, and HuggingFace Accelerate for GPU/multi-GPU support.

The recommended approach is an incremental refactoring, not a rewrite. The pipeline's file-based stage handoff and Hydra configuration are the right foundations. The core work is: (1) define `ScalableModel`, `ScalableDataset`, and `ScalableLoss` protocols, (2) refactor the existing GPT + TinyStories into the first plugin implementation, (3) build a real GPU-aware training engine using Accelerate, and (4) prove the plugin system works with a second modality (ViT + CIFAR). No new dependencies are needed for the plugin system itself -- Hydra's `instantiate()` with `_target_` YAML keys provides plugin loading for free.

The primary risks are: plugin interface violations caught only at runtime (wasting GPU hours on bad configs), power law fitting methodology errors that produce plausible-looking but wrong scaling predictions, and DDP gradient synchronization bugs that silently corrupt results. Mitigation requires early validation at plugin load time (not training time), fixing the analyzer's log-space fitting bias before making it pluggable, and building single-GPU vs multi-GPU comparison tests as a gate before production sweeps.

## Key Findings

### Recommended Stack

The existing stack (Python 3.11+, PyTorch, Hydra, scipy, matplotlib, HuggingFace datasets/transformers) is solid and should be preserved. Two additions are recommended, and one version floor should be raised.

**Core technologies:**
- **HuggingFace Accelerate** (>=1.0.0): Multi-GPU training -- wraps DDP/FSDP with minimal code invasion; plugin authors write normal single-GPU PyTorch code, Accelerate handles device placement, gradient sync, and mixed precision automatically
- **torchvision** (>=0.16.0): Image data support -- standard transforms and datasets for the image modality plugins
- **PyTorch** (raise floor to >=2.1.0): Stabilized `torch.compile()` and `scaled_dot_product_attention`, which the codebase already uses
- **Hydra `instantiate()`** (already included): Plugin loading -- `_target_` YAML keys resolve to Python classes with zero additional dependencies

**Explicitly rejected:** PyTorch Lightning (conflicts with plugin interface), custom plugin registries/decorators (Hydra already solves this), wandb/mlflow as core dependencies (should be optional).

**Version concern:** The Accelerate >=1.0.0 pin has LOW confidence -- exact current version could not be verified. Validate before committing to pyproject.toml.

### Expected Features

**Must have (table stakes -- P1 for plugin milestone):**
- Model plugin interface (`forward`, `num_params`, `configure_optimizers`)
- Dataset plugin interface (returns PyTorch DataLoader)
- Loss plugin interface (callable taking model output and targets)
- YAML config references user Python modules via `_target_` import paths
- GPT + TinyStories refactored as built-in example plugin (validates the API)
- Chinchilla table output (optimal N, D, loss for compute budget range)

**Should have (differentiators -- P2):**
- ViT + CIFAR example plugin (proves image modality works)
- Pluggable FLOP counting (needed for non-transformer architectures)
- Confidence intervals on scaling law fits (researchers need this for papers)
- Experiment tracking callbacks (W&B, MLflow -- optional integration)
- Automatic outlier detection in analyzer
- Sweep cost estimator

**Defer (v2+):**
- Multi-GPU data parallelism (complex, many users have their own setup)
- Parametric loss surface fitting (Chinchilla Approach 3)
- Multiple fitting approaches (IsoFLOP vs Kaplan-style)
- HP transfer validation tooling

**Key insight from competitor analysis:** There is no general-purpose, architecture-agnostic scaling law tool in the open-source ecosystem. flops-fit's plugin system fills a genuine gap.

### Architecture Approach

The architecture follows an incremental evolution: extract plugin protocols, wrap existing code as the first plugin, build a training engine with device abstraction, and use Hydra config groups for plugin selection. The training loop is modality-agnostic by using a `Dict[str, Tensor]` batch convention -- text produces `{"input_ids", "labels"}`, images produce `{"pixel_values", "labels"}`, and the engine just does `model(batch)` regardless.

**Major components:**
1. **Plugin Protocols** (`protocols.py`) -- `ScalableModel`, `ScalableDataset`, `ScalableLoss` using Python `typing.Protocol` with `runtime_checkable` for load-time validation
2. **Plugin Registry** (`registry.py`) -- Thin wrapper around `hydra.utils.instantiate` with protocol conformance checks at registration time
3. **Training Engine** (`engine/`) -- Core training loop (`trainer.py`), device management (`device.py`), checkpoint support (`checkpoint.py`); uses Accelerate for GPU/DDP abstraction
4. **Built-in Plugins** (`plugins/`) -- GPT + TinyStories (refactored from existing code), ViT + CIFAR (new), with corresponding loss functions
5. **Hydra Config Groups** (`conf/model/`, `conf/dataset/`, `conf/loss/`) -- Users select plugins via CLI: `model=gpt`, `model=vit`

**Critical build order:** Protocols first, then GPT refactored as plugin, then training engine, then registry wiring. Test the full pipeline end-to-end with GPT before adding new plugins.

### Critical Pitfalls

1. **Plugin interface violations caught at runtime, not load time** -- A plugin missing a method or returning wrong types fails hours into a sweep. Prevent by: validate all protocols at plugin registration time with dummy-input smoke tests; build a plugin conformance test suite.

2. **Log-space power law fitting biases toward large-scale points** -- The current analyzer fits in log-space, which is wrong for loss prediction (assumes multiplicative noise). Prevent by: fit `L(C) = A*C^alpha + L_inf` in linear space with nonlinear least squares; always include the irreducible loss term.

3. **Insufficient IsoFLOP sweep coverage produces spurious power laws** -- With only 5 compute budgets and 7 model sizes, sparse grids produce plausible-looking but wrong exponents. Prevent by: enforce minimum 7 budgets and 10 model sizes; flag when optimum is at sweep boundary.

4. **Noisy small-scale experiments corrupt fits** -- Small models have high initialization variance that pulls the fit away from the true relationship. Prevent by: require multiple seeds (3+), use median loss, weight fits by inverse variance.

5. **DDP gradient desync is silent** -- Unused parameters, inconsistent forward passes, or non-deterministic ops produce wrong gradients without errors. Prevent by: automated single-GPU vs multi-GPU loss comparison test; `find_unused_parameters=True` during development.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Plugin Architecture and Protocol System
**Rationale:** Everything else depends on the plugin interfaces being defined and validated. The architecture research identifies protocols as the critical path with zero external dependencies.
**Delivers:** `ScalableModel`, `ScalableDataset`, `ScalableLoss` protocols; plugin registry via Hydra instantiate; GPT + TinyStories refactored as first plugin; full pipeline working end-to-end with plugins.
**Addresses:** Model/data/loss plugin interfaces, YAML plugin path resolution, GPT as example plugin (P1 features).
**Avoids:** Plugin interface violations (Pitfall 4) -- build validation into registry from day one.

### Phase 2: Training Engine and GPU Support
**Rationale:** The current training loop is a stub. A real training engine is needed before any serious experiments. Accelerate integration is well-documented and the architecture research provides a clear DeviceManager pattern.
**Delivers:** Real GPU-aware training loop, device management (CPU/single-GPU), checkpoint support for crash recovery, mixed precision support.
**Uses:** HuggingFace Accelerate for device abstraction and mixed precision.
**Implements:** Training Engine component (`engine/trainer.py`, `engine/device.py`, `engine/checkpoint.py`).
**Avoids:** Noisy small-scale experiments (Pitfall 5) -- add multi-seed support to trainer.

### Phase 3: Analyzer Improvements and Chinchilla Output
**Rationale:** The fitting methodology has known bugs (log-space bias) that must be fixed before results are trustworthy. This is independent of the plugin system and can be parallelized with Phase 2.
**Delivers:** Corrected power law fitting (linear-space for loss), Chinchilla table output, outlier detection, confidence intervals on fitted exponents.
**Addresses:** Chinchilla table output (P1), confidence intervals (P2), outlier detection (P2).
**Avoids:** Log-space fitting bias (Pitfall 2), noisy small-scale corruption (Pitfall 5).

### Phase 4: Image Modality Support
**Rationale:** Proving the plugin system works with a second modality validates the architecture. Requires the plugin system (Phase 1) and training engine (Phase 2) to be working.
**Delivers:** ViT + CIFAR example plugin, torchvision integration, pluggable FLOP counting for non-transformer architectures.
**Addresses:** ViT + CIFAR example (P2), pluggable FLOP counting (P2).
**Avoids:** Cross-modality scaling law assumptions (Pitfall 6) -- verify vision fits use appropriate functional forms.

### Phase 5: Multi-GPU Data Parallelism
**Rationale:** Deferred because most scaling law experiments use single-GPU, and many researchers have their own distributed setup. Requires the training engine (Phase 2) to be stable.
**Delivers:** DDP support via Accelerate, `accelerate launch` integration, distributed data loading.
**Avoids:** DDP gradient desync (Pitfall 3) -- build single-GPU vs multi-GPU comparison test as a gate.

### Phase Ordering Rationale

- **Phases 1-2 are sequential:** The training engine consumes plugins, so protocols must be defined first. However, Phase 2 can start as soon as the protocol signatures are finalized (before all plugins are complete).
- **Phase 3 is parallelizable:** Analyzer improvements operate on results.json and have no dependency on the plugin system. Can be developed alongside Phases 1-2.
- **Phase 4 validates Phase 1:** The ViT plugin is the real test of whether the plugin architecture is general enough. If the interfaces need changes, it is better to discover this before external users adopt them.
- **Phase 5 is last by design:** Multi-GPU adds complexity and has the highest-cost pitfall (silent gradient desync). Everything else should be stable before tackling it.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2 (Training Engine):** Accelerate integration patterns, checkpoint format decisions, and the Hydra + torchrun launch interaction (Hydra's `@hydra.main` conflicts with `torchrun` -- must use Compose API instead).
- **Phase 4 (Image Modality):** Vision scaling law functional forms differ from text; need to research whether the analyzer should support pluggable fitting functions.
- **Phase 5 (Multi-GPU):** DDP + Hydra integration is a known gotcha; HuggingFace Datasets + DDP cache coordination needs attention.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Plugin Architecture):** Protocol-based plugin patterns and Hydra instantiate are well-documented; the architecture research provides concrete code patterns.
- **Phase 3 (Analyzer Improvements):** Power law fitting methodology is well-established in the scaling law literature; scipy `curve_fit` with proper initial guesses is standard.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM-HIGH | Existing stack verified from codebase; Accelerate confirmed via docs fetch; torchvision/timm versions need verification |
| Features | MEDIUM | Table stakes derived from canonical papers (HIGH); competitor analysis based on training data (LOW -- could not verify current landscape) |
| Architecture | MEDIUM-HIGH | Protocol pattern and Hydra instantiate are well-established; build order based on sound dependency analysis; DeviceManager pattern is standard |
| Pitfalls | MEDIUM-HIGH | Scaling law methodology pitfalls well-documented in literature; DDP pitfalls are stable PyTorch knowledge; plugin validation is general software engineering |

**Overall confidence:** MEDIUM

### Gaps to Address

- **Accelerate version pin:** Could not verify exact current version via PyPI. Validate `>=1.0.0` is correct before updating pyproject.toml.
- **torchvision/timm versions:** Version recommendations from training data. Verify latest compatible versions.
- **Vision scaling law forms:** Whether the analyzer needs pluggable fitting functions is not fully resolved. Research during Phase 4 planning.
- **Hydra + torchrun interaction:** The Compose API workaround for multi-GPU is documented in pitfalls but not validated. Research during Phase 5 planning.
- **Current open-source landscape:** Could not search for competing tools. The "no general-purpose tool exists" claim has MEDIUM confidence and should be spot-checked.
- **Results storage at scale:** Flat JSON may not scale beyond ~200 experiments. Decide during Phase 2 whether to migrate to JSONL or SQLite.

## Sources

### Primary (HIGH confidence)
- HuggingFace Accelerate official docs (https://huggingface.co/docs/accelerate/index) -- fetched 2026-02-15; confirmed DDP/FSDP wrapping, 4-line integration, `accelerate launch` CLI
- Existing flops-fit codebase analysis -- direct inspection of pyproject.toml, model.py, trainer.py, planner.py, analyzer.py, config files
- Hoffmann et al. (2022) "Training Compute-Optimal Large Language Models" (Chinchilla) -- IsoFLOP methodology, power law forms, sweep design
- Kaplan et al. (2020) "Scaling Laws for Neural Language Models" -- earlier methodology, fixed-D sweeps
- Python `typing.Protocol` -- core language feature since 3.8

### Secondary (MEDIUM confidence)
- Hydra `instantiate` with `_target_` pattern -- well-established but not verified against latest Hydra version
- PyTorch DDP/FSDP patterns -- stable API, specifics not verified against PyTorch 2.x docs
- Yang et al. (2022) "Tensor Programs V" (mup) and Blake et al. (2024) "u-mup" -- HP transfer across scales
- DeepSeek-AI (2024) MoE scaling analysis -- demonstrates need for pluggable FLOP counting

### Tertiary (LOW confidence)
- torchvision >=0.16.0, timm >=1.0.0 version pins -- training data knowledge, verify before use
- Accelerate >=1.0.0 version pin -- could not verify via PyPI
- Competitor feature analysis (scaling-recipes, cramming) -- from training data, current state unknown

---
*Research completed: 2026-02-15*
*Ready for roadmap: yes*
