# Roadmap: flops-fit

## Overview

Transform flops-fit from a hardcoded GPT + TinyStories scaling law pipeline into a pluggable framework where users bring their own model architecture, data loader, and loss function. The journey starts by locking down the existing pipeline with tests, then introduces plugin protocols, refactors GPT as the first plugin, builds a real training engine, improves the analyzer, proves generality with image support, adds multi-GPU training, and finishes with sweep cost estimation and end-to-end polish.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Existing Pipeline Baseline** - Lock down current functionality with tests before refactoring
- [ ] **Phase 2: Plugin Protocols and Loading** - Define and implement the plugin interface contracts
- [ ] **Phase 3: GPT Plugin Refactor** - Refactor existing GPT + TinyStories into the first plugin
- [ ] **Phase 4: Training Engine** - Build a real GPU-aware training loop with Accelerate
- [ ] **Phase 5: Analysis Improvements** - Fix fitting methodology and add Chinchilla table output
- [ ] **Phase 6: Image Modality** - Prove plugin generality with ViT + CIFAR
- [ ] **Phase 7: Multi-GPU Training** - Add data parallelism via Accelerate DDP
- [ ] **Phase 8: Sweep Cost Estimation and Polish** - Cost prediction and end-to-end validation

## Phase Details

### Phase 1: Existing Pipeline Baseline
**Goal**: The existing 4-stage pipeline (plan, train, analyze, visualize) is fully characterized with tests so refactoring cannot silently break it
**Depends on**: Nothing (first phase)
**Requirements**: EXIST-01, EXIST-02, EXIST-03, EXIST-04, EXIST-05, EXIST-06, EXIST-07
**Success Criteria** (what must be TRUE):
  1. User can run `ff-plan` and get a valid sweep plan JSON with configurable compute budgets
  2. User can run `ff-train` with mock mode and resume a partially completed sweep by experiment ID
  3. User can run `ff-analyze` on training results and get power law fits with R-squared values
  4. User can run `ff-visualize` and get IsoFLOP curve and scaling law plots saved to disk
  5. All existing CLI commands work with Hydra YAML overrides and presets
**Plans**: TBD

Plans:
- [ ] 01-01: TBD
- [ ] 01-02: TBD

### Phase 2: Plugin Protocols and Loading
**Goal**: Users can define model, data, and loss plugins as Python classes and have them loaded and validated via YAML config
**Depends on**: Phase 1
**Requirements**: PLUG-01, PLUG-02, PLUG-03, PLUG-04, PLUG-06
**Success Criteria** (what must be TRUE):
  1. User can write a model class with `forward()` and `num_params()` and reference it by import path in YAML config
  2. User can write a data loader module returning a PyTorch DataLoader and reference it by import path in YAML config
  3. User can write a loss callable and reference it by import path in YAML config
  4. Plugin protocol violations (missing methods, wrong signatures) are caught at load time with clear error messages, not hours into a sweep
**Plans**: TBD

Plans:
- [ ] 02-01: TBD
- [ ] 02-02: TBD

### Phase 3: GPT Plugin Refactor
**Goal**: The existing GPT + TinyStories code works as a plugin through the new protocol system, proving the plugin architecture end-to-end
**Depends on**: Phase 2
**Requirements**: PLUG-05
**Success Criteria** (what must be TRUE):
  1. User can run the full pipeline (plan, train, analyze, visualize) using `model=gpt dataset=tinystories` config overrides instead of hardcoded imports
  2. The GPT plugin lives in a `plugins/` directory and follows the same interface any external plugin would use
  3. Pipeline results (scaling law fits, plots) are identical to pre-refactor baseline within numerical tolerance
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD

### Phase 4: Training Engine
**Goal**: Users can train models on GPU with proper device management, mixed precision, and checkpoint recovery
**Depends on**: Phase 3
**Requirements**: (no new requirements -- infrastructure enabling TRAIN-01, IMG-01)
**Success Criteria** (what must be TRUE):
  1. User can run sweep experiments on GPU and the training loop automatically handles device placement
  2. User can resume a crashed sweep from the last checkpoint without re-running completed experiments
  3. Training uses mixed precision (fp16/bf16) when available, with automatic fallback to fp32 on CPU
  4. The GPT plugin pipeline produces correct scaling law fits when trained on real GPU (not just mock mode)
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD

### Phase 5: Analysis Improvements
**Goal**: Users get trustworthy scaling predictions with Chinchilla-style output tables and automatic outlier filtering
**Depends on**: Phase 1 (operates on results JSON, independent of plugin system)
**Requirements**: ANLZ-01, ANLZ-02
**Success Criteria** (what must be TRUE):
  1. User can generate a Chinchilla table showing optimal model size (N), data size (D), and predicted loss for a specified range of compute budgets
  2. Analyzer automatically detects and flags experiments with diverged training or anomalous loss before fitting power laws
  3. Flagged outliers are excluded from fits by default, with user override to include them
**Plans**: TBD

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD

### Phase 6: Image Modality
**Goal**: Users can run scaling law experiments on image datasets, proving the plugin system is modality-agnostic
**Depends on**: Phase 4 (needs training engine), Phase 2 (needs plugin system)
**Requirements**: IMG-01, IMG-02, IMG-03
**Success Criteria** (what must be TRUE):
  1. User can run the full pipeline with `model=vit dataset=cifar` and get scaling law fits for image classification
  2. The ViT + CIFAR example plugin is included as a built-in alongside GPT + TinyStories
  3. The training loop handles image batches without any modality-specific branching (dict-based batch convention works for both text and images)
  4. Scaling law plots for image experiments are publication-ready (same quality as text experiments)
**Plans**: TBD

Plans:
- [ ] 06-01: TBD
- [ ] 06-02: TBD

### Phase 7: Multi-GPU Training
**Goal**: Users can run sweep experiments across multiple GPUs for faster scaling law experiments
**Depends on**: Phase 4 (needs training engine with Accelerate)
**Requirements**: TRAIN-01
**Success Criteria** (what must be TRUE):
  1. User can launch a sweep with `accelerate launch` and it distributes data across available GPUs
  2. Multi-GPU training produces identical scaling law fits as single-GPU training (within numerical tolerance)
  3. Data loading is properly distributed so each GPU processes different batches without duplication
**Plans**: TBD

Plans:
- [ ] 07-01: TBD
- [ ] 07-02: TBD

### Phase 8: Sweep Cost Estimation and Polish
**Goal**: Users can estimate time and compute cost before committing to a full sweep, and the entire framework works end-to-end with confidence
**Depends on**: Phase 7
**Requirements**: TRAIN-02
**Success Criteria** (what must be TRUE):
  1. User can run a cost estimation command that reports estimated wall-clock time and compute (FLOPs) for a planned sweep before training starts
  2. Cost estimates are based on a small calibration run (not just theoretical FLOP counts)
  3. The full pipeline works end-to-end for both text and image modalities with all features (plugins, GPU training, multi-GPU, analysis, visualization)
**Plans**: TBD

Plans:
- [ ] 08-01: TBD
- [ ] 08-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8

Note: Phase 5 (Analysis) depends only on Phase 1 and can be executed in parallel with Phases 2-4 if desired.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Existing Pipeline Baseline | 0/TBD | Not started | - |
| 2. Plugin Protocols and Loading | 0/TBD | Not started | - |
| 3. GPT Plugin Refactor | 0/TBD | Not started | - |
| 4. Training Engine | 0/TBD | Not started | - |
| 5. Analysis Improvements | 0/TBD | Not started | - |
| 6. Image Modality | 0/TBD | Not started | - |
| 7. Multi-GPU Training | 0/TBD | Not started | - |
| 8. Sweep Cost Estimation and Polish | 0/TBD | Not started | - |
