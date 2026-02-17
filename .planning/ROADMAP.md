# Roadmap: flops-fit

## Overview

Transform flops-fit from a hardcoded CLI pipeline (GPT + TinyStories) into a Python library where users pass their own model class, dataset, and loss function to get Chinchilla-style scaling law predictions. The journey builds interfaces first (how users interact with the library), then the engines that execute experiments, then proves generality with two example architectures across text and image modalities.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Library Skeleton and Model Interface** - Define package structure, `find_optimal()` signature, and model contract
- [x] **Phase 2: Dataset and Loss Interfaces** - Complete the input interfaces for user-provided data and loss
- [ ] **Phase 3: Sweep Planning** - Generate IsoFLOP experiment grids from compute budgets and model interface
- [ ] **Phase 4: Training Engine** - GPU-aware training loop consuming the library interfaces
- [x] **Phase 5: Analysis and Fitting** - Power law fitting with outlier detection and Chinchilla table output
- [ ] **Phase 6: Results Object and API Integration** - Wire `find_optimal()` end-to-end with Result object
- [ ] **Phase 7: GPT + TinyStories Example** - Refactor existing GPT as library example with CLI wrapper
- [ ] **Phase 8: ViT + CIFAR Example** - Second modality proving architecture-agnostic design
- [ ] **Phase 9: Multi-GPU Data Parallelism** - Accelerate-based multi-GPU support

## Phase Details

### Phase 1: Library Skeleton and Model Interface
**Goal**: Users can import `flops_fit` and define a model class that the library knows how to scale
**Depends on**: Nothing (first phase)
**Requirements**: API-01, API-02
**Success Criteria** (what must be TRUE):
  1. User can `import flops_fit` and see `flops_fit.find_optimal` exists as a callable (stub is fine)
  2. User can pass a model class + size parameter name + kwargs, and the library creates model instances at different sizes by varying the size parameter
  3. Library validates that the model class exposes `num_params()` and raises a clear error if not
  4. The package is installable via `pip install -e .` with the new library structure
**Plans:** 1 plan

Plans:
- [x] 01-01-PLAN.md — Model factory, API stub, contract validation, and tests

### Phase 2: Dataset and Loss Interfaces
**Goal**: Users can pass their own dataset and loss function as Python objects
**Depends on**: Phase 1
**Requirements**: API-03, API-04
**Success Criteria** (what must be TRUE):
  1. User can pass a PyTorch Dataset or DataLoader and the library handles batching and iteration
  2. User can pass any callable as a loss function and the library uses it during training
  3. Library validates dataset and loss interfaces at call time with clear error messages (not deep in a training loop)
**Plans:** 2 plans

Plans:
- [x] 02-01-PLAN.md — TDD: dataset validation/wrapping (data.py) and loss validation (loss.py)
- [x] 02-02-PLAN.md — Wire validation into find_optimal() with integration tests

### Phase 3: Sweep Planning
**Goal**: Users can see what experiments will run and estimate compute cost before committing GPU hours
**Depends on**: Phase 1 (needs model interface to size models)
**Requirements**: SWEEP-01, SWEEP-02
**Success Criteria** (what must be TRUE):
  1. Given compute budgets and a model interface, the library generates an IsoFLOP experiment grid with model sizes and token counts
  2. User can call a cost estimation method and see total estimated FLOPs across all planned experiments
  3. Sweep plan is inspectable as a data structure (not just printed output)
**Plans**: 2 plans

Plans:
- [ ] 03-01-PLAN.md — TDD: sweep.py with Experiment/SweepPlan dataclasses and plan_sweep()
- [ ] 03-02-PLAN.md — Wire plan_sweep() into find_optimal() and export SweepPlan

### Phase 4: Training Engine
**Goal**: The library can train models on GPU with automatic device placement and resume interrupted sweeps
**Depends on**: Phase 1 (model interface), Phase 2 (dataset/loss interfaces), Phase 3 (sweep plan)
**Requirements**: TRAIN-01, TRAIN-03
**Success Criteria** (what must be TRUE):
  1. Library trains a model on available GPU with automatic device placement (falls back to CPU if no GPU)
  2. Training loop uses the user-provided dataset and loss function through the library interfaces
  3. An interrupted sweep can be resumed without re-running completed experiments
  4. Training results (final loss, actual FLOPs, wall time) are captured per experiment
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD
- [ ] 04-03: TBD

### Phase 5: Analysis and Fitting
**Goal**: The library fits power laws to training results and produces Chinchilla-style predictions
**Depends on**: Phase 4 (needs training results)
**Requirements**: ANLZ-01, ANLZ-02, ANLZ-03
**Success Criteria** (what must be TRUE):
  1. Library fits N_opt, D_opt, and L_opt vs compute power laws with R-squared values
  2. Outlier experiments are automatically detected and excluded before fitting
  3. Chinchilla table output shows optimal N, D, and predicted loss for a range of compute budgets
  4. Fitting uses linear-space nonlinear least squares (not log-space) with irreducible loss term
**Plans:** 3 plans

Plans:
- [ ] 05-01-PLAN.md — TDD: linear-space NLS fitting with l_inf and IQR outlier detection in analyzer.py
- [ ] 05-02-PLAN.md — TDD: chinchilla_table() on ScalingAnalysis and l_inf-aware loss prediction
- [ ] 05-03-PLAN.md — Gap closure: fix ScalingLawAnalyzer.predict() to include l_inf in expected_loss

### Phase 6: Results Object and API Integration
**Goal**: `flops_fit.find_optimal()` works end-to-end and returns a Result object with table, plot, and predict methods
**Depends on**: Phase 3 (sweep planning), Phase 4 (training), Phase 5 (analysis)
**Requirements**: API-05, API-06, API-07
**Success Criteria** (what must be TRUE):
  1. `result.chinchilla_table()` returns a table of optimal N, D, and loss for each compute budget
  2. `result.plot()` produces scaling law and IsoFLOP visualizations (matplotlib figures)
  3. `result.predict(compute_budget)` returns optimal N, D, and expected loss for a specific budget
  4. `flops_fit.find_optimal()` orchestrates the full pipeline (plan, train, analyze) and returns the Result object
**Plans:** 2 plans

Plans:
- [ ] 06-01-PLAN.md — TDD: Result dataclass (result.py) with chinchilla_table, predict, plot methods
- [ ] 06-02-PLAN.md — Update find_optimal() to return Result after training; update test_api.py

### Phase 7: GPT + TinyStories Example
**Goal**: Existing GPT code works as a library example demonstrating the full scaling law workflow
**Depends on**: Phase 6 (needs working end-to-end API)
**Requirements**: EX-01, EX-03
**Success Criteria** (what must be TRUE):
  1. A self-contained example script shows how to use `flops_fit.find_optimal()` with GPT + TinyStories
  2. A CLI wrapper example shows how to expose the library via command-line arguments
  3. The existing GPT model is importable from `flops_fit.examples.gpt` (or similar) without being part of the core library
  4. Running the example end-to-end produces scaling law predictions (in mock or CPU mode)
**Plans**: TBD

Plans:
- [ ] 07-01: TBD
- [ ] 07-02: TBD

### Phase 8: ViT + CIFAR Example
**Goal**: A second architecture and modality proves the library is truly architecture-agnostic
**Depends on**: Phase 6 (needs working end-to-end API)
**Requirements**: EX-02
**Success Criteria** (what must be TRUE):
  1. A ViT + CIFAR example script uses `flops_fit.find_optimal()` to produce scaling law predictions for image classification
  2. The ViT model conforms to the same model interface as GPT (model class + size param + num_params)
  3. The library handles image data (pixel tensors) without any text-specific assumptions in the core
**Plans**: TBD

Plans:
- [ ] 08-01: TBD
- [ ] 08-02: TBD

### Phase 9: Multi-GPU Data Parallelism
**Goal**: Users with multiple GPUs can run sweeps faster via data parallelism
**Depends on**: Phase 4 (training engine must be stable first)
**Requirements**: TRAIN-02
**Success Criteria** (what must be TRUE):
  1. Library supports multi-GPU data parallelism via HuggingFace Accelerate
  2. Multi-GPU training produces the same loss values as single-GPU (within numerical tolerance)
  3. User does not need to modify their model class or dataset to use multi-GPU
**Plans**: TBD

Plans:
- [ ] 09-01: TBD
- [ ] 09-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Library Skeleton and Model Interface | 1/1 | Complete | 2026-02-16 |
| 2. Dataset and Loss Interfaces | 2/2 | Complete | 2026-02-16 |
| 3. Sweep Planning | 0/TBD | Not started | - |
| 4. Training Engine | 0/TBD | Not started | - |
| 5. Analysis and Fitting | 3/3 | Complete | 2026-02-17 |
| 6. Results Object and API Integration | 0/2 | Not started | - |
| 7. GPT + TinyStories Example | 0/TBD | Not started | - |
| 8. ViT + CIFAR Example | 0/TBD | Not started | - |
| 9. Multi-GPU Data Parallelism | 0/TBD | Not started | - |
