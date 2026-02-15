# Requirements: flops-fit

**Defined:** 2026-02-15
**Core Value:** Given a compute budget, tell the user exactly how big their model should be and how much data to train on — for their specific architecture and dataset.

## v1 Requirements

### Plugin System

- [ ] **PLUG-01**: User can provide a model architecture as a Python class with `forward()` and `num_params()` methods
- [ ] **PLUG-02**: User can provide a data loader as a Python module that returns a PyTorch DataLoader
- [ ] **PLUG-03**: User can provide a loss function as a Python callable taking model output and targets
- [ ] **PLUG-04**: User can reference model, data, and loss plugins via import paths in YAML config (Hydra `_target_`)
- [ ] **PLUG-05**: Existing GPT + TinyStories code is refactored into a built-in example plugin
- [ ] **PLUG-06**: Plugin contracts are enforced at load time via Protocol validation (fail fast, not hours into a sweep)

### Image Support

- [ ] **IMG-01**: User can run scaling law experiments on image classification datasets
- [ ] **IMG-02**: A built-in ViT + CIFAR example plugin demonstrates image modality support
- [ ] **IMG-03**: Training loop handles image data without modality-specific branching (dict-based batch convention)

### Analysis & Output

- [ ] **ANLZ-01**: User can generate a Chinchilla-style table showing optimal N, D, and loss for a range of compute budgets
- [ ] **ANLZ-02**: Analyzer automatically detects and flags outlier experiments (diverged training, anomalous loss) before fitting

### Training

- [ ] **TRAIN-01**: User can run sweep experiments across multiple GPUs via data parallelism (Accelerate)
- [ ] **TRAIN-02**: User can estimate total wall-clock time and compute cost before running a sweep

### Existing (Carried Over)

- [ ] **EXIST-01**: IsoFLOP sweep planning with configurable compute budgets
- [ ] **EXIST-02**: Power law fitting (N_opt, D_opt, L_opt vs compute) with R-squared
- [ ] **EXIST-03**: Compute-optimal prediction for target compute budgets
- [ ] **EXIST-04**: IsoFLOP curve and scaling law visualization (publication-ready)
- [ ] **EXIST-05**: CLI pipeline (ff-plan, ff-train, ff-analyze, ff-visualize)
- [ ] **EXIST-06**: Hydra configuration with YAML overrides and presets
- [ ] **EXIST-07**: Training resume by experiment ID

## v2 Requirements

### Analysis Enhancements

- **ANLZ-03**: Confidence intervals on fitted scaling law exponents (bootstrap/MCMC)
- **ANLZ-04**: Parametric loss surface fitting (Chinchilla Approach 3: L(N,D) = A/N^a + B/D^b + E)
- **ANLZ-05**: Multiple fitting approaches (IsoFLOP vs Kaplan-style fixed-D)

### Plugin Enhancements

- **PLUG-07**: Pluggable FLOP counting for non-transformer architectures (MoE, SSMs, CNNs)
- **PLUG-08**: Experiment tracking callbacks (W&B, MLflow integration)

### Training Enhancements

- **TRAIN-03**: Full per-experiment checkpointing for interrupted individual runs
- **TRAIN-04**: Deterministic training with `torch.use_deterministic_algorithms`
- **TRAIN-05**: HP transfer validation tooling (LR sensitivity across scales)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Config-driven model architecture (compose layers from YAML) | Model architectures have complex logic YAML cannot express; users write Python classes |
| Automatic architecture search within sweep | Conflates two problems; changes FLOP-per-parameter relationship, breaking scaling law assumptions |
| Built-in training loop with all optimizers/schedulers | Combinatorial explosion; users bring their own optimizer via model's configure_optimizers |
| Real-time dashboard / web UI | Researchers already have W&B/TensorBoard; maintaining a frontend is a different product |
| Non-text/non-image modalities | Each modality has unique patterns; support text and image, others work via plugin interface |
| Cloud job submission (Slurm, K8s) | Cluster configs are heterogeneous; output sweep configs as JSON for users' own job systems |
| Auto HP tuning across scales | This is what u-mup solves; building separate HP search per scale turns 50 experiments into 500 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PLUG-01 | — | Pending |
| PLUG-02 | — | Pending |
| PLUG-03 | — | Pending |
| PLUG-04 | — | Pending |
| PLUG-05 | — | Pending |
| PLUG-06 | — | Pending |
| IMG-01 | — | Pending |
| IMG-02 | — | Pending |
| IMG-03 | — | Pending |
| ANLZ-01 | — | Pending |
| ANLZ-02 | — | Pending |
| TRAIN-01 | — | Pending |
| TRAIN-02 | — | Pending |
| EXIST-01 | — | Pending |
| EXIST-02 | — | Pending |
| EXIST-03 | — | Pending |
| EXIST-04 | — | Pending |
| EXIST-05 | — | Pending |
| EXIST-06 | — | Pending |
| EXIST-07 | — | Pending |

**Coverage:**
- v1 requirements: 20 total
- Mapped to phases: 0
- Unmapped: 20

---
*Requirements defined: 2026-02-15*
*Last updated: 2026-02-15 after initial definition*
