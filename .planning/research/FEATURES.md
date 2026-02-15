# Feature Research

**Domain:** Scaling law / compute-optimal ML experiment tooling
**Researched:** 2026-02-15
**Confidence:** MEDIUM (based on domain knowledge from scaling law papers and ML tooling ecosystem; WebSearch unavailable for verification of current open-source landscape)

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist in any scaling law tool worth using. Missing these = users write their own scripts instead.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| IsoFLOP sweep planning | Core method from Chinchilla (Hoffmann et al., 2022). Without this, it is not a scaling law tool. | LOW | Already implemented. Generates (N, D) grid per compute budget. |
| Power law fitting (N_opt, D_opt, L_opt vs C) | The entire point: extract N_opt = k * C^a relationships. Users need fitted coefficients and R-squared. | LOW | Already implemented via scipy least_squares. |
| Compute-optimal prediction | Given a target compute budget, output optimal N, D, and expected loss. This is the deliverable users care about. | LOW | Already implemented. Key value proposition. |
| IsoFLOP curve visualization | U-shaped loss-vs-N curves per compute budget are the canonical scaling law plot. Reviewers and stakeholders expect them. | LOW | Already implemented. Publication-style matplotlib. |
| Scaling law plots (log-log) | N_opt, D_opt, L_opt vs compute on log-log axes with fitted lines. Standard in every scaling law paper. | LOW | Already implemented. |
| Pluggable model architecture | Users have their own architectures (ViTs, SSMs, MoE, custom transformers). A tool locked to one arch is a toy. | HIGH | Not yet implemented. Core of the new milestone. Requires defining a model interface and plugin loading. |
| Pluggable data loading | Different modalities (text, image) need different data pipelines. Users must bring their own DataLoader. | MEDIUM | Not yet implemented. Must support arbitrary PyTorch datasets. |
| Pluggable loss function | Cross-entropy for LM, MSE for regression, custom losses for specific domains. Cannot be hardcoded. | LOW | Not yet implemented but straightforward interface. |
| YAML/config-driven experiment specification | Researchers expect declarative configs. Hydra is already the standard here. Must extend to reference user plugins. | MEDIUM | Hydra already in place. Need to add plugin path resolution. |
| CLI pipeline (plan, train, analyze, visualize) | Researchers run experiments in stages, often on different machines or with restarts. Stage separation is expected. | LOW | Already implemented (ff-plan, ff-train, ff-analyze, ff-visualize). |
| Resume/checkpoint support | Sweeps take days. Power failures, preemptions, and crashes are normal. Must resume without re-running completed experiments. | MEDIUM | Partial: trainer resumes by experiment_id. Need per-experiment checkpointing for interrupted individual runs. |
| Reproducibility (seed control) | Scaling law results must be reproducible for papers. Random seed control across all stages. | LOW | Partially implemented (seed in config). Need deterministic PyTorch settings. |
| Compute budget estimation (FLOPs) | Users need to know how much compute each experiment costs before committing resources. The 6ND approximation is standard. | LOW | Already implemented. May need to be pluggable for non-transformer architectures where 6ND does not apply. |

### Differentiators (Competitive Advantage)

Features that make flops-fit worth using over writing custom scripts. Not required, but these are what convert "interested" into "adopted."

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Pluggable FLOP counting | The 6ND rule only works for dense transformers. MoE, SSMs, CNNs have different FLOP profiles. Let users provide their own `compute_flops(model, batch)` or use torch profiler. | MEDIUM | Critical differentiator for non-transformer use. Without this, the tool only works for vanilla transformers. |
| Chinchilla table output | A formatted table showing optimal N, D, loss for a range of compute budgets (like Table 3 in Hoffmann et al.). Researchers paste this directly into papers. | LOW | High value, low effort. Already have the prediction logic; just need formatted output. |
| Multi-GPU data parallelism | Real scaling law experiments at GPU scale need multi-GPU. Single-GPU limits practical compute range. PyTorch DDP/FSDP handles this. | HIGH | Important for serious users but not needed for MVP. Many users already have their own distributed setup. |
| Built-in example plugins (GPT+TinyStories, ViT+CIFAR) | Working examples dramatically reduce onboarding friction. Users copy and modify rather than building from scratch. | MEDIUM | GPT+TinyStories exists, needs refactoring into plugin format. ViT+CIFAR would prove image support works. |
| Experiment tracking integration (W&B / MLflow) | Researchers already use W&B or MLflow. Logging scaling law metrics there lets them compare with other work. | MEDIUM | Should be optional, not required. Use callbacks/hooks pattern. |
| Confidence intervals on scaling law fits | Power law fits with error bars. Researchers need to report uncertainty. Bootstrap or MCMC on the fitted exponents. | MEDIUM | Significant scientific value. The current R-squared is a start but CIs on the exponents (a in N = k*C^a) are what papers need. |
| Parametric loss surface fitting | Fit the full L(N, D) = A/N^alpha + B/D^beta + E surface (Kaplan et al. and Chinchilla Approach 3). More data-efficient than IsoFLOP-only analysis. | HIGH | This is the "Approach 3" from Chinchilla. More sophisticated than IsoFLOP envelope. Could coexist as an alternative analyzer. |
| Hyperparameter transfer validation | Verify that HP transfer (via u-mup or similar) actually works across the model sizes in the sweep. Plot LR sensitivity at multiple scales. | MEDIUM | Unique to this tool. Other tools assume HPs transfer but do not verify. Catches a major pitfall in scaling law experiments. |
| Sweep cost estimator | Before running, estimate total wall-clock time and compute cost. Show per-experiment breakdown. | LOW | Researchers need to know if a sweep fits their budget before committing. The planner already has the info; just needs a cost model. |
| Automatic outlier detection | Flag experiments where loss is anomalously high (training diverged, bad initialization). Auto-exclude or warn before fitting. | LOW | Prevents garbage-in-garbage-out on the power law fits. Simple statistical test on residuals. |
| Multiple fitting approaches | Support IsoFLOP (Chinchilla Approach 1), parametric (Approach 3), and Kaplan-style fixed-D fitting. Let users compare. | HIGH | Academic differentiator. Most scripts implement one approach. Supporting multiple lets researchers compare methodologies. |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems. Explicitly not building these.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Config-driven model architecture (compose layers from YAML) | "I want to define my model in config, not code" | Model architectures have complex logic (skip connections, custom attention, normalization choices) that YAML cannot express. Creates a bad DSL that limits expressiveness. | Users write a Python class implementing the model interface. Config only controls size parameters (d_model, num_layers, etc.). |
| Automatic architecture search within sweep | "Find the best architecture AND scaling law simultaneously" | Conflates two problems. Architecture search changes the FLOP-per-parameter relationship, breaking the scaling law assumptions. Massive increase in sweep size. | Run scaling law experiments with a fixed architecture. Use NAS separately. |
| Built-in training loop with all optimizers/schedulers | "Support every optimizer and LR schedule" | Combinatorial explosion. Every optimizer has edge cases. Users already have training loops they trust. | Provide a minimal default training loop for the built-in examples. Plugin users bring their own training logic via the trainer interface. |
| Real-time dashboard / web UI | "Show live training curves in a browser" | Significant engineering for marginal value. Researchers already have W&B/TensorBoard. Maintaining a web frontend is a different product. | Integrate with existing experiment trackers via callbacks. |
| Support for all modalities (audio, point clouds, video, etc.) | "Make it universal" | Each modality has unique data loading, preprocessing, and FLOP calculation patterns. Trying to support everything means supporting nothing well. | Support text and image via the plugin system. Other modalities work if users implement the plugin interfaces -- the tool does not need modality-specific code. |
| Automatic hyperparameter tuning across scales | "Find optimal LR for each model size automatically" | This is what u-mup/mup solves. Building a separate HP search per scale point turns a 50-experiment sweep into a 500-experiment sweep. | Use u-mup or similar parameterization for HP transfer. Document how to validate transfer. |
| Cloud job submission (Slurm, Kubernetes, etc.) | "Submit sweep jobs to my cluster" | Cluster configurations are wildly heterogeneous. Building a job scheduler is a separate tool (Submitit, Ray, etc.). | Output sweep configs as JSON. Users feed these into their own job submission system. Provide examples for Slurm/Submitit. |

## Feature Dependencies

```
[Pluggable model interface]
    |
    +--requires--> [Pluggable FLOP counting]
    |                  (non-transformer architectures need custom FLOP formulas)
    |
    +--requires--> [Pluggable data loading]
    |                  (model expects specific input format from data loader)
    |
    +--requires--> [Pluggable loss function]
                       (model output shape determines valid loss functions)

[Built-in example plugins]
    |
    +--requires--> [Pluggable model interface]
    +--requires--> [Pluggable data loading]
    +--requires--> [Pluggable loss function]
                       (examples are the first consumers of plugin API)

[Multi-GPU support]
    |
    +--requires--> [Pluggable model interface]
                       (DDP wrapping needs to work with arbitrary models)

[Experiment tracking integration]
    |
    +--enhances--> [CLI pipeline]
                       (hooks into train/analyze stages)

[Confidence intervals on fits]
    |
    +--requires--> [Power law fitting]
                       (extends existing fitting with bootstrap/MCMC)

[Parametric loss surface fitting]
    |
    +--enhances--> [Power law fitting]
    |                  (alternative analysis approach, shares data format)
    |
    +--conflicts--> nothing (can coexist with IsoFLOP analysis)

[Chinchilla table output]
    |
    +--requires--> [Power law fitting]
                       (needs fitted coefficients to generate table)

[HP transfer validation]
    |
    +--requires--> [Pluggable model interface]
                       (needs to train same arch at multiple widths with shared HPs)
```

### Dependency Notes

- **Plugin interfaces are the critical path:** Model, data, and loss plugins must be designed together since they interact at training time. The model forward pass consumes data loader output and produces values the loss function evaluates.
- **Built-in examples validate the plugin API:** The GPT+TinyStories refactoring is the first real test of whether the plugin interfaces are sufficient. Ship the interfaces and the refactored example together.
- **Multi-GPU is independent of the plugin API design** but depends on it being finalized. DDP wrapping is a standard PyTorch pattern that works with any nn.Module.
- **Analysis features (CIs, parametric fitting, Chinchilla table) are independent of plugins.** They operate on results.json and can be built in parallel.

## MVP Definition

### Launch With (v1 -- Plugin System)

Minimum viable for the "pluggable" milestone. What is needed to say "flops-fit works with any architecture."

- [ ] Model plugin interface (forward, num_params, configure_optimizers) -- without this, the tool is GPT-only
- [ ] Data loader plugin interface (returns PyTorch DataLoader) -- without this, the tool is TinyStories-only
- [ ] Loss plugin interface (callable taking model output and targets) -- without this, cross-entropy is hardcoded
- [ ] YAML config references user Python modules by import path -- the mechanism for loading plugins
- [ ] GPT+TinyStories refactored as built-in example plugin -- validates the API and provides onboarding material
- [ ] Chinchilla table output -- low effort, high value, makes results directly usable in papers

### Add After Validation (v1.x)

Features to add once the plugin system is proven with at least 2 architectures.

- [ ] ViT+CIFAR example plugin -- proves image modality works, not just text
- [ ] Pluggable FLOP counting -- needed once non-transformer users appear (SSMs, MoE)
- [ ] Confidence intervals on scaling law fits -- researchers need this for publications
- [ ] Experiment tracking callbacks (W&B, MLflow) -- add when users request integration
- [ ] Sweep cost estimator -- helps users plan before committing resources
- [ ] Automatic outlier detection in analyzer -- reduces manual data cleaning

### Future Consideration (v2+)

Features to defer until the plugin system is battle-tested.

- [ ] Multi-GPU data parallelism -- complex, users often have their own distributed setup
- [ ] Parametric loss surface fitting (Approach 3) -- significant research engineering
- [ ] Multiple fitting approaches (IsoFLOP vs Kaplan-style) -- academic differentiator, not urgent
- [ ] HP transfer validation tooling -- valuable but requires careful experimental design

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Model plugin interface | HIGH | HIGH | P1 |
| Data loader plugin interface | HIGH | MEDIUM | P1 |
| Loss plugin interface | HIGH | LOW | P1 |
| YAML plugin path resolution | HIGH | MEDIUM | P1 |
| GPT+TinyStories as example plugin | HIGH | MEDIUM | P1 |
| Chinchilla table output | HIGH | LOW | P1 |
| ViT+CIFAR example plugin | MEDIUM | MEDIUM | P2 |
| Pluggable FLOP counting | MEDIUM | MEDIUM | P2 |
| Confidence intervals on fits | MEDIUM | MEDIUM | P2 |
| Experiment tracking callbacks | MEDIUM | MEDIUM | P2 |
| Sweep cost estimator | MEDIUM | LOW | P2 |
| Outlier detection in analyzer | MEDIUM | LOW | P2 |
| Multi-GPU support | MEDIUM | HIGH | P3 |
| Parametric loss surface fitting | LOW | HIGH | P3 |
| Multiple fitting approaches | LOW | HIGH | P3 |
| HP transfer validation | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for the plugin milestone to be considered done
- P2: Should have, add when the plugin system is validated
- P3: Nice to have, future milestones

## Competitor Feature Analysis

| Feature | Custom scripts (most researchers) | Scaling-recipes (Srinivasan) | Cramming (Geiping et al.) | Our Approach |
|---------|----------------------------------|------------------------------|---------------------------|--------------|
| Model architectures | Whatever they code | GPT only | BERT only | Any via plugin interface |
| Data support | Whatever they code | Text only | Text only | Text + image via plugins |
| Sweep planning | Manual or ad-hoc | Manual | N/A (single-model focus) | Automated IsoFLOP grid generation |
| Power law fitting | Custom scipy/numpy | Manual analysis | N/A | Automated with R-squared, predictions |
| Visualization | Custom matplotlib | Custom | Custom | Publication-ready plots built-in |
| HP transfer | mup/u-mup manual setup | u-mup built-in | Standard init | u-mup in built-in example, user choice in plugins |
| Config system | Argparse or custom | Hydra | Hydra | Hydra with plugin path resolution |
| Resume support | Ad-hoc | No | No | Built-in per-experiment resume |
| FLOP counting | Manual 6ND | Manual 6ND | N/A | 6ND default, pluggable for non-transformers |

**Key insight:** There is no general-purpose, architecture-agnostic scaling law tool in the open-source ecosystem. Researchers either write custom scripts per project or use architecture-specific codebases. flops-fit's plugin system fills a genuine gap -- the tool that does the boring parts (sweep planning, curve fitting, visualization) while letting researchers focus on their specific model and data.

## Sources

- Hoffmann et al. (2022) "Training Compute-Optimal Large Language Models" (Chinchilla paper) -- defines IsoFLOP methodology, Approach 1/2/3, and the N_opt/D_opt/L_opt power law relationships
- Kaplan et al. (2020) "Scaling Laws for Neural Language Models" -- earlier scaling law methodology with fixed-D sweeps
- Yang et al. (2022) "Tensor Programs V" -- mup for HP transfer across scales
- Blake et al. (2024) "u-mup: The Unit-Scaled Maximal Update Parametrization" -- u-mup recipe used in existing codebase
- VigneshSrinivasan10/scaling-recipes (GitHub) -- referenced in existing codebase, GPT-specific scaling recipes
- Geiping & Goldstein (2023) "Cramming" -- single-GPU BERT training study, related but different scope
- DeepSeek-AI (2024) scaling law analysis for MoE architectures -- demonstrates need for pluggable FLOP counting
- Existing flops-fit codebase analysis (planner.py, trainer.py, analyzer.py, visualizer.py, model.py)

**Confidence notes:**
- Table stakes features: HIGH confidence -- derived from reading the canonical papers and the existing codebase
- Differentiators: MEDIUM confidence -- based on domain knowledge; could not verify current open-source landscape via WebSearch
- Anti-features: HIGH confidence -- based on experience with ML tooling scope creep
- Competitor analysis: LOW confidence for "scaling-recipes" and "cramming" specifics -- based on training data, could not verify current state

---
*Feature research for: scaling law / compute-optimal ML experiment tooling*
*Researched: 2026-02-15*
