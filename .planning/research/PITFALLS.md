# Pitfalls Research

**Domain:** Scaling law / compute-optimal ML tooling (pluggable, multi-modal)
**Researched:** 2026-02-15
**Confidence:** MEDIUM (training data only -- WebSearch/WebFetch unavailable; topics are well-established but verify versions)

---

## Critical Pitfalls

### Pitfall 1: Insufficient IsoFLOP Sweep Coverage Produces Spurious Power Laws

**What goes wrong:**
With too few model sizes per compute budget (or too few compute budgets), the `find_optimal_per_budget` step picks an optimal N that is actually just the least-bad option from a sparse grid. The resulting power law fit has high R-squared because 3-4 points on a log-log plot always look linear, but the fitted exponents (e.g., a in `N_opt = k * C^a`) are wrong. Extrapolation then recommends wildly wrong model sizes at target compute budgets.

The current codebase defaults to `num_compute_budgets=5` and `num_model_sizes=7`. Five compute budgets is the absolute minimum for a credible fit; seven model sizes per budget is borderline. If some get skipped (the `tokens < model_size // 10` filter), you may end up with 3-4 model sizes per budget -- too few to find a reliable minimum.

**Why it happens:**
People optimize for wall-clock time over statistical rigor. Running 35 training jobs feels like a lot, but scaling law methodology requires enough resolution around the loss minimum to distinguish it from noise. The Chinchilla paper used ~400 experiments across 9 compute budgets.

**How to avoid:**
- Use at least 7 compute budgets (not 5) for production fits
- Use at least 10 model sizes per budget, concentrated in the region where the loss curve bends (not uniformly in log-space)
- Implement adaptive refinement: run a coarse sweep first, then add model sizes around the apparent minimum for each budget
- Add a validation check: if the optimal model size is at the boundary of the explored range for any budget, flag it and extend the range
- Report confidence intervals on exponents, not just R-squared

**Warning signs:**
- Optimal model size for any budget is the smallest or largest in the sweep
- R-squared > 0.99 on a log-log fit with fewer than 6 points (suspiciously good)
- Fitted exponent deviates significantly from literature values (Chinchilla: a ~ 0.50 for N_opt vs C)
- Loss curves per budget look flat (no clear minimum) rather than U-shaped

**Phase to address:**
Phase 1 (Plugin Architecture) -- the sweep planner abstraction should enforce minimum coverage constraints. Phase 2 (Multi-modal) -- different modalities may need different sweep densities.

---

### Pitfall 2: Fitting Power Laws in Log-Space Biases Toward Large-Scale Points

**What goes wrong:**
The current `fit_power_law` method fits `log(y) = log(k) + a * log(x)` using least squares. This minimizes relative error in log-space, which means errors at large compute budgets (where values are huge) are weighted equally to errors at small budgets (where values are small). In practice, this biases the fit toward getting the large-scale behavior right at the expense of small-scale accuracy.

More critically, for `L_opt(C)` (loss vs compute), fitting in log-space assumes multiplicative noise, but training loss noise is additive. This can produce systematically wrong loss predictions.

**Why it happens:**
Log-log linear regression is the "textbook" approach and it is numerically simple. It works well enough for N_opt and D_opt (which span orders of magnitude). But for loss -- which spans maybe 0.5 units on a linear scale -- the log transform distorts the error structure.

**How to avoid:**
- For `N_opt(C)` and `D_opt(C)`: log-space OLS is acceptable. But use `scipy.optimize.curve_fit` with the actual power law function `y = k * x^a` to get proper confidence intervals via covariance matrix
- For `L_opt(C)`: fit the 3-parameter Chinchilla loss form `L(C) = A * C^alpha + L_inf` using nonlinear least squares in linear space (not log-space). The irreducible loss `L_inf` is critical -- without it, the fit is fundamentally wrong
- Always report residual plots, not just R-squared
- Use bootstrap resampling for confidence intervals on all fitted parameters

**Warning signs:**
- Loss predictions are systematically too low for small compute budgets
- The `L_opt` fit has good R-squared but the residuals show a clear pattern (not random)
- Predicted loss at very large compute is negative or below known irreducible loss bounds

**Phase to address:**
Phase 1 (refactoring analyzer) -- fix the fitting methodology before making it pluggable. This is foundational correctness.

---

### Pitfall 3: DDP Gradient Synchronization Silently Produces Wrong Results

**What goes wrong:**
PyTorch `DistributedDataParallel` (DDP) synchronizes gradients via allreduce after each backward pass. Several common mistakes cause it to silently compute wrong gradients without raising errors:

1. **Unused parameters:** If any parameter does not contribute to the loss in a forward pass (e.g., conditional branches, unused layers), DDP will hang or produce incorrect gradients. This is especially likely in a plugin system where different model architectures have different forward pass structures.

2. **Inconsistent forward passes across ranks:** If different ranks take different code paths (e.g., one rank has a shorter batch and skips some logic), gradient bucketing breaks and allreduce operates on mismatched gradients.

3. **Non-deterministic operations:** Operations like `torch.nn.functional.interpolate` or atomicAdd in CUDA produce non-deterministic results. Across ranks, this means "identical" models diverge over time, even with the same data.

**Why it happens:**
DDP's gradient synchronization is invisible -- there is no explicit `allreduce` call in user code. When it goes wrong, losses just look slightly off rather than crashing. In scaling law experiments, "slightly off" losses corrupt the entire analysis.

**How to avoid:**
- Set `find_unused_parameters=True` during development/testing (performance cost, but catches unused param issues). Switch to `False` for production sweeps after verifying
- Use `torch.nn.parallel.DistributedDataParallel` with `static_graph=True` if the computation graph is the same every iteration (which it should be for standard training)
- Validate by running: (a) single-GPU for N steps, (b) multi-GPU for N steps with same total batch size. Losses should match within floating-point tolerance for the first several steps with identical seeds
- Set `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` for validation runs
- Always seed with `rank + base_seed` for data sampling, but `base_seed` alone for model init

**Warning signs:**
- Multi-GPU training loss diverges from single-GPU loss after first few steps
- Training hangs intermittently (classic unused parameter symptom)
- Final loss values have higher variance than expected across repeated runs
- `nan` or `inf` gradients appearing only in multi-GPU mode

**Phase to address:**
Phase 2 or 3 (Multi-GPU training) -- must be addressed before running real sweeps. Build a single-GPU vs multi-GPU validation test as a gate.

---

### Pitfall 4: Plugin Interface Violations Are Caught at Runtime, Not Load Time

**What goes wrong:**
Python's duck typing means a plugin that is missing a required method (e.g., `configure_optimizers`, `estimate_flops`, `create_dataloader`) won't fail until that method is actually called -- potentially hours into a training sweep. Worse, a plugin might implement a method with the wrong signature (wrong number of args, wrong return type) and only fail on certain code paths.

With the current GPT model's `configure_optimizers` returning a tuple of `(optimizer, settings_dict)`, a plugin that returns just an optimizer will break silently or crash mid-training.

**Why it happens:**
Python has no compile-time interface checking. ABC abstract methods help but only check method existence, not signatures. Runtime errors in ML training loops are especially costly because they waste GPU hours.

**How to avoid:**
- Define plugin interfaces using Python `Protocol` classes (from `typing`) -- these support structural subtyping and can be checked by mypy/pyright statically
- Implement a `validate_plugin(plugin)` function that is called at registration time (not training time). It should: (a) check all required methods exist, (b) call methods with dummy inputs to verify signatures and return types, (c) validate config schema compatibility
- Use Hydra's structured configs (`@dataclass` configs) to validate plugin configuration at load time rather than during training
- Write a "plugin conformance test suite" -- a pytest fixture that any plugin can be run through to verify interface compliance

**Warning signs:**
- Plugins work in unit tests but fail during actual sweeps
- `AttributeError` or `TypeError` appearing hours into training
- Different plugins require different calling conventions in the training loop

**Phase to address:**
Phase 1 (Plugin Architecture) -- define interfaces first, build validation into the plugin loader, before any plugin implementation.

---

### Pitfall 5: Noisy Small-Scale Experiments Dominate and Corrupt Scaling Law Fits

**What goes wrong:**
Small models trained on small datasets have inherently noisier loss values. When fitting scaling laws, these noisy small-scale points can pull the fit away from the true relationship. This is particularly bad because the goal is to extrapolate to *larger* scales -- if small-scale noise biases the exponent by even 0.05, predictions at 100x the training compute can be off by 2-3x.

The current mock trainer adds `np.random.normal(0, 0.02)` noise. In real training, noise at small scale can be 5-10x larger (random initialization effects, small dataset variance, learning rate sensitivity).

**Why it happens:**
Small models are fast to train, so people run them. But their loss values are dominated by initialization variance and data order. The signal-to-noise ratio for scaling behavior is poor below a certain compute threshold.

**How to avoid:**
- Run multiple seeds (at least 3) for each configuration and use the median loss, not the mean (robust to outliers)
- Weight the power law fit by inverse variance: configurations with lower variance across seeds get more weight
- Set a minimum compute budget threshold below which experiments are discarded from fitting
- Plot individual runs as scatter points overlaid on the fit line -- visual inspection catches outliers that statistics miss
- For the analyzer: implement outlier detection (e.g., Grubbs' test or MAD-based) on the loss values per compute budget

**Warning signs:**
- Loss standard deviation across seeds at small scale is > 10% of the mean loss
- Removing the smallest compute budget changes the fitted exponent by > 0.1
- Residual plot shows the smallest-scale points consistently above or below the fit line

**Phase to address:**
Phase 1 (core improvements) -- multi-seed support should be baked into the sweep planner and trainer, not bolted on later. The analyzer should handle repeated measurements from the start.

---

### Pitfall 6: Cross-Modality Scaling Laws Require Different Functional Forms

**What goes wrong:**
Text and image models follow different scaling relationships. Applying text scaling law forms (Chinchilla-style `L = A*N^a + B*D^b + L_inf`) directly to vision models produces poor fits because:

1. Vision transformers (ViTs) have a different relationship between patch size, resolution, and effective "token" count
2. Image classification loss (cross-entropy over classes) has fundamentally different noise characteristics than language modeling loss (cross-entropy over vocabulary)
3. The FLOPs accounting is different: ViTs have a fixed number of patches per image, making the N-D tradeoff structurally different

**Why it happens:**
The "scaling laws are universal" narrative is appealing but wrong in the details. The general shape (power law) holds, but the parameterization, exponents, and irreducible loss floor all differ by modality. A pluggable system that assumes one functional form will produce misleading results for non-text modalities.

**How to avoid:**
- Make the fitting function (functional form) part of the plugin interface, not hardcoded in the analyzer
- For text: use the Chinchilla 3-parameter form `L(N,D) = A/N^a + B/D^b + L_inf`
- For images: start with a similar form but with resolution-dependent terms. Allow the plugin to define `compute_budget(model_config)` rather than assuming `C = 6*N*D`
- Provide a "generic power law" fitter as default, but let modality plugins specify their own loss function form
- Document that scaling exponents are NOT transferable across modalities

**Warning signs:**
- R-squared is poor (< 0.9) when applying text-style fits to vision experiments
- Predicted optimal model sizes for vision tasks are unreasonably large or small
- The fitted irreducible loss is negative (physically impossible)

**Phase to address:**
Phase 2 (Multi-modal support) -- this must be a core design consideration when extending beyond text. The analyzer plugin interface must support custom loss forms.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcoding `C = 6*N*D` FLOPs formula | Simple, works for dense transformers | Breaks for MoE, sparse models, vision models with different attention patterns | Only for GPT-style dense transformers; make it a method override for Phase 2 |
| Log-space linear regression for all fits | Fast, simple, no convergence issues | Wrong error model for loss fitting; no confidence intervals | Never for `L_opt` fitting; acceptable for rough `N_opt`/`D_opt` during development |
| Single-seed training runs | 3x fewer GPU hours | Results dominated by initialization noise at small scale | Only for initial debugging; never for published scaling laws |
| Storing results as flat JSON | Simple, no dependencies | Slow for large sweeps (1000+ experiments); no querying capability | Acceptable for < 200 experiments; migrate to SQLite or Parquet for larger sweeps |
| Using `compute_bucket = round(log10(C), 2)` for grouping | Avoids floating-point equality issues | Silently merges close-but-different compute budgets; loses information | Acceptable if compute budgets are well-separated (> 3x apart); dangerous for dense sweeps |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Hydra + Plugin configs | Putting plugin-specific configs in the main config tree, causing validation errors when switching plugins | Use Hydra config groups: `model: gpt` vs `model: vit`, with each plugin providing its own structured config schema |
| PyTorch DDP + Hydra | Hydra's `@hydra.main` launches one process, but DDP needs `torchrun` to launch multiple. Using both causes config conflicts or duplicate logging | Use Hydra Compose API (`initialize_config_dir` + `compose`) inside a `torchrun`-launched script, not `@hydra.main` |
| HuggingFace Datasets + DDP | All ranks downloading the dataset simultaneously, corrupting the cache or hitting rate limits | Download on rank 0 first (`if rank == 0: load_dataset(...)`), then `torch.distributed.barrier()`, then all ranks load from cache |
| scipy curve_fit + power laws | Using default initial guesses `p0=[1, 1]` for `k * x^a` -- optimizer converges to local minimum or fails | Estimate `p0` from log-log linear regression first, then refine with nonlinear fit. Set `maxfev=10000` and use `bounds` to constrain `a` to physically plausible range (0 < a < 1 for scaling exponents) |
| u-mup + Plugin models | Assuming all plugins need u-mup-style LR scaling. Vision models and non-transformer architectures have different optimal parameterizations | Make the optimizer configuration part of the model plugin (as it already is with `configure_optimizers`), but clearly document that the base_width scaling is GPT-specific |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Loading full dataset per training run | Wall time dominated by data loading, not training | Pre-tokenize and memory-map the dataset once; share across all sweep runs | > 50 experiments in a sweep; dataset > 1GB |
| JSON results file growing unbounded | Incremental save becomes slow; risk of corruption if process killed during write | Use append-only JSONL (one result per line) or SQLite with WAL mode for crash safety | > 500 experiments; long-running sweeps |
| Re-instantiating model from scratch for each sweep point | Overhead of model creation and data loading per experiment | Batch experiments by model architecture (all sizes of same depth together) to reuse dataloaders | > 100 experiments |
| Fitting with all data points before filtering outliers | One crashed run with loss=NaN or loss=100.0 corrupts the fit silently (current code filters NaN but not outliers) | Implement IQR-based or MAD-based outlier detection before fitting; log rejected points | Any sweep with > 20 experiments (statistically: some will be outliers) |

## "Looks Done But Isn't" Checklist

- [ ] **Power law fit:** R-squared is reported but confidence intervals on exponents are missing -- verify you have `k_ci` and `a_ci` populated (currently always `None` in the code)
- [ ] **Multi-GPU training:** Loss matches single-GPU baseline for first 100 steps -- verify with an automated comparison test
- [ ] **Plugin registration:** Plugin loads without error but has not been tested with the full plan-train-analyze-visualize pipeline end-to-end -- verify with an integration test per plugin
- [ ] **Sweep coverage:** All compute budgets have a clear loss minimum -- verify the optimal is not at the boundary of the explored model sizes
- [ ] **Dataset handling:** Plugin claims to support a dataset but tokenization/preprocessing is incompatible with the training loop's expected tensor shapes -- verify with a single forward pass before launching the sweep
- [ ] **Config merging:** Hydra config for a new plugin overrides default values unintentionally -- verify with `hydra.utils.instantiate` in a test, printing the resolved config
- [ ] **Reproducibility:** Setting the same seed produces the same loss -- verify across single-GPU runs (DDP reproducibility is harder and should be a separate verification)

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Sparse sweep coverage | MEDIUM | Add more model sizes around the apparent minimum for each budget; re-fit with augmented data; do not need to re-run existing experiments |
| Wrong fitting functional form | LOW | Re-run analyzer with corrected form on existing results data; no retraining needed |
| DDP gradient desync | HIGH | Must re-run affected experiments from scratch after fixing the bug; cannot trust any results from desynchronized runs |
| Plugin interface mismatch mid-sweep | MEDIUM | Fix the plugin, resume sweep from checkpoint (if `resume=True` is working); only re-run failed experiments |
| Noisy small-scale corruption | MEDIUM | Re-run small-scale experiments with 3+ seeds; use median aggregation; re-fit. Or simply raise the minimum compute budget and discard small-scale points |
| Cross-modality wrong functional form | LOW-MEDIUM | Re-analyze with correct form if raw results are saved; may need to adjust sweep ranges for the new modality |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Insufficient sweep coverage | Phase 1: Planner refactoring | Automated check: no optimal at boundary; minimum 7 model sizes per budget with valid results |
| Log-space fitting bias | Phase 1: Analyzer refactoring | Compare log-space vs linear-space fits; verify residuals are unstructured |
| DDP gradient desync | Phase 2/3: Multi-GPU support | Automated test: single-GPU vs multi-GPU loss comparison for 100 steps |
| Plugin interface violations | Phase 1: Plugin architecture | Plugin conformance test suite; validate at registration time |
| Noisy small-scale experiments | Phase 1: Multi-seed support | Report per-config variance; flag high-variance points |
| Cross-modality functional forms | Phase 2: Multi-modal support | Per-modality fit quality checks; validate exponents against literature |

## Sources

- Training data knowledge of Chinchilla (Hoffmann et al., 2022) methodology and known pitfalls
- Training data knowledge of PyTorch DDP internals (official docs pattern: `find_unused_parameters`, `static_graph`, deterministic algorithms)
- Training data knowledge of Python Protocol-based plugin patterns
- Training data knowledge of Hydra config groups and compose API
- Direct code analysis of flops-fit codebase (current `analyzer.py`, `trainer.py`, `planner.py`, `model.py`)
- **Confidence caveat:** WebSearch and WebFetch were unavailable during this research. All findings are from training data (cutoff: May 2025). PyTorch DDP and scipy APIs are stable and well-established (HIGH confidence for those). Scaling law methodology pitfalls are well-documented in ML literature (HIGH confidence). Plugin architecture patterns are general software engineering (HIGH confidence). Cross-modality scaling specifics have MEDIUM confidence -- verify against recent vision scaling papers.

---
*Pitfalls research for: flops-fit scaling law tool (pluggable, multi-modal)*
*Researched: 2026-02-15*
