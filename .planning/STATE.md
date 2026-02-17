# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-16)

**Core value:** Given a compute budget, tell the user exactly how big their model should be and how much data to train on -- for their specific architecture and dataset.
**Current focus:** Phase 7 complete. GPT examples package + example scripts + 188 tests passing.

## Current Position

Phase: 8 of 9 (ViT + CIFAR Example) -- IN PROGRESS (1/2 plans complete)
Plan: 1/2 complete
Status: VisionTransformer + vit_loss_fn in examples/vit.py; CIFAR10Dataset in examples/cifar.py; 7 exports from flops_fit.examples; 188 tests passing.
Last activity: 2026-02-17 -- 08-01 complete: VisionTransformer + CIFAR10Dataset

Progress: [█████████░] 83%

## Performance Metrics

**Velocity:**
- Total plans completed: 14
- Average duration: ~7min
- Total execution time: ~89min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-baseline | 3/3 | ~50min | ~17min |
| 01-skeleton | 1/1 | ~2min | ~2min |
| 02-dataset-and-loss | 2/2 | ~4min | ~2min |
| 03-sweep-planning | 2/2 | ~5min | ~2.5min |
| 04-training-engine | 2/2 | ~4min | ~2min |
| 05-analysis-and-fitting | 3/3 | ~22min | ~7min |
| 06-results-object-and-api-integration | 2/2 | ~6min | ~3min |
| 07-gpt-and-tinystories-example | 2/3 | ~7min | ~3.5min |
| 08-vit-and-cifar-example | 1/2 | ~2min | ~2min |

**Recent Trend:**
- Last 3 plans: 05-01 (~18min), 05-02 (~2min), 05-03 (~2min)
- Trend: Gap closure plans are fast (minimal, targeted TDD fixes)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Library-first pivot: Python objects as input, not YAML/config
- Existing CLI/Hydra becomes example, not core
- Model contract: class + size parameter name + `num_params()`
- Mock train loss range is 1.5-15.0 (small models produce legitimately high loss ~9.6)
- Analyzer integration tests need 5+ compute budgets (3 produces identical optimal N, r²=0)
- create_model_for_scaling 12*L*d² approximation overshoots by ~7.5x at small scales due to SwiGLU + d_model rounding
- Known inconsistency: analyzer uses 2-decimal bucket rounding, visualizer uses 1-decimal
- Hydra config tests use initialize_config_dir (not config_module) since conf/ has no __init__.py
- Duck typing for model contract: num_params() -> int, no base class required
- Probe-based validation creates small instance (size=64) to verify contract up front
- Warning (not error) when size_param appears in model_kwargs
- IterableDataset wrapping forces shuffle=False (torch requirement)
- nn.Module signature inspection targets .forward not __call__ for accurate param counts
- Uninspectable callables (C extensions) pass validation silently
- drop_last=True on all wrapped DataLoaders for consistent batch sizes
- Standalone plan_sweep() function (not class) -- matches library-first design
- Log-spaced size values rounded to multiples of 64 for GPU efficiency
- Probe models via create_model() with try/except to skip invalid sizes gracefully
- Feasibility filter: tokens >= num_params/10 (matching existing planner.py)
- Configurable flops_per_param_per_token (default 6) for non-transformer architectures
- _local_train() uses actual model.num_params() post-creation for FLOPs (not experiment.num_params)
- SGD optimizer with lr=0.01 default for local training (scaling law experiments need loss signal, not optimized convergence)
- Model cleanup: del model + torch.cuda.empty_cache() after each experiment to prevent GPU OOM across sweeps
- run_sweep_from_plan returns list[dict] (matching existing run_sweep() API) for consistency
- Lazy-import TrainingRunner inside training branch to avoid circular imports and keep startup fast
- train=True default makes training the happy path when dataset+loss_fn provided (explicit opt-out via train=False)
- output_dir defaults to 'outputs' (string not Path) so TrainingRunner handles Path conversion internally
- Linear-space NLS replaces log-space regression: unbiased when loss has additive baseline (irreducible entropy)
- fit_power_law() parametrizes optimization as [log10(k), a, l_inf] with bounds [-10,5] x [-1,2] x [0,inf]
- Test x range for l_inf recovery must be small enough that l_inf is a significant fraction of y_min (logspace(1,5) not logspace(10,20))
- chinchilla_table() defaults to np.logspace(18, 22, 9): 9 log-spaced budgets from 1e18 to 1e22
- predict_optimal_size() l_inf propagation correct via PowerLawFit.predict() from Plan 01 (no extra code needed)
- Use l_fit.get('l_inf') or 0 not l_fit.get('l_inf', 0): JSON null deserializes to Python None, and .get('l_inf', 0) returns None for null while 'or 0' correctly coerces None to 0
- Result is a pure facade: chinchilla_table/predict/plot all delegate to Phase 5 components (ScalingAnalysis, ScalingVisualizer), no reimplementation
- test_chinchilla_table_with_custom_budgets counts data rows by line filtering (not '|---' substring count; separator row has 5 occurrences of '|---', one per column)
- Analyzer requires 2+ distinct compute budget levels for power law fitting; integration tests use 5 budgets [1e8, 3e8, 1e9, 3e9, 1e10]
- find_optimal() training branch chains: run_sweep_from_plan() -> ScalingLawAnalyzer.analyze() -> Result(analysis, visualizer); results.json written as side effect
- GPT implementation lives in examples/gpt.py; model.py is thin backward-compat re-export facade
- num_params() is the primary contract method (returns all params, not just non-embedding)
- datasets/transformers lazy-imported inside prepare_data() only -- zero import-time overhead for TinyStoriesDataset
- mode='local' default in find_optimal() preserves all existing tests; example scripts default to mode='mock' with synthetic TensorDataset
- gpt_loss_fn reshapes (B, T, V) logits to (B*T, V) and labels to (B*T,) for F.cross_entropy (GPT output tuple unpacking)
- TinyStoriesDataset mock tests inject _dataset/_tokenizer directly rather than patching load_dataset
- VisionTransformer.forward() returns logits DIRECTLY (not a tuple) -- structural contrast with GPT's (logits, loss) tuple
- vit_loss_fn(logits, labels) has no tuple unpacking -- proves library handles both output patterns
- Lazy torchvision import inside _prepare_data() only (matches TinyStories lazy HF import pattern)
- norm_first=True (pre-norm transformer) for ViT -- triggers benign PyTorch UserWarning about enable_nested_tensor, not an error

### Pending Todos

None yet.

### Blockers/Concerns

- Accelerate version pin (>=1.0.0) not verified -- validate before adding to pyproject.toml
- Hydra + torchrun conflict needs Compose API workaround -- relevant for Phase 9

## Session Continuity

Last session: 2026-02-17
Stopped at: Completed 08-01-PLAN.md (VisionTransformer + CIFAR10Dataset). 188 tests passing.
Resume file: None
