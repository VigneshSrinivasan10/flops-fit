# Codebase Concerns

**Analysis Date:** 2026-02-15

## Tech Debt

**`local` and `api` training modes are unimplemented stubs:**
- Issue: `TrainingRunner` accepts `mode="local"` and `mode="api"` but only sets `self._train_fn = None` for both. Any call to `run_experiment` immediately raises `NotImplementedError`. The default config in `trainer.yaml` is `mode: mock`, and `RECOMMENDATIONS.md` guides users to switch to `mode=local`, which will crash.
- Files: `src/flops_fit/trainer.py` (lines 100-101, 176-177)
- Impact: The entire real-training workflow is broken. Users following the documented quick-start path (`ff-train mode=local`) will get a hard error with no code to fix.
- Fix approach: Implement `_local_train` that instantiates `GPT` via `create_model_for_scaling`, runs a training loop using the dataset/hardware config from Hydra, and returns `(loss, actual_flops, wall_time)`.

**Analyzer config options declared but not consumed:**
- Issue: `analyzer.yaml` declares several fitting controls — `fitting.robust`, `fitting.enforce_monotonic`, `fitting.method` (empirical vs quadratic), `validation.leave_one_out`, `validation.exponent_tolerance`, `output.generate_configs`, `output.save_isoflops` — but none of these are read or acted on in `ScalingLawAnalyzer`. The `main()` function passes only `results_path` and `output_dir` to the class.
- Files: `src/flops_fit/analyzer.py` (lines 349-352), `src/flops_fit/conf/analyzer.yaml`
- Impact: Users who set `fitting.robust=true` or `fitting.enforce_monotonic=true` believing these affect results will get identical output regardless. Results can contain poor-quality fits (low R²) with no warning or rejection.
- Fix approach: Thread `cfg.fitting` and `cfg.validation` into `ScalingLawAnalyzer.__init__`, add R² threshold check that warns or raises when below `min_r_squared`, add optional monotonicity enforcement to `find_optimal_per_budget`.

**Planner config `architecture` section declared but not consumed:**
- Issue: `planner.yaml` has an `architecture` block (`d_model_multiple`, `min_layers`, `max_layers`, `head_options`) that is never read by `SweepPlanner`. The planner only consumes `cfg.compute` and `cfg.model`.
- Files: `src/flops_fit/planner.py` (lines 232-241), `src/flops_fit/conf/planner.yaml`
- Impact: No actual impact on output today, but creates a false impression that architecture constraints are enforced when planning sweeps.
- Fix approach: Either remove the dead config block or wire it into `generate_model_sizes` and `generate_sweep`.

**Planner `lr_schedule` config declared but not consumed:**
- Issue: `planner.yaml` declares an `lr_schedule` block (`base_lr`, `base_size`, `lr_power`) for per-model-size LR scaling, but this is never read and no LR is included in `ExperimentConfig`.
- Files: `src/flops_fit/planner.py`, `src/flops_fit/conf/planner.yaml`
- Impact: The planned sweep has no LR assignment per experiment. When real training is eventually implemented, all models will use the same LR from `trainer.yaml` rather than the intended scaled-per-size LR.
- Fix approach: Add `learning_rate` field to `ExperimentConfig` and populate it from the planner using the schedule formula.

**README references stale paths:**
- Issue: `README.md` (lines 82-92) shows `src/scaling_laws/` as the package path, but the actual package is `src/flops_fit/`. The project was renamed but the README structure diagram was not updated.
- Files: `README.md`
- Impact: Developer confusion about project structure.
- Fix approach: Update the directory tree in README to show `src/flops_fit/`.

**Stale `sl-*` command names in docstrings:**
- Issue: Several module docstrings reference the old command names `sl-plan`, `sl-train`, `sl-analyze`, `sl-visualize` in the `Usage:` examples. The actual entry points defined in `pyproject.toml` are `ff-plan`, `ff-train`, `ff-analyze`, `ff-visualize`.
- Files: `src/flops_fit/analyzer.py` (lines 17-19), `src/flops_fit/planner.py` (lines 15-17), `src/flops_fit/trainer.py` (lines 20-21), `src/flops_fit/visualizer.py` (lines 14-15)
- Impact: Copy-pasting example commands from docstrings will fail with command not found.
- Fix approach: Replace all `sl-*` references with `ff-*` (or `uv run ff-*`) in docstrings.

**`estimate_params_from_config` undercounts attention parameters:**
- Issue: `estimate_params_from_config` uses `4 * d_model * d_model` for attention, appropriate for a standard 4-matrix layout (Q, K, V, O each of size `d_model × d_model`). However, the actual `CausalSelfAttention` uses a fused `qkv` projection of size `d_model × 3*d_model` plus a projection of `d_model × d_model`, which is also 4×d_model². This happens to be consistent, but the comment says "QKV + proj" which is misleading — the QKV matrix is one weight not three.
- Files: `src/flops_fit/model.py` (lines 444-455)
- Impact: Low risk; parameter count estimates are accurate in total. Documentation is slightly misleading.

## Known Bugs

**`find_optimal_per_budget` uses a different bucket rounding than `plot_scaling_laws`:**
- Symptoms: `analyzer.py` rounds `log10(compute_budget)` to 2 decimal places (`np.round(..., 2)`) to create compute buckets, while `visualizer.py` rounds to 1 decimal place (`np.round(..., 1)`). If two experiments have compute budgets that map to the same 1-decimal bucket but different 2-decimal buckets, the analyzer may treat them as separate budgets while the visualizer groups them together, leading to mismatched optimal-point counts between analysis and plots.
- Files: `src/flops_fit/analyzer.py` (line 175), `src/flops_fit/visualizer.py` (lines 135, 216, 301)
- Trigger: Experiments with compute budgets that differ only in the second decimal of log10 space (e.g., `1.01e14` vs `1.04e14`).
- Workaround: None documented. In practice, the planner generates exact logspace values which are unlikely to collide at 2 decimal places.

**`_mock_train` uses unseeded random noise, making results non-reproducible:**
- Symptoms: Each run of `ff-train mode=mock` produces different loss values due to `np.random.normal(0, 0.02)` and `np.random.uniform(0.98, 1.02)` without seeding. The config has `trainer.seed: 42` but this seed is never applied in `TrainingRunner`.
- Files: `src/flops_fit/trainer.py` (lines 147-158)
- Trigger: Any two successive `ff-train mode=mock` runs with the same config.
- Workaround: None. Results will differ between runs, making it impossible to reproduce a specific mock sweep.

**`create_model_for_scaling` approximate parameter formula can diverge significantly:**
- Symptoms: `create_model_for_scaling` estimates `d_model` using `N ≈ 12 * L * d^2`, but the actual non-embedding parameter count (from `estimate_params_from_config`) includes FFN terms: `attention_per_layer = 4*d²` and `ffn_per_layer = 3 * d * 4d = 12*d²`, giving `16*d²` per layer, not `12*d²`. For a target of 10M params with 6 layers, the approximation gives `d_model ≈ sqrt(10M/72) ≈ 373`, rounded to 384; actual count at d=384, L=6 is approximately 16*6*384² ≈ 14.2M — 42% over target.
- Files: `src/flops_fit/model.py` (lines 493-494)
- Trigger: Any call to `create_model_for_scaling` with moderate-to-large target parameter counts.
- Workaround: Not documented. The `count_parameters` method exists on the model and could be used to verify after construction.

## Security Considerations

**No security concerns identified:**
- This is a local research tool with no network-facing components, authentication, or user-provided data execution paths. The only external network call is the Hugging Face `datasets` download in the (currently unimplemented) local training mode.
- Files: `src/flops_fit/trainer.py` (line 28 imports `datasets` but trainer mode `local` is not implemented)
- Current mitigation: N/A
- Recommendations: When implementing local training, ensure the `cache_dir` config is validated to prevent path traversal if this tool is ever exposed through a script interface.

## Performance Bottlenecks

**Incremental JSON write on every experiment is O(n) per step:**
- Problem: `TrainingRunner.run_sweep` writes the full accumulated results list to disk after every single experiment (`json.dump(results, f, ...)` inside the training loop). For large sweeps this means O(n²) total write work.
- Files: `src/flops_fit/trainer.py` (lines 237-243)
- Cause: Simple append + full rewrite chosen for simplicity and crash-safety.
- Improvement path: Append each result as a newline-delimited JSON record to a `.jsonl` file, then convert to a single JSON list at end. Alternatively, open the file in append mode with a mutex.

**Rotary embedding cache is rebuilt when sequences exceed `max_seq_len`:**
- Problem: `RotaryEmbedding._build_cache` is called at init time for `max_seq_len`. If any forward pass sees a longer sequence, the cache is rebuilt on that call. The cached tensors are stored on CPU and then moved to device on each forward pass (`to(x.device)`), which involves a device copy every forward pass rather than being pre-placed on the target device.
- Files: `src/flops_fit/model.py` (lines 93-95)
- Cause: Cache is stored as plain attributes (not registered buffers), so it does not move automatically with `.to(device)`.
- Improvement path: Store `_cos_cached` and `_sin_cached` via `register_buffer` or pre-move to device after construction.

## Fragile Areas

**`TrainingRunner.run_sweep` resume logic depends on experiment_id stability:**
- Files: `src/flops_fit/trainer.py` (lines 224-234)
- Why fragile: The resume mechanism identifies completed experiments by `experiment_id` (e.g., `"exp_0003"`). `experiment_id` is assigned by sequential counter in `SweepPlanner.generate_sweep`. If the planner config changes (e.g., `num_model_sizes` changes from 7 to 8), IDs shift, and previously completed experiments will be re-run while their old results remain in the file — producing duplicates.
- Safe modification: Only change `SweepPlanner` config if starting a clean sweep (delete `outputs/results.json` first).
- Test coverage: No tests cover the resume-after-config-change scenario.

**`ScalingLawAnalyzer.predict` reconstructs `PowerLawFit` by hand from raw JSON instead of using the dataclass:**
- Files: `src/flops_fit/analyzer.py` (lines 310-336)
- Why fragile: The `predict` method loads `scaling_laws.json` and manually does `coefficient_k * (target_compute ** exponent_a)` rather than deserializing into a `PowerLawFit` and calling `.predict()`. If `PowerLawFit` ever changes its formula or field names, this path diverges silently.
- Safe modification: Add a `PowerLawFit.from_dict` classmethod and use it in `predict`.
- Test coverage: No tests cover `ScalingLawAnalyzer.predict`.

**`GPTConfig.d_ff` is `Optional[int]` with a `__post_init__` default but typing says it could be `None`:**
- Files: `src/flops_fit/model.py` (lines 37-51)
- Why fragile: `d_ff` starts `None` but is set to `4 * d_model` in `__post_init__`. All downstream code that uses `config.d_ff` assumes it is always an `int`. Dataclass copy operations (e.g., `dataclasses.replace(config, d_ff=None)`) would re-introduce `None` and cause runtime failures in `FeedForward.__init__`.
- Safe modification: Use `field(default=None)` and treat the post-init resolution as canonical; alternatively type as `int` with a sentinel.

## Scaling Limits

**Mock training wall-time model does not reflect real CPU training:**
- Current capacity: Mock sweeps complete in seconds regardless of compute budget.
- Limit: RECOMMENDATIONS.md documents real sweep estimates of 50-100 hours for a full `cpu_full` preset. There is no budget enforcement or progress estimation for real training since `mode=local` is unimplemented.
- Scaling path: When real training is added, expose a `--dry-run` that estimates total wall time by timing one small experiment and extrapolating.

**Dataset size caps scaling experiments at ~470M tokens:**
- Current capacity: TinyStories has ~470M tokens. At the maximum compute budget in `planner.yaml` (`max_flops: 3e14`), a 500K-param model requires `D = 3e14 / (6 * 5e5) ≈ 1e8` tokens — well within capacity.
- Limit: Scaling to `max_flops=1e15+` with very small models would require more tokens than TinyStories provides, causing dataset repetition. No detection or warning is implemented.
- Scaling path: Add a dataset-size check in the planner that warns when `tokens_needed > dataset_tokens`.

## Dependencies at Risk

**No lockfile committed:**
- Risk: `pyproject.toml` specifies only minimum version bounds (e.g., `torch>=2.0.0`, `transformers>=4.35.0`). Without a committed `uv.lock`, builds can silently pick up newer incompatible versions.
- Impact: Reproducibility risk for experiments; behavior of `datasets`, `transformers`, and `scipy` can change between minor versions.
- Migration plan: Run `uv lock` and commit the resulting `uv.lock` file.

## Missing Critical Features

**No actual training loop implementation:**
- Problem: The core purpose of the toolkit — running real scaling experiments — requires `mode=local` to work. This entire code path does not exist. `GPT`, `GPTConfig`, `create_model_for_scaling` are implemented, but nothing connects them to dataset loading, the training loop, optimizer configuration, or loss recording.
- Blocks: Any real (non-mock) scaling law experiment.

**No checkpoint saving or recovery:**
- Problem: Training runner has a `paths.checkpoints` config path but no code writes or reads checkpoints. If a long-running experiment crashes mid-sweep, the partially-completed experiment contributes no data and must be re-run from scratch.
- Blocks: Reliably running multi-hour sweeps (documented as 50-100 hours total).

**Confidence intervals are declared but never computed:**
- Problem: `PowerLawFit` has `k_ci` and `a_ci` fields for 95% confidence intervals, but `fit_power_law` always returns `PowerLawFit(..., k_ci=None, a_ci=None)`. The confidence intervals are never populated.
- Files: `src/flops_fit/analyzer.py` (lines 238-243)
- Blocks: Statistical validity assessment of fitted scaling laws.

## Test Coverage Gaps

**No tests for `TrainingRunner`:**
- What's not tested: `run_experiment`, `run_sweep`, `load_sweep`, resume logic, mock noise behavior.
- Files: `src/flops_fit/trainer.py`
- Risk: Resume-on-corrupt-JSON, handling of `failed` experiment records, and the incremental save loop are all untested.
- Priority: High (core pipeline step)

**No tests for `ScalingVisualizer`:**
- What's not tested: All three plot methods, `load_data`, bucket rounding behavior.
- Files: `src/flops_fit/visualizer.py`
- Risk: Bucket rounding inconsistency with analyzer goes undetected; matplotlib state mutation from `_setup_style` could affect other tests.
- Priority: Medium

**No tests for `GPT` model:**
- What's not tested: Forward pass, loss computation, u-mup initialization, `configure_optimizers`, `create_model_for_scaling` parameter accuracy.
- Files: `src/flops_fit/model.py`
- Risk: The parameter count approximation bug in `create_model_for_scaling` is undetected. Incorrect u-mup initialization would silently produce wrong scaling behavior.
- Priority: High (correctness of scaling experiments depends on correct initialization)

**No integration test for the full pipeline:**
- What's not tested: End-to-end: `plan → train (mock) → analyze → visualize`.
- Files: All four main modules.
- Risk: Inter-module data format assumptions (e.g., column names in `results.json`) can break silently. The bucket rounding mismatch between analyzer and visualizer is an example of this class of bug.
- Priority: Medium

---

*Concerns audit: 2026-02-15*
