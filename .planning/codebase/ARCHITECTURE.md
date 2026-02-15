# Architecture

**Analysis Date:** 2026-02-15

## Pattern Overview

**Overall:** Linear Pipeline (Plan → Train → Analyze → Visualize)

**Key Characteristics:**
- Four sequential, independent pipeline stages each implemented as a standalone class with a `main()` entry point
- File-based data handoff between stages via JSON files in `outputs/`
- Hydra configuration management for all pipeline stages
- No shared state between stages at runtime; each reads from and writes to disk

## Layers

**CLI Entry Layer:**
- Purpose: Route user commands to pipeline stage entry points
- Location: `flops-fit.sh` (root), and `ff-plan`, `ff-train`, `ff-analyze`, `ff-visualize` console scripts (defined in `pyproject.toml`)
- Contains: Shell dispatcher and Hydra-decorated `main()` functions
- Depends on: Python package `flops_fit`
- Used by: End users and automation

**Pipeline Stage Layer (Core Logic):**
- Purpose: The four pipeline stages as classes and Hydra `main()` functions
- Location: `src/flops_fit/planner.py`, `src/flops_fit/trainer.py`, `src/flops_fit/analyzer.py`, `src/flops_fit/visualizer.py`
- Contains: Business logic classes (`SweepPlanner`, `TrainingRunner`, `ScalingLawAnalyzer`, `ScalingVisualizer`) plus dataclasses for results
- Depends on: Config layer (Hydra/YAML), model layer, outputs directory for JSON handoff
- Used by: CLI entry layer

**Model Layer:**
- Purpose: PyTorch GPT model with u-mup parameterization for scaling law experiments
- Location: `src/flops_fit/model.py`
- Contains: `GPT`, `GPTConfig`, `CausalSelfAttention`, `FeedForward`, `TransformerBlock`, `RMSNorm`, `RotaryEmbedding`, plus utility functions `estimate_model_flops`, `create_model_for_scaling`, `estimate_params_from_config`
- Depends on: PyTorch
- Used by: `trainer.py` (for local training mode), and directly by users for programmatic access

**Configuration Layer:**
- Purpose: Hydra YAML configs with per-stage defaults and override support
- Location: `src/flops_fit/conf/planner.yaml`, `src/flops_fit/conf/trainer.yaml`, `src/flops_fit/conf/analyzer.yaml`, `src/flops_fit/conf/visualizer.yaml`, `src/flops_fit/conf/presets/cpu_fast.yaml`, `src/flops_fit/conf/presets/cpu_full.yaml`
- Contains: Compute budgets, model size bounds, hardware settings, dataset settings, path mappings
- Depends on: Hydra-core
- Used by: All pipeline stage `main()` functions via `@hydra.main` decorator

**Output/Artifact Layer:**
- Purpose: File-based data exchange between pipeline stages
- Location: `outputs/` (gitignored, created at runtime)
- Contains: `sweep.json` (plan → train), `results.json` (train → analyze), `analysis/scaling_laws.json` (analyze → visualize), `plots/` (visualize outputs)
- Depends on: Nothing
- Used by: All pipeline stages as input or output

## Data Flow

**Primary Pipeline Flow:**

1. `ff-plan` runs `SweepPlanner.save_sweep()` → writes `outputs/sweep.json` (list of `ExperimentConfig` dicts with experiment_id, compute_budget, model_size, num_tokens)
2. `ff-train` runs `TrainingRunner.run_sweep()` → reads `outputs/sweep.json`, executes training (mock/local/api mode), writes `outputs/results.json` (list of `TrainingResult` dicts with final_loss, actual_flops, wall_time, status)
3. `ff-analyze` runs `ScalingLawAnalyzer.analyze()` → reads `outputs/results.json`, fits power laws, writes `outputs/analysis/scaling_laws.json` (fitted `PowerLawFit` parameters: coefficient_k, exponent_a, r_squared)
4. `ff-visualize` runs `ScalingVisualizer.plot_all()` → reads `outputs/results.json` + `outputs/analysis/scaling_laws.json`, writes PNG plots to `outputs/plots/`

**State Management:**
- No in-memory state between stages; all state is persisted to JSON files
- Trainer supports resume mode: reads existing `results.json` and skips completed experiment IDs
- No database; pure filesystem

## Key Abstractions

**ExperimentConfig:**
- Purpose: Represents a single (compute_budget, model_size, num_tokens) training configuration on an IsoFLOP curve
- Examples: `src/flops_fit/planner.py` (lines 35–57)
- Pattern: `@dataclass` with `__post_init__` for derived fields; `to_dict()` for JSON serialization

**TrainingResult:**
- Purpose: Records outcome of a single training run (loss, actual FLOPs, wall time, status)
- Examples: `src/flops_fit/trainer.py` (lines 37–69)
- Pattern: `@dataclass` with `to_dict()` for JSON serialization; supports failed/skipped status

**PowerLawFit:**
- Purpose: Encapsulates a fitted power law `y = k * x^a` with R² and prediction method
- Examples: `src/flops_fit/analyzer.py` (lines 39–65)
- Pattern: `@dataclass` with `predict(x)` method and `to_dict()` for serialization

**ScalingAnalysis:**
- Purpose: Groups three `PowerLawFit` objects (N_opt, D_opt, L_opt vs compute) plus optimal points
- Examples: `src/flops_fit/analyzer.py` (lines 68–113)
- Pattern: `@dataclass` with `predict_optimal_size(target_compute)` convenience method

**GPT / GPTConfig:**
- Purpose: Transformer language model with u-mup hyperparameter-transfer-friendly parameterization
- Examples: `src/flops_fit/model.py` (lines 29–514)
- Pattern: `nn.Module` with config dataclass; `configure_optimizers()` returns per-parameter-group AdamW with width-scaled LR and WD

## Entry Points

**`ff-plan` / `flops-fit plan`:**
- Location: `src/flops_fit/planner.py:main()` (decorated with `@hydra.main(config_name="planner")`)
- Triggers: CLI invocation
- Responsibilities: Instantiate `SweepPlanner` from config, call `save_sweep()`, log summary

**`ff-train` / `flops-fit train`:**
- Location: `src/flops_fit/trainer.py:main()` (decorated with `@hydra.main(config_name="trainer")`)
- Triggers: CLI invocation after plan step
- Responsibilities: Instantiate `TrainingRunner` from config, call `run_sweep()`, log completion stats

**`ff-analyze` / `flops-fit analyze`:**
- Location: `src/flops_fit/analyzer.py:main()` (decorated with `@hydra.main(config_name="analyzer")`)
- Triggers: CLI invocation after train step
- Responsibilities: Instantiate `ScalingLawAnalyzer`, call `analyze()`, print summary table, optionally predict for a target compute budget

**`ff-visualize` / `flops-fit visualize`:**
- Location: `src/flops_fit/visualizer.py:main()` (decorated with `@hydra.main(config_name="visualizer")`)
- Triggers: CLI invocation after analyze step
- Responsibilities: Instantiate `ScalingVisualizer`, call `plot_all()`, optionally call `plt.show()`

**`flops-fit.sh`:**
- Location: `/home/viggie/Projects/flops-fit/flops-fit.sh`
- Triggers: Direct shell invocation
- Responsibilities: Dispatch subcommand (`plan`, `train`, `analyze`, `visualize`/`viz`) to corresponding `ff-*` console script via `exec`

## Error Handling

**Strategy:** Fail-fast at stage boundaries (FileNotFoundError if prior stage output missing); per-experiment error capture within the training loop

**Patterns:**
- `TrainingRunner.run_experiment()` wraps each experiment in try/except, stores error in `TrainingResult.error_message` with status `"failed"`, continues sweep rather than aborting
- `ScalingVisualizer.plot_all()` wraps each plot type in try/except with warning log, continuing to remaining plots
- `ScalingLawAnalyzer.load_results()` and `TrainingRunner.load_sweep()` raise `FileNotFoundError` with actionable messages if prerequisite files are missing
- `GPTConfig.__post_init__` uses `assert` for architectural invariants (d_model divisible by num_heads)

## Cross-Cutting Concerns

**Logging:** Python `logging` module with module-level `logger = logging.getLogger(__name__)` in every stage module; Hydra configures output

**Validation:** Dataclass `__post_init__` for structural invariants; power law fitting validates minimum data points and filters `nan`/`inf` values before fitting

**Configuration:** Hydra with `@hydra.main(version_base=None, config_path="conf", config_name="...")` on every `main()` function; all stage configs set `hydra.run.dir: .` to prevent Hydra from changing working directory

---

*Architecture analysis: 2026-02-15*
