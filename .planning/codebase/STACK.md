# Technology Stack

**Analysis Date:** 2026-02-15

## Languages

**Primary:**
- Python 3.11 - All application code (`src/flops_fit/`)

**Secondary:**
- Bash - CLI entry point wrapper (`flops-fit.sh`)
- YAML - Hydra configuration files (`src/flops_fit/conf/`)

## Runtime

**Environment:**
- Python 3.11 (pinned in `.python-version`)

**Package Manager:**
- uv (configured via `[tool.uv]` in `pyproject.toml`)
- Lockfile: `uv.lock` (present but gitignored)

## Frameworks

**Configuration:**
- hydra-core >= 1.3.2 - CLI configuration management with YAML override support; used in all four entry points (`planner.py`, `trainer.py`, `analyzer.py`, `visualizer.py`)
- omegaconf - Hydra dependency; used for `DictConfig` type throughout

**Machine Learning:**
- torch >= 2.0.0 - GPT model implementation (`src/flops_fit/model.py`); used for transformer, attention, optimizers
- transformers >= 4.35.0 - Listed as dependency; tokenizer support (GPT-2 tokenizer referenced in config)
- datasets >= 2.14.0 - HuggingFace datasets; TinyStories dataset loading referenced in config

**Scientific Computing:**
- numpy >= 1.26.0 - Array operations across all modules
- scipy >= 1.11.0 - Power law curve fitting via `scipy.optimize.least_squares` in `analyzer.py`
- pandas >= 2.1.0 - DataFrame operations for results processing in `analyzer.py` and `visualizer.py`

**Visualization:**
- matplotlib >= 3.8.0 - All plots generated in `visualizer.py`; supports "paper" and "notebook" styles

**Utilities:**
- tqdm >= 4.66.0 - Progress bars in `trainer.py`
- pyyaml >= 6.0 - YAML support (Hydra dependency)

**Testing:**
- pytest >= 8.0.0 - Test runner; config in `pyproject.toml` (`testpaths = ["tests"]`, `pythonpath = ["src"]`)
- pytest-cov >= 4.1.0 - Coverage reporting

**Linting:**
- ruff >= 0.1.0 - Linting and import sorting; configured in `pyproject.toml` with line-length=100, target py311, rules E/F/I/W

## Key Dependencies

**Critical:**
- `hydra-core` - All entry points are decorated with `@hydra.main`; removing it breaks the CLI entirely
- `torch` - Core model in `src/flops_fit/model.py`; required for local training mode
- `scipy` - Power law fitting in `src/flops_fit/analyzer.py`; required for analysis step

**Infrastructure:**
- `datasets` - HuggingFace dataset loading; required for local training mode with TinyStories
- `transformers` - GPT-2 tokenizer; required for local training mode

## Configuration

**Environment:**
- No `.env` file; no environment variables required for default mock mode
- Dataset cache stored in `.cache/datasets` (gitignored via outputs exclusion)

**Build:**
- `pyproject.toml` - Single source of truth for project metadata, dependencies, scripts, and tool config
- `src/` layout with `[tool.setuptools] package-dir = {"" = "src"}` and `package = true` for uv

**Hydra Config Files:**
- `src/flops_fit/conf/planner.yaml` - Sweep planning settings
- `src/flops_fit/conf/trainer.yaml` - Training settings including hardware, dataset, hyperparameters
- `src/flops_fit/conf/analyzer.yaml` - Analysis settings
- `src/flops_fit/conf/visualizer.yaml` - Plot settings
- `src/flops_fit/conf/presets/cpu_fast.yaml` - Fast validation preset (gitignored)
- `src/flops_fit/conf/presets/cpu_full.yaml` - Full sweep preset (gitignored)

## Entry Points (CLI Scripts)

Defined in `pyproject.toml` under `[project.scripts]`:

```
ff-plan       → flops_fit.planner:main
ff-train      → flops_fit.trainer:main
ff-analyze    → flops_fit.analyzer:main
ff-visualize  → flops_fit.visualizer:main
flops-fit.sh  → bash wrapper dispatching to above commands
```

## Platform Requirements

**Development:**
- Python 3.11+
- uv package manager
- Optional: GPU (CUDA) for training; defaults to CPU mode
- Optimized config targets AMD Ryzen 7 5825U (8 cores, 30GB RAM, no GPU)

**Production:**
- No deployment target; local research tool
- Output files written to `outputs/` directory (gitignored)

---

*Stack analysis: 2026-02-15*
