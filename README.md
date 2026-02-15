# Scaling Laws Toolkit

A toolkit for running scaling law experiments using the **IsoFLOPs method** from the [Chinchilla paper](https://arxiv.org/abs/2203.15556).

## Overview

This toolkit helps you:
- **Plan** sweep configurations across compute budgets
- **Train** models (or query training APIs) at different scales
- **Analyze** results to fit power laws: `N_opt = k * C^a`
- **Visualize** IsoFLOPs curves and scaling relationships

## Installation

```bash
uv sync
```

## Usage

### 1. Plan a sweep

Generate configurations for multiple model sizes at fixed compute budgets:

```bash
ff-plan --help
```

### 2. Run training

Execute training runs (or mock runs for testing):

```bash
ff-train --help
```

### 3. Analyze results

Fit power laws and find optimal model sizes:

```bash
ff-analyze --help
```

### 4. Generate visualizations

Create IsoFLOPs curves and scaling plots:

```bash
ff-visualize --help
```

### All-in-one

```bash
flops-fit
```

## IsoFLOPs Method

The IsoFLOPs approach from Chinchilla works as follows:

1. **Fix compute budget C** (in FLOPs)
2. **Vary model size N** and training tokens D such that `6 * N * D ≈ C`
3. **Train each configuration** and record final loss
4. **Find optimal N** for each compute budget (minimum loss)
5. **Fit power law**: `N_opt = k * C^a` across compute budgets

This reveals the optimal model size for any target compute budget.

## Configuration

Configs use [Hydra](https://hydra.cc/) for flexibility. See `src/scaling_laws/conf/` for defaults.

Example override:
```bash
ff-plan compute.min_flops=1e18 compute.max_flops=1e22
```

## Project Structure

```
flops-fit/
├── src/scaling_laws/
│   ├── conf/           # Hydra config files
│   ├── planner.py      # Sweep configuration generator
│   ├── trainer.py      # Training execution
│   ├── analyzer.py     # Power law fitting
│   └── visualizer.py   # Plot generation
├── outputs/            # Experiment results
├── tests/              # Unit tests
└── pyproject.toml      # Package config
```

## References

- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al.)
