"""
flops-fit: Scaling Laws Toolkit

A toolkit for running scaling law experiments using the IsoFLOPs method
from the Chinchilla paper (Hoffmann et al., 2022).

Features:
- u-mup parameterization for hyperparameter transfer across widths
- IsoFLOPs sweep planning and execution
- Power law fitting: N_opt = k * C^a
- Visualization of scaling curves

Key components:
- model: GPT with u-mup (Unit-Scaled Maximal Update Parametrization)
- planner: Generate sweep configurations across compute budgets
- trainer: Execute training runs (or query training APIs)
- analyzer: Fit power laws and find optimal model sizes
- visualizer: Generate IsoFLOPs curves and scaling plots
"""

__version__ = "0.1.0"

from flops_fit.api import find_optimal
from flops_fit.model import GPT, GPTConfig, create_model_for_scaling, estimate_model_flops
from flops_fit.examples import TinyStoriesDataset
from flops_fit.sweep import SweepPlan, Experiment, plan_sweep
from flops_fit.planner import SweepPlanner
from flops_fit.trainer import TrainingRunner
from flops_fit.analyzer import ScalingLawAnalyzer
from flops_fit.visualizer import ScalingVisualizer
from flops_fit.result import Result

__all__ = [
    # Public API
    "find_optimal",
    # Result object
    "Result",
    # Model
    "GPT",
    "GPTConfig",
    "create_model_for_scaling",
    "estimate_model_flops",
    # Dataset
    "TinyStoriesDataset",
    # Sweep Planning
    "SweepPlan",
    "Experiment",
    "plan_sweep",
    # Pipeline
    "SweepPlanner",
    "TrainingRunner",
    "ScalingLawAnalyzer",
    "ScalingVisualizer",
]
