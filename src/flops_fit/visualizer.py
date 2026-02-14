#!/usr/bin/env python3
"""
Scaling Law Visualizer

Generates plots for scaling law experiments:
- IsoFLOPs curves: Loss vs model size for each compute budget
- Scaling plots: Optimal N, D, L vs compute budget
- Power law fits: Fitted curves overlaid on data

Visualization follows the style of Chinchilla and similar papers,
with log-scale axes and clear annotations.

Usage:
    uv run sl-visualize
    uv run sl-visualize plots.style=paper  # Publication-ready style
"""

from pathlib import Path
from typing import Literal
import json
import logging

import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd


logger = logging.getLogger(__name__)


# Color palette for compute budgets
COMPUTE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


class ScalingVisualizer:
    """
    Generate visualizations for scaling law experiments.
    
    Creates publication-quality plots showing:
    - IsoFLOPs curves (loss landscape)
    - Scaling relationships (N_opt, D_opt, L_opt vs C)
    - Power law fits
    
    Attributes:
        results_path: Path to training results JSON
        analysis_path: Path to scaling law analysis JSON
        output_dir: Directory for generated plots
        style: Plot style ("paper" for publication, "notebook" for exploration)
    """
    
    def __init__(
        self,
        results_path: str | Path = "outputs/results.json",
        analysis_path: str | Path = "outputs/analysis/scaling_laws.json",
        output_dir: str | Path = "outputs/plots",
        style: Literal["paper", "notebook"] = "paper",
    ):
        self.results_path = Path(results_path)
        self.analysis_path = Path(analysis_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style
        
        self._setup_style()
    
    def _setup_style(self):
        """Configure matplotlib style."""
        if self.style == "paper":
            plt.rcParams.update({
                "font.family": "serif",
                "font.size": 12,
                "axes.labelsize": 14,
                "axes.titlesize": 14,
                "legend.fontsize": 10,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "figure.figsize": (8, 6),
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "axes.grid": True,
                "grid.alpha": 0.3,
            })
        else:
            plt.rcParams.update({
                "figure.figsize": (10, 7),
                "figure.dpi": 100,
                "axes.grid": True,
                "grid.alpha": 0.3,
            })
    
    def load_data(self) -> tuple[pd.DataFrame, dict | None]:
        """Load results and analysis data."""
        # Load results
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results not found: {self.results_path}")
        
        with open(self.results_path) as f:
            results = json.load(f)
        df = pd.DataFrame(results)
        df = df[df["status"] == "completed"]
        
        # Load analysis (optional)
        analysis = None
        if self.analysis_path.exists():
            with open(self.analysis_path) as f:
                analysis = json.load(f)
        
        return df, analysis
    
    def plot_isoflops(self, save: bool = True) -> plt.Figure:
        """
        Plot IsoFLOPs curves: Loss vs model size for each compute budget.
        
        Each curve represents a fixed compute budget, showing how loss
        varies with model size (and correspondingly, training tokens).
        The minimum of each curve indicates the optimal model size for
        that compute budget.
        
        Args:
            save: Whether to save the figure
            
        Returns:
            Matplotlib Figure
        """
        df, analysis = self.load_data()
        
        # Create compute budget buckets
        df["compute_bucket"] = np.round(np.log10(df["compute_budget"]), 1)
        budgets = sorted(df["compute_bucket"].unique())
        
        fig, ax = plt.subplots()
        
        for i, budget in enumerate(budgets):
            mask = df["compute_bucket"] == budget
            data = df[mask].sort_values("model_size")
            
            color = COMPUTE_COLORS[i % len(COMPUTE_COLORS)]
            label = f"C = 10^{budget:.1f} FLOPs"
            
            ax.plot(
                data["model_size"],
                data["final_loss"],
                "o-",
                color=color,
                label=label,
                markersize=6,
                linewidth=1.5,
            )
            
            # Mark optimal point
            optimal_idx = data["final_loss"].idxmin()
            optimal = data.loc[optimal_idx]
            ax.plot(
                optimal["model_size"],
                optimal["final_loss"],
                "s",
                color=color,
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1.5,
                zorder=10,
            )
        
        ax.set_xscale("log")
        ax.set_xlabel("Model Size (Parameters)")
        ax.set_ylabel("Final Loss")
        ax.set_title("IsoFLOPs Curves: Loss vs Model Size")
        ax.legend(loc="upper right", framealpha=0.9)
        
        # Add annotation
        ax.annotate(
            "■ = optimal for each compute budget",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=9,
            color="gray",
        )
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "isoflops_curves.png"
            fig.savefig(path)
            logger.info(f"Saved IsoFLOPs plot to {path}")
        
        return fig
    
    def plot_scaling_laws(self, save: bool = True) -> plt.Figure:
        """
        Plot scaling laws: N_opt, D_opt, L_opt vs compute budget.
        
        Shows the power law relationships between compute budget and:
        - Optimal model size (N_opt)
        - Optimal training tokens (D_opt)  
        - Achievable loss (L_opt)
        
        Includes fitted power law curves if analysis is available.
        
        Args:
            save: Whether to save the figure
            
        Returns:
            Matplotlib Figure
        """
        df, analysis = self.load_data()
        
        # Get optimal points per budget
        df["compute_bucket"] = np.round(np.log10(df["compute_budget"]), 1)
        optimal_idx = df.groupby("compute_bucket")["final_loss"].idxmin()
        optimal_df = df.loc[optimal_idx]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        C = optimal_df["compute_budget"].values
        N = optimal_df["model_size"].values
        D = optimal_df["num_tokens"].values
        L = optimal_df["final_loss"].values
        
        # Plot 1: N_opt vs C
        ax = axes[0]
        ax.scatter(C, N, s=80, color="steelblue", edgecolor="black", zorder=5)
        
        if analysis:
            fit = analysis["n_opt_fit"]
            C_range = np.logspace(np.log10(C.min()), np.log10(C.max()), 100)
            N_fit = fit["coefficient_k"] * np.power(C_range, fit["exponent_a"])
            ax.plot(C_range, N_fit, "r-", linewidth=2, label=f"N = {fit['coefficient_k']:.2e} × C^{fit['exponent_a']:.3f}")
            ax.legend(loc="upper left")
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Compute Budget (FLOPs)")
        ax.set_ylabel("Optimal Model Size (N)")
        ax.set_title("Optimal Model Size vs Compute")
        
        # Plot 2: D_opt vs C
        ax = axes[1]
        ax.scatter(C, D, s=80, color="forestgreen", edgecolor="black", zorder=5)
        
        if analysis:
            fit = analysis["d_opt_fit"]
            D_fit = fit["coefficient_k"] * np.power(C_range, fit["exponent_a"])
            ax.plot(C_range, D_fit, "r-", linewidth=2, label=f"D = {fit['coefficient_k']:.2e} × C^{fit['exponent_a']:.3f}")
            ax.legend(loc="upper left")
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Compute Budget (FLOPs)")
        ax.set_ylabel("Optimal Tokens (D)")
        ax.set_title("Optimal Training Tokens vs Compute")
        
        # Plot 3: L_opt vs C
        ax = axes[2]
        ax.scatter(C, L, s=80, color="darkorange", edgecolor="black", zorder=5)
        
        if analysis:
            fit = analysis["l_opt_fit"]
            L_fit = fit["coefficient_k"] * np.power(C_range, fit["exponent_a"])
            ax.plot(C_range, L_fit, "r-", linewidth=2, label=f"L = {fit['coefficient_k']:.2e} × C^{fit['exponent_a']:.3f}")
            ax.legend(loc="upper right")
        
        ax.set_xscale("log")
        ax.set_xlabel("Compute Budget (FLOPs)")
        ax.set_ylabel("Optimal Loss")
        ax.set_title("Achievable Loss vs Compute")
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "scaling_laws.png"
            fig.savefig(path)
            logger.info(f"Saved scaling laws plot to {path}")
        
        return fig
    
    def plot_tokens_per_param(self, save: bool = True) -> plt.Figure:
        """
        Plot the ratio of tokens to parameters for optimal configurations.
        
        The Chinchilla paper found that optimal configurations have roughly
        equal scaling for tokens and parameters. This plot shows D/N ratio
        across compute budgets.
        
        Args:
            save: Whether to save the figure
            
        Returns:
            Matplotlib Figure
        """
        df, analysis = self.load_data()
        
        # Get optimal points
        df["compute_bucket"] = np.round(np.log10(df["compute_budget"]), 1)
        optimal_idx = df.groupby("compute_bucket")["final_loss"].idxmin()
        optimal_df = df.loc[optimal_idx]
        
        C = optimal_df["compute_budget"].values
        ratio = optimal_df["num_tokens"].values / optimal_df["model_size"].values
        
        fig, ax = plt.subplots()
        
        ax.scatter(C, ratio, s=80, color="purple", edgecolor="black", zorder=5)
        
        # Add horizontal line at median
        median_ratio = np.median(ratio)
        ax.axhline(median_ratio, color="red", linestyle="--", linewidth=2, 
                   label=f"Median ratio: {median_ratio:.1f}")
        
        # Chinchilla reference (roughly 20 tokens per param)
        ax.axhline(20, color="gray", linestyle=":", linewidth=1.5,
                   label="Chinchilla (~20)")
        
        ax.set_xscale("log")
        ax.set_xlabel("Compute Budget (FLOPs)")
        ax.set_ylabel("Tokens / Parameters (D/N)")
        ax.set_title("Optimal Tokens-to-Parameters Ratio")
        ax.legend(loc="upper right")
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "tokens_per_param.png"
            fig.savefig(path)
            logger.info(f"Saved tokens/param plot to {path}")
        
        return fig
    
    def plot_all(self, save: bool = True) -> list[plt.Figure]:
        """
        Generate all plots.
        
        Args:
            save: Whether to save figures
            
        Returns:
            List of matplotlib Figures
        """
        figures = []
        
        try:
            figures.append(self.plot_isoflops(save=save))
        except Exception as e:
            logger.warning(f"Failed to plot IsoFLOPs: {e}")
        
        try:
            figures.append(self.plot_scaling_laws(save=save))
        except Exception as e:
            logger.warning(f"Failed to plot scaling laws: {e}")
        
        try:
            figures.append(self.plot_tokens_per_param(save=save))
        except Exception as e:
            logger.warning(f"Failed to plot tokens/param: {e}")
        
        logger.info(f"Generated {len(figures)} plots in {self.output_dir}")
        return figures


@hydra.main(version_base=None, config_path="conf", config_name="visualizer")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for visualization.
    
    Uses Hydra for configuration. Override defaults via CLI:
        uv run sl-visualize plots.style=notebook
    """
    logger.info("Generating visualizations...")
    
    visualizer = ScalingVisualizer(
        results_path=cfg.paths.results,
        analysis_path=cfg.paths.analysis,
        output_dir=cfg.paths.plots,
        style=cfg.plots.get("style", "paper"),
    )
    
    figures = visualizer.plot_all(save=True)
    
    print(f"\nGenerated {len(figures)} plots in {cfg.paths.plots}")
    
    # Show if interactive
    if cfg.plots.get("show", False):
        plt.show()


if __name__ == "__main__":
    main()
