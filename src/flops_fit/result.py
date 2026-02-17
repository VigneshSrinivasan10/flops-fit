"""Result object returned from find_optimal().

Wraps ScalingAnalysis and ScalingVisualizer to expose a cohesive
user-facing API after running the full scaling law pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from flops_fit.analyzer import ScalingAnalysis
    from flops_fit.visualizer import ScalingVisualizer


@dataclass
class Result:
    """
    Results from find_optimal() pipeline.

    Aggregates power law analysis and visualization capability.
    All heavy lifting is delegated to ScalingAnalysis (fitting)
    and ScalingVisualizer (plotting) from Phase 5.

    Attributes:
        analysis: Fitted ScalingAnalysis from ScalingLawAnalyzer.analyze()
        visualizer: ScalingVisualizer configured with same output_dir paths
        output_dir: Root directory for all pipeline artifacts
        compute_budgets: Compute budgets used in the sweep
    """

    analysis: ScalingAnalysis
    visualizer: ScalingVisualizer
    output_dir: str | Path = "outputs"
    compute_budgets: list[float] = field(default_factory=list)

    def chinchilla_table(
        self, compute_budgets: list[float] | None = None
    ) -> str:
        """Return Chinchilla-style markdown table of optimal model configurations.

        Args:
            compute_budgets: FLOPs budgets to tabulate. If None, uses 9
                log-spaced budgets from 1e18 to 1e22.

        Returns:
            Markdown-formatted table string with columns:
            Compute Budget | Optimal N | Optimal D | D/N Ratio | Predicted Loss
        """
        return self.analysis.chinchilla_table(compute_budgets)

    def predict(self, compute_budget: float) -> dict:
        """Predict optimal model configuration for a target compute budget.

        Args:
            compute_budget: Target FLOPs

        Returns:
            Dict with keys: target_compute, optimal_params, optimal_tokens,
            expected_loss, tokens_per_param
        """
        return self.analysis.predict_optimal_size(compute_budget)

    def plot(self, show: bool = False) -> list[plt.Figure]:
        """Generate all scaling law visualizations.

        Creates three figures:
        - IsoFLOPs curves (loss vs model size per compute budget)
        - Scaling laws (N_opt, D_opt, L_opt vs compute budget)
        - Tokens-per-param ratio across compute budgets

        Figures are saved as PNGs to {output_dir}/plots/ by the visualizer.

        Args:
            show: If True, display plots interactively via plt.show()

        Returns:
            List of matplotlib Figure objects
        """
        figures = self.visualizer.plot_all(save=True)
        if show:
            plt.show()
        return figures
