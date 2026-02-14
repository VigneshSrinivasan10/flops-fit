#!/usr/bin/env python3
"""
Scaling Law Analyzer

Fits power laws to experimental results to find optimal scaling relationships.
Implements the IsoFLOPs analysis from Chinchilla:

1. For each compute budget C, find the model size N with minimum loss
2. Fit the relationship: N_opt = k * C^a
3. Use this to predict optimal model sizes for new compute budgets

Key relationships:
- N_opt(C) = k_N * C^a  (optimal params vs compute)
- D_opt(C) = k_D * C^b  (optimal tokens vs compute)
- L_opt(C) = k_L * C^c  (achievable loss vs compute)

Usage:
    uv run sl-analyze
    uv run sl-analyze paths.results=outputs/results.json
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import json
import logging

import hydra
from omegaconf import DictConfig
import numpy as np
from scipy import optimize
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class PowerLawFit:
    """Result of fitting a power law: y = k * x^a"""
    
    name: str
    coefficient_k: float  # Multiplicative constant
    exponent_a: float  # Power law exponent
    r_squared: float  # Goodness of fit
    
    # Confidence intervals (95%)
    k_ci: tuple[float, float] | None = None
    a_ci: tuple[float, float] | None = None
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict y values for given x."""
        return self.coefficient_k * np.power(x, self.exponent_a)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "coefficient_k": self.coefficient_k,
            "exponent_a": self.exponent_a,
            "r_squared": self.r_squared,
            "k_ci": list(self.k_ci) if self.k_ci else None,
            "a_ci": list(self.a_ci) if self.a_ci else None,
            "formula": f"{self.name} = {self.coefficient_k:.4e} * C^{self.exponent_a:.4f}",
        }


@dataclass
class ScalingAnalysis:
    """Complete scaling law analysis results."""
    
    # Power law fits
    n_opt_fit: PowerLawFit  # N_opt(C)
    d_opt_fit: PowerLawFit  # D_opt(C)
    l_opt_fit: PowerLawFit  # L_opt(C)
    
    # Optimal points per compute budget
    optimal_points: list[dict] = field(default_factory=list)
    
    # Chinchilla-style ratio
    optimal_ratio: float | None = None  # D_opt / N_opt ratio
    
    def predict_optimal_size(self, target_compute: float) -> dict:
        """
        Predict optimal model configuration for a target compute budget.
        
        Args:
            target_compute: Target FLOPs
            
        Returns:
            Dictionary with optimal N, D, and expected loss
        """
        n_opt = self.n_opt_fit.predict(np.array([target_compute]))[0]
        d_opt = self.d_opt_fit.predict(np.array([target_compute]))[0]
        l_opt = self.l_opt_fit.predict(np.array([target_compute]))[0]
        
        return {
            "target_compute": target_compute,
            "optimal_params": int(n_opt),
            "optimal_tokens": int(d_opt),
            "expected_loss": float(l_opt),
            "tokens_per_param": d_opt / n_opt if n_opt > 0 else 0,
        }
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "n_opt_fit": self.n_opt_fit.to_dict(),
            "d_opt_fit": self.d_opt_fit.to_dict(),
            "l_opt_fit": self.l_opt_fit.to_dict(),
            "optimal_points": self.optimal_points,
            "optimal_ratio": self.optimal_ratio,
        }


class ScalingLawAnalyzer:
    """
    Analyze experiment results to extract scaling laws.
    
    The analyzer:
    1. Groups results by compute budget
    2. Finds the optimal model size for each budget (minimum loss)
    3. Fits power laws: N_opt = k * C^a, D_opt = k * C^b, L = k * C^c
    
    These power laws enable predicting optimal configurations for
    new compute budgets without running experiments.
    
    Attributes:
        results_path: Path to results JSON file
        output_dir: Directory to save analysis
    """
    
    def __init__(
        self,
        results_path: str | Path = "outputs/results.json",
        output_dir: str | Path = "outputs/analysis",
    ):
        self.results_path = Path(results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_results(self) -> pd.DataFrame:
        """Load training results into a DataFrame."""
        if not self.results_path.exists():
            raise FileNotFoundError(
                f"Results file not found: {self.results_path}. "
                "Run 'sl-train' first to generate results."
            )
        
        with open(self.results_path) as f:
            results = json.load(f)
        
        df = pd.DataFrame(results)
        
        # Filter to completed experiments
        df = df[df["status"] == "completed"]
        logger.info(f"Loaded {len(df)} completed experiments")
        
        return df
    
    def find_optimal_per_budget(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find the optimal model size for each compute budget.
        
        For each unique compute budget, finds the configuration with
        the lowest final loss.
        
        Args:
            df: DataFrame with training results
            
        Returns:
            DataFrame with one row per compute budget (optimal config)
        """
        # Group by compute budget (with some tolerance for floating point)
        df["compute_bucket"] = np.round(np.log10(df["compute_budget"]), 2)
        
        # Find minimum loss per bucket
        idx = df.groupby("compute_bucket")["final_loss"].idxmin()
        optimal_df = df.loc[idx].copy()
        
        logger.info(f"Found {len(optimal_df)} optimal configurations")
        return optimal_df
    
    def fit_power_law(
        self,
        x: np.ndarray,
        y: np.ndarray,
        name: str,
    ) -> PowerLawFit:
        """
        Fit a power law: y = k * x^a
        
        Uses log-linear regression for numerical stability:
        log(y) = log(k) + a * log(x)
        
        Args:
            x: Independent variable (e.g., compute budget)
            y: Dependent variable (e.g., optimal params)
            name: Name for the fit (e.g., "N_opt")
            
        Returns:
            PowerLawFit with coefficients and goodness of fit
        """
        # Filter out invalid values
        valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        
        if len(x) < 2:
            raise ValueError(f"Not enough valid points to fit {name}")
        
        # Log transform
        log_x = np.log10(x)
        log_y = np.log10(y)
        
        # Linear regression in log space
        def linear_model(params, x):
            return params[0] + params[1] * x
        
        def residuals(params, x, y):
            return y - linear_model(params, x)
        
        # Initial guess
        p0 = [np.mean(log_y), 0.5]
        
        # Fit
        result = optimize.least_squares(residuals, p0, args=(log_x, log_y))
        log_k, a = result.x
        k = 10 ** log_k
        
        # R-squared
        y_pred = linear_model(result.x, log_x)
        ss_res = np.sum((log_y - y_pred) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        logger.info(f"{name} = {k:.4e} * C^{a:.4f} (R² = {r_squared:.4f})")
        
        return PowerLawFit(
            name=name,
            coefficient_k=k,
            exponent_a=a,
            r_squared=r_squared,
        )
    
    def analyze(self) -> ScalingAnalysis:
        """
        Run complete scaling law analysis.
        
        Returns:
            ScalingAnalysis with all fitted power laws
        """
        logger.info("Starting scaling law analysis...")
        
        # Load and process data
        df = self.load_results()
        optimal_df = self.find_optimal_per_budget(df)
        
        # Extract arrays
        C = optimal_df["compute_budget"].values
        N = optimal_df["model_size"].values
        D = optimal_df["num_tokens"].values
        L = optimal_df["final_loss"].values
        
        # Fit power laws
        n_opt_fit = self.fit_power_law(C, N, "N_opt")
        d_opt_fit = self.fit_power_law(C, D, "D_opt")
        l_opt_fit = self.fit_power_law(C, L, "L_opt")
        
        # Compute optimal ratio
        optimal_ratio = np.median(D / N)
        logger.info(f"Median D/N ratio: {optimal_ratio:.2f}")
        
        # Build optimal points list
        optimal_points = optimal_df[[
            "compute_budget", "model_size", "num_tokens", "final_loss"
        ]].to_dict(orient="records")
        
        analysis = ScalingAnalysis(
            n_opt_fit=n_opt_fit,
            d_opt_fit=d_opt_fit,
            l_opt_fit=l_opt_fit,
            optimal_points=optimal_points,
            optimal_ratio=optimal_ratio,
        )
        
        # Save analysis
        self.save_analysis(analysis)
        
        return analysis
    
    def save_analysis(self, analysis: ScalingAnalysis):
        """Save analysis results to JSON."""
        output_path = self.output_dir / "scaling_laws.json"
        
        with open(output_path, "w") as f:
            json.dump(analysis.to_dict(), f, indent=2)
        
        logger.info(f"Saved analysis to {output_path}")
    
    def predict(self, target_compute: float) -> dict:
        """
        Load saved analysis and predict optimal config.
        
        Args:
            target_compute: Target compute budget in FLOPs
            
        Returns:
            Dictionary with optimal configuration
        """
        analysis_path = self.output_dir / "scaling_laws.json"
        
        if not analysis_path.exists():
            raise FileNotFoundError(
                f"Analysis not found: {analysis_path}. "
                "Run 'sl-analyze' first."
            )
        
        with open(analysis_path) as f:
            data = json.load(f)
        
        # Reconstruct fits
        n_fit = data["n_opt_fit"]
        d_fit = data["d_opt_fit"]
        l_fit = data["l_opt_fit"]
        
        n_opt = n_fit["coefficient_k"] * (target_compute ** n_fit["exponent_a"])
        d_opt = d_fit["coefficient_k"] * (target_compute ** d_fit["exponent_a"])
        l_opt = l_fit["coefficient_k"] * (target_compute ** l_fit["exponent_a"])
        
        return {
            "target_compute": target_compute,
            "optimal_params": int(n_opt),
            "optimal_tokens": int(d_opt),
            "expected_loss": float(l_opt),
            "tokens_per_param": d_opt / n_opt if n_opt > 0 else 0,
        }


@hydra.main(version_base=None, config_path="conf", config_name="analyzer")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for scaling law analysis.
    
    Uses Hydra for configuration. Override defaults via CLI:
        uv run sl-analyze paths.results=custom_results.json
    """
    logger.info("Starting scaling law analysis...")
    
    analyzer = ScalingLawAnalyzer(
        results_path=cfg.paths.results,
        output_dir=cfg.paths.analysis,
    )
    
    analysis = analyzer.analyze()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SCALING LAW ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nOptimal Model Size:   {analysis.n_opt_fit.to_dict()['formula']}")
    print(f"Optimal Tokens:       {analysis.d_opt_fit.to_dict()['formula']}")
    print(f"Achievable Loss:      {analysis.l_opt_fit.to_dict()['formula']}")
    print(f"\nMedian D/N ratio:     {analysis.optimal_ratio:.2f}")
    print("=" * 60)
    
    # Example prediction
    if cfg.get("predict_compute"):
        target = float(cfg.predict_compute)
        pred = analysis.predict_optimal_size(target)
        print(f"\nPrediction for C = {target:.2e} FLOPs:")
        print(f"  Optimal params:  {pred['optimal_params']:,}")
        print(f"  Optimal tokens:  {pred['optimal_tokens']:,}")
        print(f"  Expected loss:   {pred['expected_loss']:.4f}")


if __name__ == "__main__":
    main()
