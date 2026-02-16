#!/usr/bin/env python3
"""
Sweep Configuration Planner

Generates configurations for scaling law experiments using the IsoFLOPs method.
For each compute budget C, creates multiple configurations with different model
sizes N and corresponding token counts D such that 6*N*D ≈ C.

Key concepts:
- Compute budget (C): Total FLOPs for training = 6 * N * D
- Model size (N): Number of parameters
- Training tokens (D): Number of tokens to train on
- IsoFLOP curve: All (N, D) pairs with the same compute budget

Usage:
    uv run sl-plan
    uv run sl-plan compute.min_flops=1e18 compute.max_flops=1e22
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
import json
import logging

import hydra
from omegaconf import DictConfig
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single training experiment."""
    
    experiment_id: str
    compute_budget: float  # Total FLOPs
    model_size: int  # Number of parameters (N)
    num_tokens: int  # Training tokens (D)
    
    # Derived values
    tokens_per_param: float = field(init=False)
    
    def __post_init__(self):
        self.tokens_per_param = self.num_tokens / self.model_size if self.model_size > 0 else 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "compute_budget": self.compute_budget,
            "model_size": self.model_size,
            "num_tokens": self.num_tokens,
            "tokens_per_param": self.tokens_per_param,
        }


class SweepPlanner:
    """
    Generate sweep configurations for IsoFLOPs experiments.
    
    The IsoFLOPs method from Chinchilla works by:
    1. Fixing multiple compute budgets (e.g., 1e18, 1e19, 1e20 FLOPs)
    2. For each budget, varying model size N while adjusting tokens D
    3. Training each configuration and recording final loss
    4. Finding the optimal N for each compute budget
    
    This class generates all the (N, D) configurations to train.
    
    Attributes:
        min_flops: Minimum compute budget to explore
        max_flops: Maximum compute budget to explore
        num_compute_budgets: Number of compute budgets to sample
        num_model_sizes: Number of model sizes per compute budget
        min_model_size: Smallest model to consider
        max_model_size: Largest model to consider
    """
    
    def __init__(
        self,
        min_flops: float = 1e17,
        max_flops: float = 1e21,
        num_compute_budgets: int = 5,
        num_model_sizes: int = 7,
        min_model_size: int = 10_000_000,  # 10M
        max_model_size: int = 10_000_000_000,  # 10B
    ):
        self.min_flops = min_flops
        self.max_flops = max_flops
        self.num_compute_budgets = num_compute_budgets
        self.num_model_sizes = num_model_sizes
        self.min_model_size = min_model_size
        self.max_model_size = max_model_size
    
    def compute_flops(self, n_params: int, n_tokens: int) -> float:
        """
        Compute training FLOPs using the Chinchilla approximation.
        
        FLOPs ≈ 6 * N * D
        
        This accounts for forward pass (2ND) + backward pass (4ND).
        
        Args:
            n_params: Number of model parameters
            n_tokens: Number of training tokens
            
        Returns:
            Estimated training FLOPs
        """
        return 6 * n_params * n_tokens
    
    def tokens_for_compute(self, compute_budget: float, n_params: int) -> int:
        """
        Calculate tokens needed to hit a compute budget for a given model size.
        
        D = C / (6 * N)
        
        Args:
            compute_budget: Target FLOPs
            n_params: Number of model parameters
            
        Returns:
            Number of training tokens
        """
        return int(compute_budget / (6 * n_params))
    
    def generate_compute_budgets(self) -> np.ndarray:
        """Generate logarithmically spaced compute budgets."""
        return np.logspace(
            np.log10(self.min_flops),
            np.log10(self.max_flops),
            self.num_compute_budgets,
        )
    
    def generate_model_sizes(self, compute_budget: float) -> np.ndarray:
        """
        Generate model sizes for a given compute budget.
        
        Model sizes are constrained by:
        - Minimum model size
        - Maximum model size
        - Compute budget (can't have more params than compute allows)
        
        Args:
            compute_budget: Target FLOPs
            
        Returns:
            Array of model sizes to evaluate
        """
        # Maximum feasible model size for this compute budget
        # (assuming at least 1 token per parameter)
        max_feasible = int(compute_budget / 6)
        
        # Constrain to our bounds
        actual_min = self.min_model_size
        actual_max = min(self.max_model_size, max_feasible)
        
        if actual_max <= actual_min:
            return [actual_min]
        
        return np.logspace(
            np.log10(actual_min),
            np.log10(actual_max),
            self.num_model_sizes,
        ).astype(int).tolist()
    
    def generate_sweep(self) -> Iterator[ExperimentConfig]:
        """
        Generate all experiment configurations for the sweep.
        
        Yields:
            ExperimentConfig for each (compute_budget, model_size) combination
        """
        compute_budgets = self.generate_compute_budgets()
        
        exp_idx = 0
        for compute_budget in compute_budgets:
            model_sizes = self.generate_model_sizes(compute_budget)
            
            for model_size in model_sizes:
                num_tokens = self.tokens_for_compute(compute_budget, model_size)
                
                # Skip if tokens < model size (too few tokens)
                if num_tokens < model_size // 10:
                    logger.debug(
                        f"Skipping N={model_size:.2e}: too few tokens ({num_tokens:.2e})"
                    )
                    continue
                
                yield ExperimentConfig(
                    experiment_id=f"exp_{exp_idx:04d}",
                    compute_budget=compute_budget,
                    model_size=model_size,
                    num_tokens=num_tokens,
                )
                exp_idx += 1
    
    def save_sweep(self, output_path: str | Path) -> list[dict]:
        """
        Generate sweep and save to JSON file.
        
        Args:
            output_path: Path to save the sweep configuration
            
        Returns:
            List of experiment configurations as dictionaries
        """
        configs = [cfg.to_dict() for cfg in self.generate_sweep()]
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(configs, f, indent=2)
        
        logger.info(f"Saved {len(configs)} experiment configs to {output_path}")
        return configs


@hydra.main(version_base=None, config_path="conf", config_name="planner")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for sweep planning.
    
    Uses Hydra for configuration management. Override defaults via CLI:
        uv run sl-plan compute.min_flops=1e18 compute.max_flops=1e22
    """
    logger.info("Generating sweep configurations...")
    
    planner = SweepPlanner(
        min_flops=cfg.compute.min_flops,
        max_flops=cfg.compute.max_flops,
        num_compute_budgets=cfg.compute.num_budgets,
        num_model_sizes=cfg.compute.num_model_sizes,
        min_model_size=cfg.model.min_size,
        max_model_size=cfg.model.max_size,
    )
    
    configs = planner.save_sweep(cfg.output.sweep_path)
    
    # Print summary
    logger.info(f"Generated {len(configs)} experiments")
    logger.info(f"Compute range: {cfg.compute.min_flops:.2e} - {cfg.compute.max_flops:.2e} FLOPs")
    logger.info(f"Model range: {cfg.model.min_size:.2e} - {cfg.model.max_size:.2e} params")


if __name__ == "__main__":
    main()
