#!/usr/bin/env python3
"""
Training Runner

Executes training runs for scaling law experiments. Can operate in multiple modes:
- Mock mode: Generate synthetic losses for testing
- Local mode: Train models locally (requires PyTorch)
- API mode: Query external training APIs

For IsoFLOPs experiments, we train many models at different scales and record
their final losses. The trainer handles:
- Loading sweep configurations
- Executing or queuing training runs
- Recording results (loss, actual compute, wall time)

Usage:
    uv run sl-train
    uv run sl-train mode=mock  # Use synthetic data for testing
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal
import json
import logging
import time

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from tqdm import tqdm

from flops_fit.model_factory import create_model
from flops_fit.data import wrap_dataset
from flops_fit.sweep import SweepPlan, Experiment


logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    """Return cuda:0 if available, else cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@dataclass
class TrainingResult:
    """Results from a single training run."""
    
    experiment_id: str
    compute_budget: float
    model_size: int
    num_tokens: int
    
    # Results
    final_loss: float
    actual_flops: float  # May differ slightly from budget
    wall_time_seconds: float
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: Literal["completed", "failed", "skipped"] = "completed"
    error_message: str | None = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "compute_budget": self.compute_budget,
            "model_size": self.model_size,
            "num_tokens": self.num_tokens,
            "final_loss": self.final_loss,
            "actual_flops": self.actual_flops,
            "wall_time_seconds": self.wall_time_seconds,
            "timestamp": self.timestamp,
            "status": self.status,
            "error_message": self.error_message,
        }


class TrainingRunner:
    """
    Execute training runs for scaling law experiments.
    
    The runner loads experiment configurations from the planner and executes
    training runs, either locally, via API, or using mock data for testing.
    
    For real experiments, you'd integrate with your training infrastructure
    (e.g., PyTorch training loop, cloud training APIs, etc.).
    
    Attributes:
        mode: Training mode ("mock", "local", "api")
        sweep_path: Path to sweep configuration JSON
        output_dir: Directory to save results
    """
    
    def __init__(
        self,
        mode: str = "mock",
        sweep_path: str | Path = "outputs/sweep.json",
        output_dir: str | Path = "outputs",
    ):
        self.mode = mode
        self.sweep_path = Path(sweep_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._train_fn: Callable | None = None
        if mode == "mock":
            self._train_fn = self._mock_train
    
    def load_sweep(self) -> list[dict]:
        """Load sweep configuration from JSON file."""
        if not self.sweep_path.exists():
            raise FileNotFoundError(
                f"Sweep file not found: {self.sweep_path}. "
                "Run 'sl-plan' first to generate sweep configurations."
            )
        
        with open(self.sweep_path) as f:
            return json.load(f)
    
    def _mock_train(
        self,
        model_size: int,
        num_tokens: int,
        compute_budget: float,
    ) -> tuple[float, float, float]:
        """
        Generate synthetic training results for testing.
        
        Uses a simplified scaling law model:
        L(N, D) = (N_c / N)^α + (D_c / D)^β + L_∞
        
        Where:
        - N_c, D_c are critical scales
        - α, β are scaling exponents
        - L_∞ is irreducible loss
        
        Args:
            model_size: Number of parameters
            num_tokens: Training tokens
            compute_budget: Target FLOPs
            
        Returns:
            (final_loss, actual_flops, wall_time_seconds)
        """
        # Scaling law parameters (approximate Chinchilla values)
        N_c = 4e8  # Critical model size
        D_c = 2e10  # Critical data size
        alpha = 0.34  # Model size exponent
        beta = 0.28  # Data exponent
        L_inf = 1.69  # Irreducible loss (bits per byte)
        
        # Add noise to make it realistic
        noise = np.random.normal(0, 0.02)
        
        # Compute loss
        loss = (
            (N_c / model_size) ** alpha
            + (D_c / num_tokens) ** beta
            + L_inf
            + noise
        )
        
        # Actual FLOPs (slight variation from budget)
        actual_flops = compute_budget * np.random.uniform(0.98, 1.02)
        
        # Mock wall time (scales roughly with compute)
        wall_time = np.log10(compute_budget) * 0.1  # seconds per log10(FLOP)
        
        return float(loss), float(actual_flops), float(wall_time)
    
    def run_experiment(self, config: dict) -> TrainingResult:
        """
        Run a single training experiment.
        
        Args:
            config: Experiment configuration dictionary
            
        Returns:
            TrainingResult with loss and metadata
        """
        try:
            if self._train_fn is None:
                raise NotImplementedError(f"Training mode '{self.mode}' not implemented")
            
            loss, actual_flops, wall_time = self._train_fn(
                model_size=config["model_size"],
                num_tokens=config["num_tokens"],
                compute_budget=config["compute_budget"],
            )
            
            return TrainingResult(
                experiment_id=config["experiment_id"],
                compute_budget=config["compute_budget"],
                model_size=config["model_size"],
                num_tokens=config["num_tokens"],
                final_loss=loss,
                actual_flops=actual_flops,
                wall_time_seconds=wall_time,
                status="completed",
            )
            
        except Exception as e:
            logger.error(f"Experiment {config['experiment_id']} failed: {e}")
            return TrainingResult(
                experiment_id=config["experiment_id"],
                compute_budget=config["compute_budget"],
                model_size=config["model_size"],
                num_tokens=config["num_tokens"],
                final_loss=float("nan"),
                actual_flops=0.0,
                wall_time_seconds=0.0,
                status="failed",
                error_message=str(e),
            )
    
    def run_sweep(self, resume: bool = True) -> list[TrainingResult]:
        """
        Run all experiments in the sweep.
        
        Args:
            resume: If True, skip already-completed experiments
            
        Returns:
            List of all training results
        """
        configs = self.load_sweep()
        results_path = self.output_dir / "results.json"
        
        # Load existing results if resuming
        completed = set()
        results = []
        if resume and results_path.exists():
            with open(results_path) as f:
                existing = json.load(f)
                results = existing
                completed = {r["experiment_id"] for r in existing if r["status"] == "completed"}
            logger.info(f"Resuming: {len(completed)} experiments already completed")
        
        # Run remaining experiments
        remaining = [c for c in configs if c["experiment_id"] not in completed]
        logger.info(f"Running {len(remaining)} experiments...")
        
        for config in tqdm(remaining, desc="Training"):
            result = self.run_experiment(config)
            results.append(result.to_dict())
            
            # Save incrementally
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
        
        logger.info(f"Completed {len(results)} total experiments")
        return results
    
    def _local_train(
        self,
        experiment: Experiment,
        model_cls,
        size_param: str,
        model_kwargs: dict,
        dataset_or_loader,
        loss_fn: Callable,
        epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 0.01,
    ) -> tuple[float, float, float]:
        """Train a real PyTorch model and return (final_loss, actual_flops, wall_time).

        Args:
            experiment: Experiment dataclass with size_param_value and num_tokens.
            model_cls: Model class to instantiate.
            size_param: Name of constructor parameter controlling model size.
            model_kwargs: Other constructor kwargs.
            dataset_or_loader: A torch Dataset or DataLoader.
            loss_fn: Loss function callable (outputs, targets) -> scalar tensor.
            epochs: Number of training epochs.
            batch_size: Batch size when wrapping a Dataset.
            learning_rate: Learning rate for SGD optimizer.

        Returns:
            (final_loss, actual_flops, wall_time_seconds)
        """
        model = create_model(model_cls, size_param, experiment.size_param_value, model_kwargs)
        dataloader = wrap_dataset(dataset_or_loader, batch_size=batch_size)
        device = _get_device()
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        start_time = time.time()
        model.train()

        total_loss = 0.0
        total_batches = 0

        for _epoch in range(epochs):
            for _batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs = batch
                    targets = batch

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        wall_time = time.time() - start_time
        final_loss = total_loss / total_batches if total_batches > 0 else float("nan")

        # Compute actual FLOPs: C = 6 * N * D (Chinchilla formula)
        actual_n = model.num_params()
        actual_flops = 6 * actual_n * experiment.num_tokens

        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return float(final_loss), float(actual_flops), float(wall_time)

    def run_experiment_from_sweep(
        self,
        experiment: Experiment,
        model_cls,
        size_param: str,
        model_kwargs: dict,
        dataset_or_loader,
        loss_fn: Callable,
        **train_kwargs,
    ) -> "TrainingResult":
        """Run a single experiment defined by an Experiment dataclass.

        Wraps _local_train() with error handling, returning a TrainingResult
        with status='completed' on success or status='failed' on exception.

        Args:
            experiment: Experiment dataclass from a SweepPlan.
            model_cls: Model class to instantiate.
            size_param: Name of constructor parameter controlling model size.
            model_kwargs: Other constructor kwargs.
            dataset_or_loader: A torch Dataset or DataLoader.
            loss_fn: Loss function callable.
            **train_kwargs: Additional kwargs forwarded to _local_train().

        Returns:
            TrainingResult with status 'completed' or 'failed'.
        """
        try:
            loss, actual_flops, wall_time = self._local_train(
                experiment=experiment,
                model_cls=model_cls,
                size_param=size_param,
                model_kwargs=model_kwargs,
                dataset_or_loader=dataset_or_loader,
                loss_fn=loss_fn,
                **train_kwargs,
            )
            return TrainingResult(
                experiment_id=experiment.experiment_id,
                compute_budget=experiment.compute_budget,
                model_size=experiment.num_params,
                num_tokens=experiment.num_tokens,
                final_loss=loss,
                actual_flops=actual_flops,
                wall_time_seconds=wall_time,
                status="completed",
            )
        except Exception as e:
            logger.error(f"Experiment {experiment.experiment_id} failed: {e}")
            return TrainingResult(
                experiment_id=experiment.experiment_id,
                compute_budget=experiment.compute_budget,
                model_size=experiment.num_params,
                num_tokens=experiment.num_tokens,
                final_loss=float("nan"),
                actual_flops=0.0,
                wall_time_seconds=0.0,
                status="failed",
                error_message=str(e),
            )

    def run_sweep_from_plan(
        self,
        plan: SweepPlan,
        model_cls,
        size_param: str,
        model_kwargs: dict,
        dataset_or_loader,
        loss_fn: Callable,
        resume: bool = True,
        **train_kwargs,
    ) -> list[dict]:
        """Run all experiments in a SweepPlan, with optional resume support.

        Iterates through plan.experiments, skipping any whose experiment_id
        is already recorded in results.json (when resume=True). Writes
        results.json incrementally after each experiment completes.

        Args:
            plan: SweepPlan containing Experiment entries to run.
            model_cls: Model class to instantiate.
            size_param: Name of constructor parameter controlling model size.
            model_kwargs: Other constructor kwargs.
            dataset_or_loader: A torch Dataset or DataLoader.
            loss_fn: Loss function callable.
            resume: If True, skip experiments already completed in results.json.
            **train_kwargs: Additional kwargs forwarded to run_experiment_from_sweep().

        Returns:
            List of result dicts (all experiments: pre-existing + newly run).
        """
        results_path = self.output_dir / "results.json"

        # Load existing results if resuming
        completed = set()
        results = []
        if resume and results_path.exists():
            with open(results_path) as f:
                existing = json.load(f)
                results = existing
                completed = {
                    r["experiment_id"]
                    for r in existing
                    if r["status"] == "completed"
                }
            logger.info(f"Resuming: {len(completed)} experiments already completed")

        # Run remaining experiments
        remaining = [e for e in plan.experiments if e.experiment_id not in completed]
        logger.info(f"Running {len(remaining)} experiments...")

        for experiment in tqdm(remaining, desc="Training"):
            result = self.run_experiment_from_sweep(
                experiment=experiment,
                model_cls=model_cls,
                size_param=size_param,
                model_kwargs=model_kwargs,
                dataset_or_loader=dataset_or_loader,
                loss_fn=loss_fn,
                **train_kwargs,
            )
            results.append(result.to_dict())

            # Save incrementally
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

        logger.info(f"Completed {len(results)} total experiments")
        return results

    def save_results(self, results: list[TrainingResult | dict], path: str | Path | None = None):
        """Save results to JSON file."""
        path = Path(path) if path else self.output_dir / "results.json"
        
        # Convert to dicts if needed
        result_dicts = [
            r.to_dict() if isinstance(r, TrainingResult) else r
            for r in results
        ]
        
        with open(path, "w") as f:
            json.dump(result_dicts, f, indent=2)
        
        logger.info(f"Saved {len(result_dicts)} results to {path}")


@hydra.main(version_base=None, config_path="conf", config_name="trainer")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training runner.
    
    Uses Hydra for configuration management. Override defaults via CLI:
        uv run sl-train mode=mock
        uv run sl-train mode=local trainer.batch_size=32
    """
    logger.info(f"Starting training runner (mode={cfg.mode})...")
    
    runner = TrainingRunner(
        mode=cfg.mode,
        sweep_path=cfg.paths.sweep,
        output_dir=cfg.paths.output,
    )
    
    results = runner.run_sweep(resume=cfg.get("resume", True))
    
    # Summary
    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")
    logger.info(f"Training complete: {completed} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
