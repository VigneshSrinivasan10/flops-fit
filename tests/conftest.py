"""Shared test fixtures for flops-fit pipeline tests."""

import json

import pytest


@pytest.fixture
def sample_sweep_configs():
    """Minimal valid sweep config list matching ExperimentConfig.to_dict() schema.

    Returns 12 configs: 3 compute budgets (1e17, 1e18, 1e19) x 4 model sizes each.
    Uses exact powers of 10 for compute budgets to avoid bucket rounding issues.
    Ensures 6 * model_size * num_tokens approximately equals compute_budget.
    """
    configs = []
    compute_budgets = [1e17, 1e18, 1e19]
    # For each compute budget, pick 4 model sizes that make sense
    model_sizes_per_budget = {
        1e17: [10_000_000, 30_000_000, 100_000_000, 300_000_000],
        1e18: [30_000_000, 100_000_000, 300_000_000, 1_000_000_000],
        1e19: [100_000_000, 300_000_000, 1_000_000_000, 3_000_000_000],
    }

    idx = 0
    for budget in compute_budgets:
        for model_size in model_sizes_per_budget[budget]:
            num_tokens = int(budget / (6 * model_size))
            tokens_per_param = num_tokens / model_size
            configs.append(
                {
                    "experiment_id": f"exp_{idx:04d}",
                    "compute_budget": budget,
                    "model_size": model_size,
                    "num_tokens": num_tokens,
                    "tokens_per_param": tokens_per_param,
                }
            )
            idx += 1

    return configs


@pytest.fixture
def sweep_json(tmp_path, sample_sweep_configs):
    """Write sample sweep configs to a temp JSON file and return the path."""
    path = tmp_path / "sweep.json"
    with open(path, "w") as f:
        json.dump(sample_sweep_configs, f, indent=2)
    return path


@pytest.fixture
def sample_results():
    """Completed training result dicts matching TrainingResult.to_dict() schema.

    Returns 12 results corresponding to sample_sweep_configs.
    Uses deterministic loss values derived from a simplified scaling law (no randomness).
    Loss = (4e8 / model_size)^0.34 + (2e10 / num_tokens)^0.28 + 1.69
    """
    compute_budgets = [1e17, 1e18, 1e19]
    model_sizes_per_budget = {
        1e17: [10_000_000, 30_000_000, 100_000_000, 300_000_000],
        1e18: [30_000_000, 100_000_000, 300_000_000, 1_000_000_000],
        1e19: [100_000_000, 300_000_000, 1_000_000_000, 3_000_000_000],
    }

    results = []
    idx = 0
    for budget in compute_budgets:
        for model_size in model_sizes_per_budget[budget]:
            num_tokens = int(budget / (6 * model_size))
            # Deterministic loss from simplified scaling law (no noise)
            N_c = 4e8
            D_c = 2e10
            alpha = 0.34
            beta = 0.28
            L_inf = 1.69
            loss = (N_c / model_size) ** alpha + (D_c / num_tokens) ** beta + L_inf

            results.append(
                {
                    "experiment_id": f"exp_{idx:04d}",
                    "compute_budget": budget,
                    "model_size": model_size,
                    "num_tokens": num_tokens,
                    "final_loss": loss,
                    "actual_flops": budget * 1.0,  # Exact match for determinism
                    "wall_time_seconds": 0.1 * (idx + 1),
                    "timestamp": "2026-01-01T00:00:00",
                    "status": "completed",
                    "error_message": None,
                }
            )
            idx += 1

    return results


@pytest.fixture
def results_json(tmp_path, sample_results):
    """Write sample results to a temp JSON file and return the path."""
    path = tmp_path / "results.json"
    with open(path, "w") as f:
        json.dump(sample_results, f, indent=2)
    return path


@pytest.fixture
def sample_analysis():
    """ScalingAnalysis.to_dict() schema with realistic power law values."""
    return {
        "n_opt_fit": {
            "name": "N_opt",
            "coefficient_k": 0.01,
            "exponent_a": 0.5,
            "r_squared": 0.98,
            "k_ci": None,
            "a_ci": None,
            "formula": "N_opt = 1.0000e-02 * C^0.5000",
        },
        "d_opt_fit": {
            "name": "D_opt",
            "coefficient_k": 0.1,
            "exponent_a": 0.5,
            "r_squared": 0.97,
            "k_ci": None,
            "a_ci": None,
            "formula": "D_opt = 1.0000e-01 * C^0.5000",
        },
        "l_opt_fit": {
            "name": "L_opt",
            "coefficient_k": 100.0,
            "exponent_a": -0.1,
            "r_squared": 0.95,
            "k_ci": None,
            "a_ci": None,
            "formula": "L_opt = 1.0000e+02 * C^-0.1000",
        },
        "optimal_points": [
            {"compute_budget": 1e17, "model_size": 31622776, "num_tokens": 316227760, "final_loss": 5.0},
            {"compute_budget": 1e18, "model_size": 100000000, "num_tokens": 1000000000, "final_loss": 4.0},
            {"compute_budget": 1e19, "model_size": 316227766, "num_tokens": 3162277660, "final_loss": 3.2},
        ],
        "optimal_ratio": 10.0,
    }


@pytest.fixture
def analysis_json(tmp_path, sample_analysis):
    """Write sample analysis to a temp JSON file and return the path."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    path = analysis_dir / "scaling_laws.json"
    with open(path, "w") as f:
        json.dump(sample_analysis, f, indent=2)
    return path
