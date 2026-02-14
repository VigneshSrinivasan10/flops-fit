"""Tests for the sweep planner module."""

import pytest
import numpy as np

from flops_fit.planner import SweepPlanner, ExperimentConfig


class TestSweepPlanner:
    """Test suite for SweepPlanner."""
    
    def test_compute_flops(self):
        """Test FLOPs calculation: FLOPs = 6 * N * D."""
        planner = SweepPlanner()
        
        # 1B params, 100B tokens
        flops = planner.compute_flops(1_000_000_000, 100_000_000_000)
        assert flops == 6 * 1e9 * 1e11
    
    def test_tokens_for_compute(self):
        """Test token calculation for given compute and model size."""
        planner = SweepPlanner()
        
        # If C = 6e18 and N = 1e9, then D = 1e9
        compute = 6e18
        n_params = int(1e9)
        tokens = planner.tokens_for_compute(compute, n_params)
        
        # Should be approximately 1e9
        assert abs(tokens - 1e9) < 1e6  # Allow small rounding
    
    def test_generate_compute_budgets(self):
        """Test logarithmic spacing of compute budgets."""
        planner = SweepPlanner(
            min_flops=1e17,
            max_flops=1e21,
            num_compute_budgets=5,
        )
        
        budgets = planner.generate_compute_budgets()
        
        assert len(budgets) == 5
        assert budgets[0] == pytest.approx(1e17, rel=0.01)
        assert budgets[-1] == pytest.approx(1e21, rel=0.01)
        
        # Check logarithmic spacing
        log_budgets = np.log10(budgets)
        diffs = np.diff(log_budgets)
        assert np.allclose(diffs, diffs[0], rtol=0.01)
    
    def test_generate_sweep(self):
        """Test sweep generation produces valid configs."""
        planner = SweepPlanner(
            min_flops=1e17,
            max_flops=1e19,
            num_compute_budgets=3,
            num_model_sizes=3,
        )
        
        configs = list(planner.generate_sweep())
        
        # Should have multiple configs
        assert len(configs) > 0
        
        # Each config should be valid
        for config in configs:
            assert isinstance(config, ExperimentConfig)
            assert config.model_size > 0
            assert config.num_tokens > 0
            assert config.compute_budget > 0
            
            # Verify compute relationship holds
            expected_flops = 6 * config.model_size * config.num_tokens
            assert expected_flops == pytest.approx(config.compute_budget, rel=0.1)
    
    def test_experiment_config_tokens_per_param(self):
        """Test tokens_per_param calculation."""
        config = ExperimentConfig(
            experiment_id="test",
            compute_budget=6e18,
            model_size=1_000_000_000,  # 1B
            num_tokens=10_000_000_000,  # 10B
        )
        
        assert config.tokens_per_param == 10.0


class TestExperimentConfig:
    """Test suite for ExperimentConfig."""
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = ExperimentConfig(
            experiment_id="exp_001",
            compute_budget=1e18,
            model_size=100_000_000,
            num_tokens=1_000_000_000,
        )
        
        d = config.to_dict()
        
        assert d["experiment_id"] == "exp_001"
        assert d["compute_budget"] == 1e18
        assert d["model_size"] == 100_000_000
        assert d["num_tokens"] == 1_000_000_000
        assert "tokens_per_param" in d
