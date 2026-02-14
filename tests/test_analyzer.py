"""Tests for the scaling law analyzer module."""

import pytest
import numpy as np

from flops_fit.analyzer import ScalingLawAnalyzer, PowerLawFit


class TestPowerLawFit:
    """Test suite for PowerLawFit."""
    
    def test_predict(self):
        """Test power law prediction."""
        fit = PowerLawFit(
            name="test",
            coefficient_k=2.0,
            exponent_a=0.5,
            r_squared=0.99,
        )
        
        x = np.array([1, 4, 9, 16])
        y = fit.predict(x)
        
        # y = 2 * x^0.5 = 2 * sqrt(x)
        expected = 2 * np.sqrt(x)
        np.testing.assert_allclose(y, expected)
    
    def test_to_dict(self):
        """Test serialization."""
        fit = PowerLawFit(
            name="N_opt",
            coefficient_k=1e-8,
            exponent_a=0.73,
            r_squared=0.95,
        )
        
        d = fit.to_dict()
        
        assert d["name"] == "N_opt"
        assert d["coefficient_k"] == pytest.approx(1e-8)
        assert d["exponent_a"] == pytest.approx(0.73)
        assert "formula" in d


class TestScalingLawAnalyzer:
    """Test suite for ScalingLawAnalyzer."""
    
    def test_fit_power_law(self, tmp_path):
        """Test power law fitting."""
        # Create analyzer
        analyzer = ScalingLawAnalyzer(
            results_path=tmp_path / "results.json",
            output_dir=tmp_path / "analysis",
        )
        
        # Generate synthetic data: y = 0.1 * x^0.5
        x = np.logspace(10, 20, 20)
        y = 0.1 * np.power(x, 0.5) * np.random.uniform(0.95, 1.05, 20)
        
        fit = analyzer.fit_power_law(x, y, "test")
        
        # Should recover approximate exponent
        assert fit.exponent_a == pytest.approx(0.5, abs=0.1)
        assert fit.r_squared > 0.9
    
    def test_fit_power_law_handles_invalid(self, tmp_path):
        """Test that fitting handles invalid values gracefully."""
        analyzer = ScalingLawAnalyzer(
            results_path=tmp_path / "results.json",
            output_dir=tmp_path / "analysis",
        )
        
        # Include some invalid values
        x = np.array([1e10, 0, 1e12, -1, 1e14, np.nan])
        y = np.array([100, 50, 1000, 200, 10000, 500])
        
        fit = analyzer.fit_power_law(x, y, "test")
        
        # Should still produce a fit (using valid points only)
        assert fit.r_squared >= 0
