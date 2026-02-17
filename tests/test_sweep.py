"""Tests for flops_fit.sweep: sweep planning module."""

import pytest

from flops_fit.sweep import Experiment, SweepPlan, plan_sweep


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockModel:
    """A simple mock model matching the flops_fit contract.

    Same pattern as test_api.py: ``__init__(self, width=64, num_layers=4)``,
    ``num_params()`` returns ``width * num_layers * 100``.
    """

    def __init__(self, width=64, num_layers=4):
        self.width = width
        self.num_layers = num_layers

    def num_params(self):
        return self.width * self.num_layers * 100


class EvenOnlyModel:
    """Model that only accepts even width values."""

    def __init__(self, width=64, num_layers=4):
        if width % 2 != 0:
            raise ValueError(f"width must be even, got {width}")
        self.width = width
        self.num_layers = num_layers

    def num_params(self):
        return self.width * self.num_layers * 100


# ---------------------------------------------------------------------------
# 1. Experiment dataclass
# ---------------------------------------------------------------------------


class TestExperiment:
    def test_create_experiment(self):
        exp = Experiment(
            experiment_id="exp_0000",
            compute_budget=1e15,
            size_param_value=128,
            num_params=51200,
            num_tokens=1000000,
            tokens_per_param=19.53125,
        )
        assert exp.experiment_id == "exp_0000"
        assert exp.compute_budget == 1e15
        assert exp.size_param_value == 128
        assert exp.num_params == 51200
        assert exp.num_tokens == 1000000
        assert exp.tokens_per_param == pytest.approx(19.53125)


# ---------------------------------------------------------------------------
# 2. SweepPlan dataclass
# ---------------------------------------------------------------------------


class TestSweepPlan:
    def test_sweep_plan_properties(self):
        experiments = [
            Experiment("exp_0000", 1e15, 128, 51200, 1000000, 19.53),
            Experiment("exp_0001", 1e16, 256, 102400, 10000000, 97.66),
        ]
        plan = SweepPlan(
            experiments=experiments,
            model_cls_name="MockModel",
            size_param="width",
            compute_budgets=[1e15, 1e16],
            model_kwargs={"num_layers": 4},
        )
        assert plan.total_flops == pytest.approx(1e15 + 1e16)
        assert plan.num_experiments == 2
        assert plan.compute_budgets == [1e15, 1e16]
        assert plan.model_cls_name == "MockModel"
        assert plan.size_param == "width"

    def test_sweep_plan_repr(self):
        experiments = [
            Experiment("exp_0000", 1e15, 128, 51200, 1000000, 19.53),
        ]
        plan = SweepPlan(
            experiments=experiments,
            model_cls_name="MockModel",
            size_param="width",
            compute_budgets=[1e15],
            model_kwargs={},
        )
        r = repr(plan)
        assert "1" in r  # experiment count
        assert "1" in r  # budget count
        assert "e+" in r.lower() or "E+" in r  # total_flops in scientific notation


# ---------------------------------------------------------------------------
# 3. plan_sweep() basic
# ---------------------------------------------------------------------------


class TestPlanSweepBasic:
    def test_returns_sweep_plan(self):
        plan = plan_sweep(
            model_cls=MockModel,
            size_param="width",
            model_kwargs={"num_layers": 4},
            compute_budgets=[1e15, 1e16],
        )
        assert isinstance(plan, SweepPlan)

    def test_has_experiments(self):
        plan = plan_sweep(
            model_cls=MockModel,
            size_param="width",
            model_kwargs={"num_layers": 4},
            compute_budgets=[1e15, 1e16],
        )
        assert len(plan.experiments) > 0

    def test_experiments_have_valid_budgets(self):
        budgets = [1e15, 1e16]
        plan = plan_sweep(
            model_cls=MockModel,
            size_param="width",
            model_kwargs={"num_layers": 4},
            compute_budgets=budgets,
        )
        for exp in plan.experiments:
            assert exp.compute_budget in budgets

    def test_experiments_have_positive_values(self):
        plan = plan_sweep(
            model_cls=MockModel,
            size_param="width",
            model_kwargs={"num_layers": 4},
            compute_budgets=[1e15, 1e16],
        )
        for exp in plan.experiments:
            assert exp.num_tokens > 0
            assert exp.num_params > 0


# ---------------------------------------------------------------------------
# 4. plan_sweep() size probing
# ---------------------------------------------------------------------------


class TestPlanSweepProbing:
    def test_different_sizes_within_budget(self):
        """Experiments within a single budget should have different size_param_values."""
        plan = plan_sweep(
            model_cls=MockModel,
            size_param="width",
            model_kwargs={"num_layers": 4},
            compute_budgets=[1e16],
        )
        budget_exps = [e for e in plan.experiments if e.compute_budget == 1e16]
        sizes = [e.size_param_value for e in budget_exps]
        assert len(set(sizes)) > 1, "Should have multiple distinct size_param_values"

    def test_num_params_matches_mock(self):
        """num_params in experiments should match what MockModel returns."""
        plan = plan_sweep(
            model_cls=MockModel,
            size_param="width",
            model_kwargs={"num_layers": 4},
            compute_budgets=[1e16],
        )
        for exp in plan.experiments:
            expected = exp.size_param_value * 4 * 100  # MockModel formula
            assert exp.num_params == expected


# ---------------------------------------------------------------------------
# 5. plan_sweep() feasibility filtering
# ---------------------------------------------------------------------------


class TestPlanSweepFeasibility:
    def test_infeasible_configs_filtered(self):
        """Very small compute budget should filter out large models."""
        plan = plan_sweep(
            model_cls=MockModel,
            size_param="width",
            model_kwargs={"num_layers": 4},
            compute_budgets=[1e8],
        )
        for exp in plan.experiments:
            assert exp.tokens_per_param >= 0.1, (
                f"Infeasible experiment not filtered: "
                f"tokens_per_param={exp.tokens_per_param}"
            )


# ---------------------------------------------------------------------------
# 6. plan_sweep() skips invalid sizes
# ---------------------------------------------------------------------------


class TestPlanSweepInvalidSizes:
    def test_skips_invalid_size_values(self):
        """Model that rejects odd widths should not crash plan_sweep."""
        plan = plan_sweep(
            model_cls=EvenOnlyModel,
            size_param="width",
            model_kwargs={"num_layers": 4},
            compute_budgets=[1e15, 1e16],
        )
        assert isinstance(plan, SweepPlan)
        # Should still have some experiments (even-valued sizes work)
        assert len(plan.experiments) > 0

    def test_invalid_sizes_produce_fewer_experiments(self):
        """EvenOnlyModel should produce <= experiments compared to MockModel."""
        plan_mock = plan_sweep(
            model_cls=MockModel,
            size_param="width",
            model_kwargs={"num_layers": 4},
            compute_budgets=[1e16],
        )
        plan_even = plan_sweep(
            model_cls=EvenOnlyModel,
            size_param="width",
            model_kwargs={"num_layers": 4},
            compute_budgets=[1e16],
        )
        assert plan_even.num_experiments <= plan_mock.num_experiments


# ---------------------------------------------------------------------------
# 7. plan_sweep() optional params
# ---------------------------------------------------------------------------


class TestPlanSweepOptionalParams:
    def test_num_sizes_per_budget(self):
        """num_sizes_per_budget controls max experiments per budget."""
        plan = plan_sweep(
            model_cls=MockModel,
            size_param="width",
            model_kwargs={"num_layers": 4},
            compute_budgets=[1e16],
            num_sizes_per_budget=3,
        )
        budget_exps = [e for e in plan.experiments if e.compute_budget == 1e16]
        assert len(budget_exps) <= 3

    def test_min_max_size(self):
        """min_size and max_size constrain the probed size range."""
        plan = plan_sweep(
            model_cls=MockModel,
            size_param="width",
            model_kwargs={"num_layers": 4},
            compute_budgets=[1e16],
            min_size=128,
            max_size=1024,
        )
        for exp in plan.experiments:
            assert exp.size_param_value >= 128
            assert exp.size_param_value <= 1024

    def test_flops_per_param_per_token(self):
        """Custom flops_per_param_per_token changes token calculation."""
        plan_default = plan_sweep(
            model_cls=MockModel,
            size_param="width",
            model_kwargs={"num_layers": 4},
            compute_budgets=[1e16],
        )
        plan_custom = plan_sweep(
            model_cls=MockModel,
            size_param="width",
            model_kwargs={"num_layers": 4},
            compute_budgets=[1e16],
            flops_per_param_per_token=12,
        )
        # Higher flops_per_param_per_token -> fewer tokens per experiment
        # (for same budget and same model size)
        # Find matching size_param_values to compare
        default_by_size = {e.size_param_value: e for e in plan_default.experiments}
        custom_by_size = {e.size_param_value: e for e in plan_custom.experiments}
        common_sizes = set(default_by_size.keys()) & set(custom_by_size.keys())
        assert len(common_sizes) > 0, "Should have at least one common size"
        for sv in common_sizes:
            assert custom_by_size[sv].num_tokens < default_by_size[sv].num_tokens


# ---------------------------------------------------------------------------
# 8. SweepPlan repr
# ---------------------------------------------------------------------------


class TestSweepPlanRepr:
    def test_repr_contains_key_info(self):
        plan = plan_sweep(
            model_cls=MockModel,
            size_param="width",
            model_kwargs={"num_layers": 4},
            compute_budgets=[1e15, 1e16],
        )
        r = repr(plan)
        assert str(plan.num_experiments) in r
        assert "2" in r  # 2 budgets
        assert "e+" in r.lower() or "E+" in r  # total_flops
