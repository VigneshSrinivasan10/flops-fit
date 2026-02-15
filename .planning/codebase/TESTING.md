# Testing Patterns

**Analysis Date:** 2026-02-15

## Test Framework

**Runner:**
- pytest >= 8.0.0
- Config: `pyproject.toml` under `[tool.pytest.ini_options]`

**Assertion Library:**
- pytest built-ins (`assert`, `pytest.approx`, `pytest.raises` implied)
- `numpy.testing.assert_allclose` for array comparisons

**Coverage:**
- pytest-cov >= 4.1.0 (installed as dev dependency)

**Run Commands:**
```bash
uv run pytest                       # Run all tests
uv run pytest tests/test_planner.py # Run single file
uv run pytest --cov=flops_fit       # With coverage
```

**Configuration (pyproject.toml):**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

## Test File Organization

**Location:**
- Separate `tests/` directory at project root (not co-located with source)

**Naming:**
- `test_<module>.py` mirroring source module names: `test_analyzer.py`, `test_planner.py`

**Structure:**
```
tests/
├── __init__.py           # Empty package marker
├── test_analyzer.py      # Tests for src/flops_fit/analyzer.py
└── test_planner.py       # Tests for src/flops_fit/planner.py
```

Note: `model.py`, `trainer.py`, and `visualizer.py` have no test files.

## Test Structure

**Suite Organization:**
```python
class TestClassName:
    """Test suite for ClassName."""

    def test_method_name(self):
        """Test description."""
        # arrange
        obj = ClassName(...)

        # act
        result = obj.method(...)

        # assert
        assert result == expected
```

Classes group tests by the class under test. Test method names are descriptive and match `test_<what_is_tested>`.

**Pattern from `tests/test_planner.py`:**
```python
class TestSweepPlanner:
    def test_compute_flops(self):
        planner = SweepPlanner()
        flops = planner.compute_flops(1_000_000_000, 100_000_000_000)
        assert flops == 6 * 1e9 * 1e11

    def test_generate_sweep(self):
        planner = SweepPlanner(
            min_flops=1e17,
            max_flops=1e19,
            num_compute_budgets=3,
            num_model_sizes=3,
        )
        configs = list(planner.generate_sweep())
        assert len(configs) > 0
        for config in configs:
            assert isinstance(config, ExperimentConfig)
            assert config.model_size > 0

class TestExperimentConfig:
    def test_to_dict(self):
        config = ExperimentConfig(...)
        d = config.to_dict()
        assert d["experiment_id"] == "exp_001"
```

**Fixtures Used:**
- `tmp_path` (pytest built-in) is the only fixture used, for `ScalingLawAnalyzer` tests that require a path argument

**No custom fixtures defined.** No `conftest.py` present.

## Mocking

**Framework:** None. No `unittest.mock`, `pytest-mock`, or other mocking libraries are used.

**Strategy:** Tests instantiate real objects with minimal or temporary configurations. The `tmp_path` fixture provides isolated temporary directories for path-dependent classes. Randomness in `_mock_train` is not seeded in tests (a potential source of test flakiness for numeric assertions).

**What NOT to Mock:**
- File I/O — tests use `tmp_path` and real file operations
- Numpy/scipy — tested against real computed values

## Fixtures and Factories

**Test Data:**
- Data generated inline in each test using `np.logspace`, `np.array`, and constructed dataclass instances
- No shared fixtures or factory functions
- Pattern for numeric data in `tests/test_analyzer.py`:
```python
# Generate synthetic data: y = 0.1 * x^0.5
x = np.logspace(10, 20, 20)
y = 0.1 * np.power(x, 0.5) * np.random.uniform(0.95, 1.05, 20)
```

**Location:**
- No separate fixtures file. Test data is created inline.

## Coverage

**Requirements:** None enforced (no `--cov-fail-under` set)

**View Coverage:**
```bash
uv run pytest --cov=flops_fit --cov-report=term-missing
```

**Gaps:**
- `src/flops_fit/model.py` — no tests
- `src/flops_fit/trainer.py` — no tests
- `src/flops_fit/visualizer.py` — no tests
- `ScalingLawAnalyzer.analyze()`, `load_results()`, `save_analysis()`, `predict()` — not directly tested
- `ScalingLawAnalyzer.find_optimal_per_budget()` — not tested
- `SweepPlanner.save_sweep()` — not tested
- Hydra `main()` functions — not tested

## Test Types

**Unit Tests:**
- All existing tests are unit tests
- Scope: individual methods on a single class
- No external services, no database, no HTTP calls

**Integration Tests:**
- None present

**E2E Tests:**
- Not used

## Common Patterns

**Float Comparison:**
```python
# Use pytest.approx for scalar floats
assert d["exponent_a"] == pytest.approx(0.73)
assert budgets[0] == pytest.approx(1e17, rel=0.01)

# Use numpy.testing for arrays
np.testing.assert_allclose(y, expected)

# Use abs tolerance for approximate recovery
assert fit.exponent_a == pytest.approx(0.5, abs=0.1)
```

**Range and Property Assertions:**
```python
assert fit.r_squared > 0.9
assert len(configs) > 0
assert config.model_size > 0
```

**Iteration/Collection Testing:**
```python
configs = list(planner.generate_sweep())
for config in configs:
    assert isinstance(config, ExperimentConfig)
    assert config.compute_budget > 0
```

**Robustness Testing (invalid input):**
```python
def test_fit_power_law_handles_invalid(self, tmp_path):
    # Include some invalid values
    x = np.array([1e10, 0, 1e12, -1, 1e14, np.nan])
    y = np.array([100, 50, 1000, 200, 10000, 500])
    fit = analyzer.fit_power_law(x, y, "test")
    assert fit.r_squared >= 0
```

**Serialization Testing:**
```python
def test_to_dict(self):
    d = obj.to_dict()
    assert d["key"] == expected_value
    assert "required_key" in d
```

---

*Testing analysis: 2026-02-15*
