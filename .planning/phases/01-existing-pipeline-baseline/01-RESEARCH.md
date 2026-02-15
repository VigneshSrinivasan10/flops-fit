# Phase 1: Existing Pipeline Baseline - Research

**Researched:** 2026-02-15
**Domain:** Python testing, Hydra configuration testing, matplotlib plot testing, pipeline characterization
**Confidence:** HIGH

## Summary

Phase 1 is about locking down the existing 4-stage pipeline (plan, train, analyze, visualize) with comprehensive tests before any refactoring begins. The codebase is a well-structured linear pipeline with file-based JSON handoff between stages, Hydra configuration management, and a GPT model with u-mup parameterization. The code already exists and works, but test coverage is severely lacking: only `planner.py` and `analyzer.py` have partial unit tests. `trainer.py`, `visualizer.py`, and `model.py` have zero tests. There are no integration tests covering the full pipeline flow.

The existing codebase has several documented concerns (bucket rounding mismatch between analyzer and visualizer, unseeded mock randomness, stale command names in docstrings, unconsumed config options) that tests should capture and characterize. The phase goal is NOT to fix these issues but to write tests that document current behavior -- creating a safety net for future refactoring.

**Primary recommendation:** Write characterization tests that assert current behavior (even if imperfect), then add targeted unit tests for each module's public API, and finally an integration test for the full plan-train-analyze-visualize pipeline in mock mode.

## Standard Stack

### Core (Already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | >= 8.0.0 | Test runner and assertions | Already configured in pyproject.toml |
| pytest-cov | >= 4.1.0 | Coverage reporting | Already in dev dependencies |
| numpy | >= 1.26.0 | Numeric assertions via `numpy.testing` | Already a project dependency |

### Supporting (Needed for this phase)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest (tmp_path) | built-in | Isolated temp directories for file I/O tests | All tests that read/write JSON files |
| pytest (monkeypatch) | built-in | Override Hydra config, seed random state | Testing CLI main() functions |
| matplotlib | >= 3.8.0 | Already installed; test plot generation | Visualizer tests check files are created |

### Not Needed
| Library | Why Not |
|---------|---------|
| pytest-mock / unittest.mock | Existing pattern uses real objects with tmp_path; keep consistent |
| pytest-hydra / hydra test utilities | Hydra provides `initialize_config_dir` and `compose` for testing; no extra package needed |
| pytest-mpl | Overkill for this phase; just verify files are created and have nonzero size |

**Installation:**
```bash
uv sync --dev  # Already has pytest, pytest-cov in dev deps
```

## Architecture Patterns

### Test File Structure
```
tests/
├── __init__.py              # Existing empty marker
├── conftest.py              # NEW: shared fixtures (mock results, sweep configs)
├── test_planner.py          # EXISTING: expand with save_sweep, edge cases
├── test_trainer.py          # NEW: mock training, resume logic, sweep loading
├── test_analyzer.py         # EXISTING: expand with analyze(), predict(), find_optimal
├── test_visualizer.py       # NEW: plot generation, file output, style handling
├── test_model.py            # NEW: forward pass, param count, u-mup init, config
└── test_pipeline.py         # NEW: end-to-end plan->train->analyze->visualize
```

### Pattern 1: Characterization Tests
**What:** Tests that capture current behavior exactly, even if the behavior has known issues.
**When to use:** When the goal is to prevent regressions during refactoring, not to fix bugs.
**Example:**
```python
def test_analyzer_uses_2_decimal_bucket_rounding(tmp_path):
    """Characterize: analyzer rounds log10(compute) to 2 decimals.
    Note: visualizer uses 1 decimal -- this is a known inconsistency."""
    analyzer = ScalingLawAnalyzer(results_path=tmp_path / "r.json", output_dir=tmp_path)
    df = pd.DataFrame({"compute_budget": [1.01e14, 1.04e14], ...})
    df_opt = analyzer.find_optimal_per_budget(df)
    # Current behavior: these are separate buckets at 2-decimal precision
    assert len(df_opt) == 2
```

### Pattern 2: Fixture-Based Pipeline Data
**What:** Shared pytest fixtures that produce valid intermediate pipeline data (sweep JSON, results JSON, analysis JSON).
**When to use:** Multiple test modules need the same realistic input data.
**Example:**
```python
# conftest.py
@pytest.fixture
def sample_sweep_configs():
    """Minimal valid sweep config list matching ExperimentConfig.to_dict() format."""
    return [
        {
            "experiment_id": "exp_0000",
            "compute_budget": 1e17,
            "model_size": 10_000_000,
            "num_tokens": 1_666_666,
            "tokens_per_param": 0.1666,
        },
        # ... more configs
    ]

@pytest.fixture
def sweep_json(tmp_path, sample_sweep_configs):
    """Write sweep configs to a temp JSON file."""
    path = tmp_path / "sweep.json"
    with open(path, "w") as f:
        json.dump(sample_sweep_configs, f)
    return path
```

### Pattern 3: Testing Hydra main() Functions
**What:** Use `hydra.initialize_config_dir` + `hydra.compose` to test Hydra-decorated functions without subprocess invocation.
**When to use:** Testing the CLI entry points that use `@hydra.main`.
**Example:**
```python
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

def test_planner_main_produces_sweep_json(tmp_path):
    """Test ff-plan CLI entry point produces valid output."""
    config_dir = str(Path(__file__).parent.parent / "src" / "flops_fit" / "conf")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="planner", overrides=[
            f"output.sweep_path={tmp_path}/sweep.json",
            "compute.num_budgets=3",
            "compute.num_model_sizes=3",
        ])
        # Call the logic directly (not the decorated main)
        planner = SweepPlanner(
            min_flops=cfg.compute.min_flops,
            max_flops=cfg.compute.max_flops,
            num_compute_budgets=cfg.compute.num_budgets,
            num_model_sizes=cfg.compute.num_model_sizes,
            min_model_size=cfg.model.min_size,
            max_model_size=cfg.model.max_size,
        )
        configs = planner.save_sweep(cfg.output.sweep_path)
    assert len(configs) > 0
    assert (tmp_path / "sweep.json").exists()
```

**Important Hydra testing note:** Hydra's `@hydra.main` decorator changes the working directory by default. The project configs set `hydra.run.dir: .` to prevent this, but tests should still use `initialize_config_dir` + `compose` rather than calling `main()` directly to avoid Hydra's global state issues. Each test using Hydra config composition must use the `initialize_config_dir` context manager, which handles cleanup.

### Pattern 4: Seeded Mock Training Tests
**What:** Seed numpy random state before mock training to get deterministic results.
**When to use:** Any test involving `TrainingRunner._mock_train`.
**Example:**
```python
def test_mock_train_produces_reasonable_loss(self):
    np.random.seed(42)
    runner = TrainingRunner(mode="mock", sweep_path="dummy", output_dir=tmp_path)
    loss, flops, time = runner._mock_train(
        model_size=10_000_000, num_tokens=100_000_000, compute_budget=6e15
    )
    assert 1.5 < loss < 5.0  # Reasonable range for LM loss
    assert flops > 0
    assert time > 0
```

### Anti-Patterns to Avoid
- **Testing Hydra main() via subprocess:** Slow, fragile, hard to debug. Use `initialize_config_dir` + `compose` instead.
- **Asserting exact float values from mock training:** The mock uses random noise; assert ranges, not exact values.
- **Testing matplotlib rendering quality:** Just check files exist and have nonzero size. Image comparison testing is fragile across environments.
- **Fixing bugs in tests:** Phase 1 characterizes behavior, not fixes. Mark known issues with comments but assert current behavior.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Temp directories for tests | Manual mkdir/cleanup | `pytest.fixture` `tmp_path` | Auto-cleanup, isolated per test |
| Float comparison | Manual `abs(a-b) < epsilon` | `pytest.approx(val, rel=0.01)` | Handles edge cases, readable |
| Array comparison | Manual loops | `numpy.testing.assert_allclose` | Proper broadcasting, error messages |
| Config composition for tests | Parsing YAML manually | `hydra.compose()` | Respects defaults, overrides, types |
| Test data generation | Copy-paste JSON literals | Shared fixtures in `conftest.py` | DRY, consistent across test modules |

## Common Pitfalls

### Pitfall 1: Hydra GlobalHydra State Leaks
**What goes wrong:** Hydra maintains global state. If one test initializes Hydra and doesn't clean up, subsequent tests fail with "GlobalHydra is already initialized".
**Why it happens:** `hydra.initialize()` sets global state; multiple tests calling it without context manager cleanup cause conflicts.
**How to avoid:** Always use `initialize_config_dir` as a context manager (`with initialize_config_dir(...):`) which handles cleanup. Alternatively, call `GlobalHydra.instance().clear()` in teardown.
**Warning signs:** Tests pass individually but fail when run together.

### Pitfall 2: Matplotlib Backend Issues in CI/Tests
**What goes wrong:** `plt.show()` or interactive backends cause tests to hang or crash in headless environments.
**Why it happens:** Default matplotlib backend may be Tk/Qt which requires display.
**How to avoid:** Use `matplotlib.use("Agg")` at the top of test files or in `conftest.py`. The visualizer code already doesn't call `plt.show()` by default (controlled by config), but `_setup_style` mutates global `plt.rcParams`.
**Warning signs:** Tests hang indefinitely or fail with "no display" errors.

### Pitfall 3: Working Directory Changes from Hydra
**What goes wrong:** Hydra's `@hydra.main` changes `os.getcwd()` by default, breaking relative path resolution.
**Why it happens:** Hydra creates per-run output directories and `chdir`s into them.
**How to avoid:** The project already sets `hydra.run.dir: .` and `hydra.output_subdir: null` in all configs, which prevents this. Tests should verify this behavior is preserved by using absolute paths or `tmp_path`.
**Warning signs:** FileNotFoundError on relative paths that work outside tests.

### Pitfall 4: Non-Deterministic Mock Results
**What goes wrong:** `TrainingRunner._mock_train` uses `np.random.normal` and `np.random.uniform` without seeding, producing different results each run.
**Why it happens:** No seed is set despite `trainer.seed: 42` being in config.
**How to avoid:** In tests, always `np.random.seed(42)` before calling mock training functions. Assert value ranges, not exact values.
**Warning signs:** Tests that occasionally fail on CI with numeric comparison errors.

### Pitfall 5: Asserting on Bucket-Sensitive Analysis
**What goes wrong:** Tests that depend on compute budget bucketing may produce different bucket counts depending on floating-point rounding.
**Why it happens:** `np.round(np.log10(value), 2)` is sensitive to floating-point representation of log values.
**How to avoid:** Use compute budgets that are exact powers of 10 (1e17, 1e18, etc.) in test data to ensure clean bucket boundaries.
**Warning signs:** Test produces different number of optimal points on different platforms.

## Code Examples

### Example 1: Testing TrainingRunner Resume Logic
```python
def test_resume_skips_completed_experiments(tmp_path):
    """Verify resume mode skips experiments already in results.json."""
    # Create sweep with 3 experiments
    sweep = [
        {"experiment_id": f"exp_{i:04d}", "compute_budget": 1e17,
         "model_size": 10_000_000, "num_tokens": 1_666_666}
        for i in range(3)
    ]
    sweep_path = tmp_path / "sweep.json"
    with open(sweep_path, "w") as f:
        json.dump(sweep, f)

    # Pre-populate results with first experiment completed
    results_path = tmp_path / "results.json"
    existing = [{
        "experiment_id": "exp_0000", "compute_budget": 1e17,
        "model_size": 10_000_000, "num_tokens": 1_666_666,
        "final_loss": 2.5, "actual_flops": 1e17,
        "wall_time_seconds": 0.1, "timestamp": "2026-01-01",
        "status": "completed", "error_message": None,
    }]
    with open(results_path, "w") as f:
        json.dump(existing, f)

    np.random.seed(42)
    runner = TrainingRunner(mode="mock", sweep_path=sweep_path, output_dir=tmp_path)
    results = runner.run_sweep(resume=True)

    # Should have 3 total results (1 existing + 2 new)
    assert len(results) == 3
    # First should be the pre-existing one
    assert results[0]["experiment_id"] == "exp_0000"
    assert results[0]["final_loss"] == 2.5  # Unchanged
```

### Example 2: Testing Visualizer Creates Plot Files
```python
import matplotlib
matplotlib.use("Agg")

def test_plot_isoflops_creates_file(tmp_path, sample_results_json, sample_analysis_json):
    """Verify IsoFLOPs plot is saved as PNG."""
    visualizer = ScalingVisualizer(
        results_path=sample_results_json,
        analysis_path=sample_analysis_json,
        output_dir=tmp_path / "plots",
    )
    fig = visualizer.plot_isoflops(save=True)
    plot_path = tmp_path / "plots" / "isoflops_curves.png"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    plt.close(fig)  # Clean up matplotlib state
```

### Example 3: Testing GPT Model Forward Pass
```python
import torch

def test_gpt_forward_pass_shape():
    """Verify GPT produces correct output shape."""
    config = GPTConfig(vocab_size=100, d_model=64, num_layers=2, num_heads=2, max_seq_len=32)
    model = GPT(config)
    input_ids = torch.randint(0, 100, (2, 16))  # batch=2, seq=16
    logits, loss = model(input_ids)
    assert logits.shape == (2, 16, 100)  # (batch, seq, vocab)
    assert loss is None  # No labels provided

def test_gpt_forward_with_labels():
    """Verify GPT computes loss when labels provided."""
    config = GPTConfig(vocab_size=100, d_model=64, num_layers=2, num_heads=2, max_seq_len=32)
    model = GPT(config)
    input_ids = torch.randint(0, 100, (2, 16))
    labels = torch.randint(0, 100, (2, 16))
    logits, loss = model(input_ids, labels=labels)
    assert loss is not None
    assert loss.item() > 0
    assert torch.isfinite(loss)
```

### Example 4: Full Pipeline Integration Test
```python
def test_full_pipeline_mock_mode(tmp_path):
    """End-to-end: plan -> train(mock) -> analyze -> visualize."""
    np.random.seed(42)

    # Step 1: Plan
    planner = SweepPlanner(
        min_flops=1e17, max_flops=1e19,
        num_compute_budgets=3, num_model_sizes=4,
        min_model_size=10_000_000, max_model_size=1_000_000_000,
    )
    sweep_path = tmp_path / "sweep.json"
    configs = planner.save_sweep(sweep_path)
    assert len(configs) > 0

    # Step 2: Train (mock)
    runner = TrainingRunner(mode="mock", sweep_path=sweep_path, output_dir=tmp_path)
    results = runner.run_sweep(resume=False)
    completed = [r for r in results if r["status"] == "completed"]
    assert len(completed) == len(configs)

    # Step 3: Analyze
    analyzer = ScalingLawAnalyzer(
        results_path=tmp_path / "results.json",
        output_dir=tmp_path / "analysis",
    )
    analysis = analyzer.analyze()
    assert analysis.n_opt_fit.r_squared > 0
    assert analysis.d_opt_fit.r_squared > 0
    assert analysis.l_opt_fit.r_squared > 0

    # Step 4: Visualize
    matplotlib.use("Agg")
    visualizer = ScalingVisualizer(
        results_path=tmp_path / "results.json",
        analysis_path=tmp_path / "analysis" / "scaling_laws.json",
        output_dir=tmp_path / "plots",
    )
    figures = visualizer.plot_all(save=True)
    assert len(figures) == 3  # isoflops, scaling_laws, tokens_per_param
    for fig in figures:
        plt.close(fig)

    # Verify output files
    assert (tmp_path / "plots" / "isoflops_curves.png").exists()
    assert (tmp_path / "plots" / "scaling_laws.png").exists()
    assert (tmp_path / "plots" / "tokens_per_param.png").exists()
```

## Codebase-Specific Testing Concerns

### What Must Be Tested (from CONCERNS.md audit)

| Module | Critical Test Gaps | Priority |
|--------|-------------------|----------|
| `trainer.py` | `run_sweep`, resume logic, mock train, `load_sweep` | HIGH |
| `model.py` | Forward pass, loss computation, u-mup init, param counting | HIGH |
| `visualizer.py` | Plot generation, file output, data loading | MEDIUM |
| `analyzer.py` | `analyze()`, `predict()`, `find_optimal_per_budget()` | HIGH |
| `planner.py` | `save_sweep()` file I/O | MEDIUM |
| Pipeline | End-to-end plan->train->analyze->visualize | MEDIUM |
| Hydra CLI | Config loading, overrides, preset loading | LOW |

### Known Bugs to Characterize (Not Fix)

1. **Bucket rounding mismatch:** Analyzer uses 2-decimal rounding, visualizer uses 1-decimal. Write tests that document both behaviors.
2. **Unseeded mock randomness:** `_mock_train` ignores `trainer.seed`. Tests should seed manually and document the gap.
3. **`create_model_for_scaling` parameter overshoot:** Uses `12*L*d^2` approximation but actual is `16*L*d^2`. Write a test that documents the actual vs approximate ratio.
4. **Stale `sl-*` command names in docstrings:** Not testable directly but worth noting in test comments.

### JSON Schema Contracts Between Stages

The pipeline relies on implicit JSON schemas for inter-stage communication. Tests should verify these contracts:

| Stage Output | Key Fields | Consumed By |
|-------------|------------|-------------|
| `sweep.json` | `experiment_id`, `compute_budget`, `model_size`, `num_tokens` | TrainingRunner.load_sweep |
| `results.json` | Above + `final_loss`, `actual_flops`, `wall_time_seconds`, `status`, `timestamp`, `error_message` | ScalingLawAnalyzer.load_results, ScalingVisualizer.load_data |
| `scaling_laws.json` | `n_opt_fit.{coefficient_k, exponent_a, r_squared}`, same for `d_opt_fit`, `l_opt_fit`, `optimal_points`, `optimal_ratio` | ScalingVisualizer.load_data, ScalingLawAnalyzer.predict |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `pytest.fixture` with `yield` | `tmp_path` built-in fixture | pytest 3.9+ | No manual cleanup needed |
| `hydra.experimental.initialize` | `hydra.initialize_config_dir` | Hydra 1.2+ | Stable API for test config composition |
| `matplotlib.pyplot.savefig` testing | Check file exists + nonzero size | N/A | Avoid fragile pixel comparison |

## Open Questions

1. **Should Hydra main() functions be tested directly or only the underlying classes?**
   - What we know: Testing `@hydra.main` decorated functions is awkward due to global state. The project separates logic into classes that can be tested independently.
   - Recommendation: Test the classes directly (HIGH priority). Add one or two smoke tests that compose Hydra config to verify wiring (LOW priority). Do not call `main()` directly in tests.

2. **Should tests fix the bucket rounding mismatch or just document it?**
   - What we know: Phase 1 is about characterization, not fixes.
   - Recommendation: Write separate tests for analyzer (2-decimal) and visualizer (1-decimal) bucketing, with comments noting the inconsistency. Mark as a known issue for future phases.

3. **How much model testing is needed given it's not used in mock mode?**
   - What we know: `model.py` has the GPT implementation used for local training (currently unimplemented). Phase 1 covers EXIST-01 through EXIST-07, which focus on the pipeline.
   - Recommendation: Test basic forward pass, param counting, and `create_model_for_scaling` accuracy. These are fast tests and create the safety net needed for Phase 3 (GPT Plugin Refactor).

## Sources

### Primary (HIGH confidence)
- Direct codebase reading: all `.py` files in `src/flops_fit/`, all `.yaml` configs, all existing tests
- `.planning/codebase/CONCERNS.md` - documented bugs and tech debt
- `.planning/codebase/TESTING.md` - existing test patterns and gaps
- `.planning/codebase/ARCHITECTURE.md` - pipeline structure and data flow

### Secondary (MEDIUM confidence)
- Hydra testing patterns: Based on training data knowledge of Hydra 1.3.x API (`initialize_config_dir`, `compose`). The project uses `hydra-core>=1.3.2` which supports these APIs.
- pytest patterns: Based on training data knowledge of pytest 8.x. The project uses `pytest>=8.0.0`.

### Tertiary (LOW confidence)
- None. All research is based on direct codebase analysis and well-established testing patterns.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Already defined in pyproject.toml, no new libraries needed
- Architecture: HIGH - Based on direct codebase reading and existing test patterns
- Pitfalls: HIGH - Hydra testing gotchas and matplotlib backend issues are well-known
- Code examples: HIGH - Based on actual codebase classes and their APIs

**Research date:** 2026-02-15
**Valid until:** No expiration (codebase-specific research, not library-version-dependent)
