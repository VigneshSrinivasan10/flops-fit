# Phase 6: Results Object and API Integration - Research

**Researched:** 2026-02-17
**Domain:** End-to-end API integration, Result object design, pipeline orchestration
**Confidence:** HIGH

## Summary

Phase 6 completes the public API surface by wrapping the analysis and visualization pipeline into a `Result` object returned from `find_optimal()`. The goal is a single orchestration point that coordinates plan → train → analyze → visualize operations and exposes three user-facing methods: `chinchilla_table()`, `plot()`, and `predict(compute_budget)`.

The Result object is a lightweight wrapper around `ScalingAnalysis` (Phase 5) and `ScalingVisualizer` (Phase 5) that aggregates the pipeline outputs and presents them through a cohesive interface. The orchestration logic belongs in `find_optimal()` (Phase 1 stub, now live in api.py), which currently stops after training returns results.

**Primary recommendation:** Create a `Result` dataclass that wraps `ScalingAnalysis`, add three methods delegating to analysis/visualizer, then modify `find_optimal()` to orchestrate analyze→visualize→return Result when training completes.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `dataclasses` | stdlib | Result object definition | Built-in, zero-dependency type safety |
| `ScalingAnalysis` | Phase 5 | Power law fitting and prediction | Already implemented and tested |
| `ScalingVisualizer` | Phase 5 | Visualization generation | Already implemented; produces matplotlib figures |
| `matplotlib` | 3.x | Return figure objects for user inspection | Standard for scientific Python; already a project dependency |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pathlib.Path` | stdlib | Output directory management | Already used throughout codebase |
| `typing` | stdlib | Type hints for Result methods | Part of existing conventions |
| `json` | stdlib | Serialization of analysis results | Already used in analyzer |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Lightweight `Result` dataclass | Create a full Result class hierarchy | Would add complexity; dataclass is sufficient |
| Immediate `find_optimal()` return | Lazy Result creation on first method call | Upfront approach simpler; analysis must run anyway |
| Store analysis JSON to disk in Result | Reconstruct on demand from stored JSON | Storing to disk is Phase 5's job; Result focuses on runtime object |

## Architecture Patterns

### Result Object Contract

The Result object is a simple aggregator that bundles:
- `ScalingAnalysis` instance (power law fits, optimal points)
- `ScalingVisualizer` instance (plotting capability)
- Metadata (output directory, model info, compute budgets used)

**Responsibility boundary:**
- Result: exposes user-facing API (chinchilla_table, plot, predict)
- ScalingAnalysis: owns fitting logic and stored JSON
- ScalingVisualizer: owns plot generation and saved PNG files

### Recommended Project Structure
```
src/flops_fit/
├── api.py                 # find_optimal() orchestration (updated)
├── result.py              # NEW: Result dataclass (or add to analyzer.py)
├── analyzer.py            # ScalingAnalysis (unchanged from Phase 5)
└── visualizer.py          # ScalingVisualizer (unchanged from Phase 5)
```

### Pattern 1: Pipeline Orchestration in find_optimal()

**What:** After TrainingRunner.run_sweep_from_plan() returns results, `find_optimal()` chains:
1. Load training results into DataFrame
2. Create ScalingLawAnalyzer and call analyze()
3. Create ScalingVisualizer with results + analysis paths
4. Wrap in Result object and return

**When to use:** All training paths that complete training (train=True, dataset+loss_fn provided)

**Example:**
```python
# In api.py find_optimal()
if train and dataset is not None and loss_fn is not None:
    from flops_fit.trainer import TrainingRunner
    runner = TrainingRunner(mode="local", output_dir=output_dir)
    runner.run_sweep_from_plan(...)  # Returns list[dict], writes results.json

    # Phase 6: Create analysis and result
    from flops_fit.analyzer import ScalingLawAnalyzer
    from flops_fit.visualizer import ScalingVisualizer
    from flops_fit.result import Result

    analyzer = ScalingLawAnalyzer(
        results_path=Path(output_dir) / "results.json",
        output_dir=Path(output_dir) / "analysis",
    )
    analysis = analyzer.analyze()  # Writes scaling_laws.json

    visualizer = ScalingVisualizer(
        results_path=Path(output_dir) / "results.json",
        analysis_path=Path(output_dir) / "analysis" / "scaling_laws.json",
        output_dir=Path(output_dir) / "plots",
    )

    result = Result(
        analysis=analysis,
        visualizer=visualizer,
        output_dir=output_dir,
        compute_budgets=compute_budgets,
    )
    return result
```

### Pattern 2: Result Method Delegation

**What:** Result methods delegate to underlying analysis/visualizer:
- `result.chinchilla_table(budgets)` → calls `analysis.chinchilla_table(budgets)`
- `result.predict(budget)` → calls `analysis.predict_optimal_size(budget)`
- `result.plot()` → calls `visualizer.plot_all(save=True)` and returns figures

**When to use:** User-facing API — keep implementations in respective modules, Result just exposes them

**Example:**
```python
@dataclass
class Result:
    analysis: ScalingAnalysis
    visualizer: ScalingVisualizer
    output_dir: str | Path
    compute_budgets: list[float] = field(default_factory=list)

    def chinchilla_table(self, compute_budgets: list[float] | None = None) -> str:
        """Return Chinchilla-style table of optimal configs."""
        return self.analysis.chinchilla_table(compute_budgets)

    def predict(self, compute_budget: float) -> dict:
        """Predict optimal N, D, loss for a specific compute budget."""
        return self.analysis.predict_optimal_size(compute_budget)

    def plot(self) -> list[plt.Figure]:
        """Generate all visualizations (IsoFLOPs, scaling laws, D/N ratio)."""
        return self.visualizer.plot_all(save=True)
```

### Anti-Patterns to Avoid

- **Duplicating fitting logic in Result:** Don't reimplement predict or table generation; delegate to ScalingAnalysis
- **Result as storage layer:** Don't add serialization methods; that's ScalingLawAnalyzer's job
- **Forcing visualization on every method call:** plot() returns figures but doesn't auto-display; let user control show/save
- **Synchronous file I/O on import:** Ensure Result constructor doesn't block on disk I/O; paths are passed but not validated until methods called

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Power law fitting | Custom NLS solver | scipy.optimize.least_squares + l_inf handling (Phase 5) | Unstable numerics, outlier handling complex |
| Visualization | Custom matplotlib code | ScalingVisualizer.plot_*() methods | Consistent styling, tested color palettes, publication-ready |
| Table formatting | Manual string building | ScalingAnalysis.chinchilla_table() | Markdown format, alignment, formatting already correct |
| Pipeline orchestration | Ad-hoc script logic | find_optimal() wrapper | Single point of control, resumable, testable |

**Key insight:** The Result object is a *facade* over existing components, not a new implementation. All heavy lifting (fitting, plotting) happens in Phase 5; Phase 6 is glue.

## Common Pitfalls

### Pitfall 1: Analyzing Before Training Completes

**What goes wrong:** Result object created before results.json is written, analyzer fails with FileNotFoundError

**Why it happens:** Orchestration logic runs analysis before TrainingRunner finishes writing results

**How to avoid:** Always analyze AFTER runner.run_sweep_from_plan() completes and synchronously in same transaction

**Warning signs:** FileNotFoundError on results.json, analyzer initialization failures

### Pitfall 2: Path Assumptions in find_optimal()

**What goes wrong:** Hard-coded paths like "outputs/analysis" break when user provides custom output_dir

**Why it happens:** Analyzer and visualizer use default paths; need to align with output_dir from find_optimal()

**How to avoid:** Construct all paths relative to output_dir parameter; pass explicitly to ScalingLawAnalyzer and ScalingVisualizer

**Warning signs:** Tests pass locally but fail with custom output_dir; missing files in user-specified directories

### Pitfall 3: Returning Raw Results List Instead of Result Object

**What goes wrong:** find_optimal() returns list[dict] from trainer instead of Result, user can't call .plot() or .predict()

**Why it happens:** Copy-paste from Phase 4; forgot to wrap results in Result object

**How to avoid:** Always return Result at end of orchestration chain; list[dict] only for internal use

**Warning signs:** Type hints show list[dict] as return type; test expects SweepPlan or Result but gets list

### Pitfall 4: Visualizer/Analyzer Using Default Paths

**What goes wrong:** ScalingVisualizer looks for "outputs/results.json" even though training wrote to "/tmp/custom/results.json"

**Why it happens:** Analyzer/visualizer constructors have hardcoded defaults; find_optimal() doesn't override them

**How to avoid:** Explicitly pass results_path, analysis_path, output_dir to Analyzer and Visualizer constructors in find_optimal()

**Warning signs:** "Results not found: outputs/results.json" even though user trained to custom directory

## Code Examples

### Example 1: Complete find_optimal() Flow with Result

```python
# Source: src/flops_fit/api.py (modified)
from pathlib import Path
from flops_fit.analyzer import ScalingLawAnalyzer
from flops_fit.visualizer import ScalingVisualizer
from flops_fit.result import Result

def find_optimal(
    model_cls,
    model_size_param,
    model_kwargs=None,
    dataset=None,
    loss_fn=None,
    compute_budgets=None,
    train: bool = True,
    output_dir: str = "outputs",
    resume: bool = True,
    **kwargs,
) -> Result | SweepPlan:
    """Find compute-optimal model size and return Result with prediction API."""
    if model_kwargs is None:
        model_kwargs = {}

    validate_model_contract(model_cls, model_size_param, model_kwargs)
    if dataset is not None:
        validate_dataset(dataset)
    if loss_fn is not None:
        validate_loss_fn(loss_fn)

    if compute_budgets is not None:
        plan = plan_sweep(
            model_cls=model_cls,
            size_param=model_size_param,
            model_kwargs=model_kwargs,
            compute_budgets=compute_budgets,
        )

        # Phase 4: Execute training
        if train and dataset is not None and loss_fn is not None:
            from flops_fit.trainer import TrainingRunner
            runner = TrainingRunner(mode="local", output_dir=output_dir)
            runner.run_sweep_from_plan(
                plan=plan,
                model_cls=model_cls,
                size_param=model_size_param,
                model_kwargs=model_kwargs,
                dataset_or_loader=dataset,
                loss_fn=loss_fn,
                resume=resume,
            )

            # Phase 6: Analyze and wrap in Result
            output_path = Path(output_dir)
            analyzer = ScalingLawAnalyzer(
                results_path=output_path / "results.json",
                output_dir=output_path / "analysis",
            )
            analysis = analyzer.analyze()

            visualizer = ScalingVisualizer(
                results_path=output_path / "results.json",
                analysis_path=output_path / "analysis" / "scaling_laws.json",
                output_dir=output_path / "plots",
            )

            result = Result(
                analysis=analysis,
                visualizer=visualizer,
                output_dir=str(output_path),
                compute_budgets=compute_budgets,
            )
            return result

        # Inspection mode: just return the plan
        return plan

    raise NotImplementedError(
        "find_optimal() model validation passed. "
        "Full pipeline not yet implemented."
    )
```

### Example 2: Result Object Definition

```python
# Source: src/flops_fit/result.py (NEW)
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
from flops_fit.analyzer import ScalingAnalysis
from flops_fit.visualizer import ScalingVisualizer

@dataclass
class Result:
    """
    Results from find_optimal() pipeline.

    Provides user-facing API for inspecting scaling law analysis:
    - chinchilla_table(): Markdown table of optimal configs
    - predict(compute_budget): Optimal N, D, loss for a budget
    - plot(): Generate all visualizations
    """

    analysis: ScalingAnalysis
    visualizer: ScalingVisualizer
    output_dir: str | Path = "outputs"
    compute_budgets: list[float] = field(default_factory=list)

    def chinchilla_table(
        self, compute_budgets: list[float] | None = None
    ) -> str:
        """
        Return Chinchilla-style table of optimal model sizes.

        Args:
            compute_budgets: List of FLOPs budgets to tabulate.
                If None, uses 9 log-spaced budgets from 1e18 to 1e22.

        Returns:
            Markdown-formatted table string
        """
        return self.analysis.chinchilla_table(compute_budgets)

    def predict(self, compute_budget: float) -> dict:
        """
        Predict optimal model configuration for a compute budget.

        Args:
            compute_budget: Target FLOPs

        Returns:
            Dict with optimal_params, optimal_tokens, expected_loss, tokens_per_param
        """
        return self.analysis.predict_optimal_size(compute_budget)

    def plot(self, show: bool = False) -> list[plt.Figure]:
        """
        Generate all visualizations.

        Creates three plots:
        - IsoFLOPs curves (loss vs model size per budget)
        - Scaling laws (N_opt, D_opt, L_opt vs compute)
        - Tokens-per-param ratio

        Args:
            show: If True, display plots interactively (matplotlib.pyplot.show)

        Returns:
            List of matplotlib Figure objects (saved to disk by visualizer)
        """
        figures = self.visualizer.plot_all(save=True)
        if show:
            plt.show()
        return figures
```

### Example 3: Using Result in User Code

```python
# Source: Example usage (not in codebase)
import flops_fit

result = flops_fit.find_optimal(
    model_cls=MyModel,
    model_size_param="hidden_dim",
    dataset=train_dataset,
    loss_fn=torch.nn.CrossEntropyLoss(),
    compute_budgets=[1e18, 1e19, 1e20, 1e21],
    output_dir="./scaling_results",
)

# API-05: Generate Chinchilla-style table
print(result.chinchilla_table())
# | Compute Budget | Optimal N | Optimal D | D/N Ratio | Predicted Loss |
# |---|---|---|---|---|
# | 1.00e+18 | 1,234,567 | 25,000,000 | 20.3 | 2.1540 |
# ...

# API-07: Predict for a new budget
pred = result.predict(5e20)
# {
#   "target_compute": 5e20,
#   "optimal_params": 5678901,
#   "optimal_tokens": 115000000,
#   "expected_loss": 1.8234,
#   "tokens_per_param": 20.2,
# }

# API-06: Generate plots
figs = result.plot(show=True)
# Returns list of 3 matplotlib Figure objects
# Saves to ./scaling_results/plots/
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Separate CLI commands (plan, train, analyze, visualize) | Unified `find_optimal()` returning Result | Phase 6 | Single function call; user doesn't manage intermediate files |
| User manually loads JSON and calls visualizer | Result.plot() calls visualizer transparently | Phase 6 | Simpler API; no JSON path management |
| Isolated train-only return value | Result wrapping analysis + visualizer | Phase 6 | User can predict and visualize without separate calls |

**Deprecated/outdated:**
- Raw trainer output (list[dict]): Now wrapped in Result object for user API
- Separate analyzer/visualizer instantiation in user code: Now handled by find_optimal()

## Open Questions

1. **Should Result store or reconstruct ScalingAnalysis?**
   - What we know: ScalingLawAnalyzer.analyze() returns ScalingAnalysis dataclass; JSON persisted to disk
   - What's unclear: Does Result hold the in-memory object or reconstruct from JSON on demand?
   - Recommendation: Hold in-memory ScalingAnalysis (passed from find_optimal); simpler, no re-parsing JSON

2. **What happens if training is skipped (train=False)?**
   - What we know: find_optimal() returns SweepPlan when train=False
   - What's unclear: Should find_optimal() with train=False ever return Result?
   - Recommendation: No; training must complete to generate results. Return type is SweepPlan | Result, dependent on train flag

3. **Does Result.plot() return figures or save+return paths?**
   - What we know: ScalingVisualizer.plot_all() returns Figure objects and saves PNGs
   - What's unclear: Should Result.plot() return Figure objects, file paths, or both?
   - Recommendation: Return Figure objects (for inspection/modification); paths already saved by visualizer

4. **Should Result validate that analysis and visualizer reference the same data?**
   - What we know: Analyzer and visualizer both read results.json and scaling_laws.json
   - What's unclear: What if paths mismatch (e.g., analyzer used results_v1.json, visualizer uses results_v2.json)?
   - Recommendation: LOW priority; find_optimal() constructs both with same paths, so mismatch unlikely in normal use

## Sources

### Primary (HIGH confidence)
- **ScalingAnalysis (Phase 5):** analyzer.py lines 81-173 — chinchilla_table() and predict_optimal_size() already implemented
- **ScalingVisualizer (Phase 5):** visualizer.py lines 117-363 — plot_isoflops(), plot_scaling_laws(), plot_tokens_per_param(), plot_all()
- **TrainingRunner (Phase 4):** trainer.py lines 85-220+ — run_sweep_from_plan() returns list[dict], writes results.json
- **Current find_optimal() (Phase 1/4):** api.py lines 12-113 — orchestration stub, training execution logic present

### Secondary (MEDIUM confidence)
- **Test expectations:** test_api.py lines 226-319 — test_find_optimal_executes_training_returns_results shows current return is list[dict]; Phase 6 changes return type to Result
- **Pipeline integration:** test_pipeline.py lines 26-75 — test_full_pipeline_mock_mode demonstrates chaining: plan → train → analyze → visualize
- **Analyzer contract:** test_analyzer.py lines 315-378 — TestScalingAnalysis validates predict_optimal_size() and chinchilla_table() methods

### Tertiary (LOW confidence)
- None — all critical code paths verified in Phase 5 implementation and existing tests

## Metadata

**Confidence breakdown:**
- **Standard Stack:** HIGH — ScalingAnalysis and ScalingVisualizer fully implemented Phase 5; dependencies on scipy, matplotlib verified
- **Architecture:** HIGH — Orchestration pattern clear from existing api.py + trainer.py flow; Result object pattern is lightweight wrapper
- **Pitfalls:** MEDIUM — Identified from code inspection; not yet tested in full integration (Phase 6 tasks will validate)

**Research date:** 2026-02-17
**Valid until:** 2026-03-03 (14 days — stable codebase, design unlikely to change)
**Confidence overall:** HIGH — Phase 5 fully implemented; Phase 6 is integration + wrapper layer
