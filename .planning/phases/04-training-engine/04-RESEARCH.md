# Phase 4: Training Engine - Research

**Researched:** 2026-02-17
**Domain:** PyTorch training loops, device management, experiment execution, result persistence, training interruption/resumption
**Confidence:** HIGH

## Summary

Phase 4 implements the core training loop that executes experiments from a `SweepPlan` and captures training metrics. The heavy lifting is already done: `trainer.py` exists with `TrainingRunner`, mock mode works, resume logic is implemented, and result serialization is in place. What remains is implementing the "local" training mode that runs actual PyTorch training using user-provided datasets and loss functions.

The critical design decisions are: (1) automatic GPU device placement that falls back to CPU, (2) integrating Phase 2's dataset/loss interfaces into the training loop, (3) consuming `SweepPlan` experiments to configure each training run, (4) capturing actual FLOPs, wall time, and final loss metrics, and (5) preserving resume capability so interrupted sweeps don't recompute completed experiments.

The existing codebase provides the skeleton. Phase 4's primary work is: (a) implement a `_local_train()` method on `TrainingRunner` that trains a real model using PyTorch, (b) wire `wrap_dataset()` and loss validation into the training loop, (c) implement device placement logic (GPU if available, CPU fallback), (d) integrate `SweepPlan` consumption into `find_optimal()` to execute training when budgets are provided, and (e) ensure results are persisted atomically so resumption is safe.

**Primary recommendation:** Extend `TrainingRunner` with a `_local_train()` method that: uses `torch.cuda.is_available()` for device selection, wraps the dataset with `wrap_dataset()`, implements a standard PyTorch training loop (forward pass, loss computation, backward, optimizer step), and captures FLOPs using the formula `C = 6 * N * D`. Wire training execution into `find_optimal()` so passing `compute_budgets` + `dataset` + `loss_fn` triggers training (not just planning). Reuse existing resume logic and result serialization.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | >=2.0.0 | `nn.Module`, DataLoader, device management, backward pass | Project dependency; de facto standard for PyTorch training |
| torch.cuda | built-in | `is_available()`, device placement, GPU detection | Standard PyTorch CUDA utilities for device management |
| tqdm | >=4.66.0 | Progress bars during training | Already a project dependency; standard for ML training loops |
| numpy | >=1.26.0 | FLOPs calculation, numerical operations | Already a dependency; used for sweep planning |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.optim | built-in | Optimizer implementations (SGD, Adam, etc.) | Required for gradient descent; standard PyTorch |
| logging | stdlib | Structured logging of training progress | Standard Python logging; already used in codebase |
| json | stdlib | Results persistence to JSON files | Standard; already used in trainer.py for results.json |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| PyTorch for training | TensorFlow, JAX | PyTorch is the project's core dependency. Switching would require rewriting model/dataset interfaces. Not justified for this phase. |
| Simple training loop (no checkpointing) | Checkpointing per-epoch | Checkpointing adds I/O complexity. For scaling law experiments (relatively short, single-epoch runs), simple loop is sufficient. Checkpoint support can be v2. |
| Automatic device placement (is_available check) | Manual device config via CLI/yaml | Automatic fallback to CPU is more ergonomic -- users don't need to know about GPU availability. Matches library-first design. |
| Optimizer choice left to user | Fixed Adam optimizer | Users should control optimizer. Add optional `optimizer_cls` parameter to training function. Default to SGD (simpler, no momentum complexity). |
| FLOPs calculation via formula (6*N*D) | Actual hardware profiling | Formula is the standard Chinchilla approximation and works for all dense transformers. Profiling adds dependency (e.g., torch.profiler) and platform sensitivity. Stick with formula. |

**Installation:**
```bash
# No new dependencies — all are already in pyproject.toml
pip install torch>=2.0.0 tqdm>=4.66.0
```

## Architecture Patterns

### Recommended Project Structure

```
src/flops_fit/
    __init__.py          # export find_optimal, train_sweep
    api.py               # MODIFY: integrate training execution into find_optimal()
    trainer.py           # MODIFY: implement _local_train() method, add device placement
    sweep.py             # UNCHANGED (from Phase 3)
    model_factory.py     # UNCHANGED (from Phase 1)
    data.py              # UNCHANGED (from Phase 2)
    loss.py              # UNCHANGED (from Phase 2)
    planner.py           # UNCHANGED (existing CLI planner)
    model.py             # UNCHANGED
    analyzer.py          # UNCHANGED
    visualizer.py        # UNCHANGED
```

**Rationale:** The `TrainingRunner` class is the core abstraction. Phase 4 extends it with a `_local_train()` implementation. The `find_optimal()` API is modified to trigger training execution when `compute_budgets`, `dataset`, and `loss_fn` are all provided. No new modules are needed.

### Pattern 1: Device Placement Strategy

**What:** Automatic detection of GPU availability with CPU fallback.

**When to use:** In any training function that needs to move models and data to accelerators.

**Example:**
```python
# trainer.py
import torch

def _get_device():
    """Get the device to use for training.

    Returns 'cuda:0' if available, else 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')

# Inside _local_train():
device = _get_device()
model.to(device)
for batch in dataloader:
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    ...
```

**Why this pattern:**
- Transparent to users: no CLI flags needed
- Safe fallback: if GPU unavailable or out of memory, CPU training still works (slower but correct)
- Matches library design philosophy: library handles details, user calls simple function

**Pitfall to avoid:** Don't assume GPU is available. Always check and fall back gracefully.

### Pattern 2: Integration with SweepPlan and Dataset/Loss

**What:** Training loop accepts an `Experiment` from the sweep plan, user's dataset, and user's loss function. It creates a fresh model for each experiment, trains it, and captures results.

**When to use:** Inside `_local_train()` and the training execution path.

**Example:**
```python
# trainer.py
def _local_train(
    self,
    experiment: Experiment,
    model_cls,
    size_param: str,
    model_kwargs: dict,
    dataset_or_loader: DataLoader,
    loss_fn: Callable,
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> tuple[float, float, float]:
    """Train a model on an experiment and return (loss, actual_flops, wall_time).

    Args:
        experiment: The Experiment from SweepPlan with size_param_value, num_tokens, etc.
        model_cls: The model class to train.
        size_param: Name of the size parameter (e.g., 'd_model').
        model_kwargs: Other constructor kwargs.
        dataset_or_loader: User-provided dataset or DataLoader.
        loss_fn: User-provided loss function.
        epochs: Number of training epochs (default 1).
        batch_size: Batch size (used if wrapping dataset).
        learning_rate: Learning rate for optimizer.

    Returns:
        (final_loss, actual_flops, wall_time_seconds)
    """
    import time
    from flops_fit.model_factory import create_model
    from flops_fit.data import wrap_dataset

    # 1. Create model at the experiment's size
    model = create_model(
        model_cls,
        size_param,
        experiment.size_param_value,
        model_kwargs,
    )

    # 2. Wrap dataset
    dataloader = wrap_dataset(dataset_or_loader, batch_size=batch_size)

    # 3. Device placement
    device = _get_device()
    model.to(device)

    # 4. Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 5. Training loop
    start_time = time.time()
    model.train()

    total_batches = 0
    total_loss = 0.0

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            # Handle both (inputs, targets) and single-tensor batches
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, targets = batch
            else:
                # Single tensor: use as both input and target (for regression/MLM)
                inputs = batch
                targets = batch

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    wall_time = time.time() - start_time

    # Final loss (average across all batches)
    final_loss = total_loss / total_batches if total_batches > 0 else float('nan')

    # Actual FLOPs: C = 6 * N * D
    # N = num_params from experiment
    # D = num_tokens from experiment (conceptual; actual depends on batch size)
    # For simplicity, use the experiment's planned values
    actual_flops = 6 * experiment.num_params * experiment.num_tokens

    # Cleanup
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return final_loss, actual_flops, wall_time
```

**Why this pattern:**
- Experiment configures the model size and token budget
- User's dataset and loss are decoupled from library internals
- Fresh model per experiment ensures clean state
- Memory cleanup prevents accumulation across many experiments

### Pattern 3: Consuming SweepPlan in Training Execution

**What:** The training engine receives a `SweepPlan` and iterates through its experiments, training each one and collecting results.

**When to use:** In the orchestration layer that executes the full sweep (inside `find_optimal()` when training is enabled).

**Example:**
```python
# Inside find_optimal() or a new train_sweep() function
def _execute_sweep(
    sweep_plan: SweepPlan,
    dataset_or_loader: DataLoader,
    loss_fn: Callable,
    model_cls,
    output_dir: Path | str = "outputs",
    resume: bool = True,
) -> list[dict]:
    """Execute all experiments in a sweep plan.

    Args:
        sweep_plan: The SweepPlan from plan_sweep().
        dataset_or_loader: User's dataset or DataLoader.
        loss_fn: User's loss function.
        model_cls: The model class to train.
        output_dir: Directory to save results.
        resume: If True, skip already-completed experiments.

    Returns:
        List of training results (dicts).
    """
    runner = TrainingRunner(
        mode="local",
        output_dir=output_dir,
    )

    # Store sweep plan metadata for reference
    runner._sweep_plan = sweep_plan
    runner._dataset = dataset_or_loader
    runner._loss_fn = loss_fn
    runner._model_cls = model_cls

    # Run each experiment
    results = []
    completed = set()

    results_path = Path(output_dir) / "results.json"
    if resume and results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
            results = existing
            completed = {r["experiment_id"] for r in existing if r["status"] == "completed"}

    for exp in tqdm(sweep_plan.experiments, desc="Training sweep"):
        if exp.experiment_id in completed:
            continue

        result = runner.run_experiment_from_sweep(
            experiment=exp,
            dataset_or_loader=dataset_or_loader,
            loss_fn=loss_fn,
            model_cls=model_cls,
        )
        results.append(result.to_dict())

        # Persist incrementally
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    return results
```

### Pattern 4: Integrating Training into find_optimal()

**What:** Modify `find_optimal()` so that when `compute_budgets`, `dataset`, and `loss_fn` are all provided, it not only plans the sweep but also executes it immediately.

**When to use:** In the public API layer, balancing convenience (users want single-call execution) with flexibility (users might want to inspect the plan first).

**Example:**
```python
# api.py (modified)
def find_optimal(
    model_cls,
    model_size_param,
    model_kwargs=None,
    dataset=None,
    loss_fn=None,
    compute_budgets=None,
    train: bool = True,
    **kwargs,
):
    """Find compute-optimal model size using scaling law experiments.

    Args:
        model_cls: Model class.
        model_size_param: Name of the size parameter.
        model_kwargs: Other constructor kwargs.
        dataset: Training dataset (Phase 2).
        loss_fn: Loss function (Phase 2).
        compute_budgets: List of compute budgets in FLOPs (Phase 3).
        train: If True and dataset+loss_fn provided, execute training immediately.
               If False, just return the sweep plan (for inspection).
        **kwargs: Additional options (e.g., output_dir, batch_size).

    Returns:
        SweepPlan if train=False or dataset is None.
        list[TrainingResult] if train=True and dataset provided.
        (Raises NotImplementedError if compute_budgets not provided.)
    """
    if model_kwargs is None:
        model_kwargs = {}

    # Validation (Phase 1, 2)
    validate_model_contract(model_cls, model_size_param, model_kwargs)
    if dataset is not None:
        validate_dataset(dataset)
    if loss_fn is not None:
        validate_loss_fn(loss_fn)

    # Planning (Phase 3)
    if compute_budgets is not None:
        plan = plan_sweep(
            model_cls=model_cls,
            size_param=model_size_param,
            model_kwargs=model_kwargs,
            compute_budgets=compute_budgets,
        )

        # Execution (Phase 4)
        if train and dataset is not None and loss_fn is not None:
            return _execute_sweep(
                sweep_plan=plan,
                dataset_or_loader=dataset,
                loss_fn=loss_fn,
                model_cls=model_cls,
                output_dir=kwargs.get("output_dir", "outputs"),
            )

        # Just return plan for inspection
        return plan

    raise NotImplementedError("compute_budgets required for find_optimal()")
```

### Anti-Patterns to Avoid

- **Assuming GPU is always available:** Always check `torch.cuda.is_available()` and fall back to CPU.

- **Not cleaning up models after training:** Each experiment creates a fresh model in memory. Delete it and empty the cache to prevent OOM on sweeps with many experiments.

- **Hardcoding optimizer or learning rate:** Let users pass optimizer_cls and learning_rate as optional parameters. Provide sensible defaults (SGD, lr=0.01).

- **Training multiple epochs by default:** IsoFLOPs experiments typically use single-epoch training. If sweeping across token counts, increasing epochs complicates the token-per-param ratio. Default to epochs=1; let users override.

- **Not persisting results atomically during sweep:** Write results.json after each experiment completes, not just at the end. This enables safe resumption if interrupted.

- **Using actual FLOPs formula incorrectly:** The formula is `C = 6 * N * D`, where N is parameter count and D is token count. Do NOT double-count forward/backward passes; the 6x factor already includes both.

- **Assuming batch_size divides the dataset evenly:** Use `drop_last=True` in DataLoader (already in `wrap_dataset()`) to ensure consistent batch sizes across epochs.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GPU vs CPU detection | Manual device string parsing | `torch.cuda.is_available()` and `torch.device()` | Standard, well-tested, handles edge cases (multiple GPUs, cuda not installed, etc.) |
| DataLoader wrapping | Custom iteration logic | `torch.utils.data.DataLoader` with `wrap_dataset()` | Handles batching, shuffling, multiprocessing, device placement |
| Gradient descent | Manual weight updates | `torch.optim.SGD`, `torch.optim.Adam` | Optimized, support momentum/adaptive learning rates, avoid numerical errors |
| Memory profiling / FLOPs measurement | Custom profiler | Formula-based calculation (6*N*D) | Simple, fast, works across all architectures without instrumentation |
| Results serialization | Custom pickle/protobuf | JSON via `TrainingResult.to_dict()` | Human-readable, already used in existing codebase, sufficient for this phase |
| Experiment orchestration | Manual loop with error handling | Existing `TrainingRunner` class with `run_experiment()` | Encapsulates resume logic, result persistence, error handling |

**Key insight:** PyTorch provides all the primitives needed. The training engine's job is to orchestrate them (device placement, dataset integration, result capture) without adding complexity.

## Common Pitfalls

### Pitfall 1: OOM Errors During Sweep Due to Model Accumulation

**What goes wrong:** Training 50 experiments on a GPU with limited memory. Partway through, GPU runs out of memory even though each individual model fit before.

**Why it happens:** Models aren't garbage collected immediately after training. PyTorch retains computation graphs. Memory fragmentation accumulates.

**How to avoid:**
1. Delete the model object explicitly: `del model` after training
2. Clear CUDA cache: `torch.cuda.empty_cache()` after each experiment
3. Use `torch.no_grad()` for validation-only passes to avoid building graphs
4. Monitor memory with `torch.cuda.memory_allocated()` during development

**Warning signs:** GPU memory usage creeps up with each experiment. `RuntimeError: out of memory` happens midway through a sweep that should fit.

**Code example:**
```python
# Inside _local_train():
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### Pitfall 2: Incorrect FLOPs Calculation Due to Misunderstanding the Formula

**What goes wrong:** User calculates FLOPs as `C = 2 * N * D` (thinking forward + backward = 2x), but actual experiments use the Chinchilla 6x factor. Results don't match expected scaling laws.

**Why it happens:** The 6x factor in `C = 6 * N * D` is non-obvious. It includes forward pass (2 ops per param per token) + backward (4 ops), not just 1+1.

**How to avoid:** Document the formula clearly. Use a named constant: `FLOPS_PER_PARAM_PER_TOKEN = 6`. Add a comment explaining the breakdown. Allow users to override via `flops_per_param_per_token` parameter (v2).

**Warning signs:** Calculated FLOPs don't match measured FLOPs from profilers by ~3x.

### Pitfall 3: Resume Logic Doesn't Handle Partial Results from Crashed Runs

**What goes wrong:** User resumes a sweep after a crash. Some experiments have partial results (loss=NaN, wall_time=0) from the crash. Resume logic sees these as "completed" and skips them.

**Why it happens:** The resume check looks for `status == "completed"`, but crashed runs might have `status = "failed"` with NaN loss.

**How to avoid:**
1. Only count `status == "completed"` as truly done (not "failed" or "partial")
2. After resuming, **re-run any experiment with NaN or 0 wall_time** (these indicate incomplete runs)
3. Test resume on partially-written results.json

**Warning signs:** Resuming a sweep produces NaN losses in final results.

### Pitfall 4: Dataset Shuffling Breaks Token Count Consistency

**What goes wrong:** User provides a dataset with shuffle=True. During training, the randomness means actual token count processed differs from planned tokens per experiment.

**Why it happens:** `wrap_dataset()` respects the user's shuffle flag. If shuffled, the data order is non-deterministic, affecting the effective epoch length (due to `drop_last=True`).

**How to avoid:**
1. `wrap_dataset()` already forces `shuffle=False` for IterableDataset (which are always shuffled by definition)
2. For regular Datasets, respect the user's choice but warn if shuffle=True and tokens matter
3. Document that for scaling law experiments, shuffle order doesn't affect FLOPs (which is determined by num_tokens config, not actual data seen)

**Warning signs:** Small variations in loss between runs of the same experiment (this is actually expected noise, not a problem).

### Pitfall 5: Loss Function Doesn't Match Model Output Shape

**What goes wrong:** User passes model that outputs shape [batch_size, d_model] but loss_fn expects [batch_size] or [batch_size, num_classes].

**Why it happens:** `validate_loss_fn()` only checks arity (accepts 2+ args), not tensor shape compatibility.

**How to avoid:**
1. This is a user responsibility -- validate loss and model work together before calling `find_optimal()`
2. In Phase 4's training loop, wrap the forward pass in try/except to catch shape mismatches and provide clear error messages
3. Run a test batch through model + loss during sweep planning (Phase 3 could add this)

**Warning signs:** `RuntimeError: shape mismatch` during training starts. Error occurs on the first batch, not later.

### Pitfall 6: Using Wrong Optimizer Leads to Diverging Loss

**What goes wrong:** Default learning rate or optimizer is too aggressive. Loss diverges to infinity or NaN.

**Why it happens:** Different model architectures and datasets benefit from different learning rates. A fixed default (e.g., lr=0.1) is too high for some experiments.

**How to avoid:**
1. Use a conservative default (e.g., lr=0.001 for Adam, lr=0.01 for SGD)
2. Let users pass custom `optimizer_cls` and `learning_rate` if needed
3. For Phase 4 (first implementation), use SGD with lr=0.01 as default (simple, stable)
4. Document that users should tune lr for their architecture

**Warning signs:** Training loss goes to NaN after first epoch. Loss increases instead of decreasing.

### Pitfall 7: Device Placement Issues with Mixed Precision or Distributed Training

**What goes wrong:** User's model uses mixed precision or distributed training. Simple device placement breaks.

**Why it happens:** This phase assumes single-GPU (or single-CPU) training. Mixed precision and distributed training require additional setup.

**How to avoid:** Phase 4 is single-device training only. Document this as a v2 concern. If user needs distributed training, they should run their own training loop (library doesn't support it yet).

**Warning signs:** Phase 4 is out of scope if user mentions multi-GPU or mixed precision.

## Code Examples

### Example 1: Simple Training Execution

```python
import torch
from flops_fit import find_optimal

# User's model
class SimpleModel(torch.nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
        self.width = width

    def forward(self, x):
        return self.linear(x)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

# User's dataset
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = x.sum().view(1)  # Simple target
        return x, y

# User's loss
loss_fn = torch.nn.MSELoss()

# Find optimal
results = find_optimal(
    model_cls=SimpleModel,
    model_size_param="width",
    dataset=SimpleDataset(),
    loss_fn=loss_fn,
    compute_budgets=[1e15, 1e16],
)

# Results is a list of training results
for result in results:
    print(f"Model size: {result['model_size']}, Loss: {result['final_loss']:.4f}")
```

### Example 2: Inspecting Plan Before Training

```python
# Just get the plan, don't train yet
plan = find_optimal(
    model_cls=SimpleModel,
    model_size_param="width",
    compute_budgets=[1e15, 1e16],
    train=False,  # Don't execute, just plan
)

print(plan)
# SweepPlan(14 experiments, 2 budgets, total_flops=2.00e+16)

# Review the experiments
for exp in plan.experiments:
    print(f"  N={exp.num_params:,}  D={exp.num_tokens:,}  C={exp.compute_budget:.1e}")

# If happy with the plan, train
results = find_optimal(
    model_cls=SimpleModel,
    model_size_param="width",
    dataset=SimpleDataset(),
    loss_fn=loss_fn,
    compute_budgets=[1e15, 1e16],
    train=True,  # Now execute
)
```

### Example 3: Resume Interrupted Sweep

```python
# First run (interrupted after 5 experiments)
results_1 = find_optimal(
    model_cls=SimpleModel,
    model_size_param="width",
    dataset=SimpleDataset(),
    loss_fn=loss_fn,
    compute_budgets=[1e15, 1e16],
    output_dir="/tmp/my_sweep",
)
# Interrupted here...

# Resume later
results_2 = find_optimal(
    model_cls=SimpleModel,
    model_size_param="width",
    dataset=SimpleDataset(),
    loss_fn=loss_fn,
    compute_budgets=[1e15, 1e16],
    output_dir="/tmp/my_sweep",
    resume=True,  # Skip completed experiments
)

# Results_2 includes all experiments (first 5 from results_1, rest newly computed)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Mock training only (TrainingRunner mode="mock") | Local training with real PyTorch loops (mode="local") | Phase 4 (this phase) | Can now train actual models; predictions validated against real losses |
| Manual device management by user | Automatic GPU detection with CPU fallback | Phase 4 (this phase) | Transparent to users; no CLI flags needed |
| No integration with find_optimal() | Training triggered from find_optimal() when dataset+loss provided | Phase 4 (this phase) | Single-call entry point; more ergonomic |
| Hardcoded single-epoch training | Configurable epochs (default 1) | Phase 4 (this phase) | Supports experiments with variable iteration counts |
| No atomic result persistence during sweep | Incremental writes after each experiment | Phase 4 (this phase) | Safe resumption without result loss; auditable progress |

## Open Questions

1. **What optimizer and learning rate should be the default?**
   - What we know: Different architectures need different settings. SGD is simpler than Adam but may converge slower.
   - What's unclear: Should Phase 4 use a single fixed default, or provide a callback for users to customize?
   - Recommendation: Default to SGD with lr=0.01. This is stable across most architectures. Add optional `optimizer_cls` and `learning_rate` parameters to `_local_train()` for users who want to customize. Document that users should tune lr for their architecture.

2. **Should training support multiple epochs or always single-epoch?**
   - What we know: IsoFLOPs experiments typically use single-epoch training to keep token count constant. Adding epochs complicates the token-per-param ratio.
   - What's unclear: Should Phase 4 support multi-epoch training for users who want it?
   - Recommendation: Default to 1 epoch. Add optional `epochs` parameter. Document that multi-epoch training changes the effective token budget and may not follow scaling laws.

3. **How should the training loop handle datasets that don't fit in memory?**
   - What we know: PyTorch DataLoaders handle this via batching and num_workers.
   - What's unclear: Should Phase 4 implement streaming/checkpoint logic, or assume user's dataset fits?
   - Recommendation: Phase 4 assumes dataset fits (or DataLoader handles streaming). Streaming datasets and checkpointing are v2 concerns. Document the assumption.

4. **What about batch size selection?**
   - What we know: Batch size affects training dynamics and can affect loss. IsoFLOPs doesn't specify optimal batch size.
   - What's unclear: Should batch size be constant across experiments, or should it scale with model size?
   - Recommendation: Accept optional `batch_size` parameter (default 32). Let users override. Batch size doesn't affect FLOPs (which is C = 6*N*D based on tokens, not batches), so different experiments can use different batch sizes if desired.

5. **Should Phase 4 validate that loss_fn outputs match model outputs before training?**
   - What we know: Phase 2 validates loss_fn arity but not shape compatibility.
   - What's unclear: Should Phase 3 run a test forward pass to catch shape mismatches early?
   - Recommendation: Phase 3 (sweep planning) runs a test batch through model + loss using the smallest experiment size. If it fails, the error is raised during planning, not during training of experiment 1. This adds a few seconds to `plan_sweep()` but saves debugging time.

## Sources

### Primary (HIGH confidence)
- Existing codebase: `src/flops_fit/trainer.py` -- TrainingRunner, TrainingResult, mock training, resume logic already implemented
- Existing codebase: `src/flops_fit/sweep.py` -- Experiment, SweepPlan dataclasses (Phase 3 output)
- Existing codebase: `src/flops_fit/data.py` -- wrap_dataset() function with drop_last=True, shuffle handling
- Existing codebase: `src/flops_fit/loss.py` -- validate_loss_fn() for loss contract checking
- Existing codebase: `src/flops_fit/model_factory.py` -- create_model() for instantiating models at different sizes
- Existing codebase: `src/flops_fit/api.py` -- find_optimal() stub to be extended in Phase 4
- PyTorch official docs (https://pytorch.org/docs/stable/torch.html) -- torch.cuda, torch.optim, DataLoader
- Project dependencies: `pyproject.toml` -- torch>=2.0.0 is already a dependency

### Secondary (MEDIUM confidence)
- `.planning/phases/02-dataset-and-loss-interfaces/02-RESEARCH.md` -- dataset/loss validation patterns
- `.planning/phases/03-sweep-planning/03-RESEARCH.md` -- SweepPlan consumption and experiment structure
- Existing tests: `tests/test_trainer.py` -- test patterns for TrainingRunner, run_sweep, resume logic
- PyTorch best practices (implicit from ecosystem): always check torch.cuda.is_available(), use torch.device(), clean up models

### Tertiary (LOW confidence)
- None. This phase uses well-established PyTorch APIs and patterns.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- PyTorch is the project's core dependency; torch.cuda and torch.optim are stable APIs
- Architecture: HIGH -- Extends existing TrainingRunner class; wires into existing find_optimal(). Simple patterns.
- Pitfalls: HIGH -- Based on direct analysis of existing trainer.py, data.py, loss.py, and common PyTorch training loop issues

**Research date:** 2026-02-17
**Valid until:** Until PyTorch releases a major version with breaking changes to torch.cuda or torch.optim (unlikely in next 12 months)
