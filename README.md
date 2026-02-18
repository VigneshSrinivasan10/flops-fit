# flops-fit

Fit scaling laws and find compute-optimal model sizes using the **IsoFLOPs method** from the [Chinchilla paper](https://arxiv.org/abs/2203.15556).

Works with **any PyTorch model** — pass your model class and a dataset, get back a Chinchilla-style scaling table and plots.

## Installation

```bash
uv sync
```

## Quickstart

```python
import flops_fit
from flops_fit.examples import GPT, GPTConfig, TinyStoriesDataset
import torch.nn.functional as F

VOCAB_SIZE = 50257

# 1. Wrap your model as a factory that accepts the size parameter
def make_gpt(d_model: int):
    return GPT(GPTConfig(d_model=d_model, num_layers=4, num_heads=4,
                         vocab_size=VOCAB_SIZE, max_seq_len=256))

# 2. Define a loss function: (model_output, labels) -> scalar
def loss_fn(outputs, labels):
    logits, _ = outputs
    return F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1))

# 3. Load a dataset
dataset = TinyStoriesDataset(split="train", seq_len=256)
dataset.prepare_data()

# 4. Run the full pipeline
result = flops_fit.find_optimal(
    model_cls=make_gpt,
    model_size_param="d_model",
    dataset=dataset,
    loss_fn=loss_fn,
    compute_budgets=[1e18, 3e18, 1e19, 3e19, 1e20],
)

# 5. Inspect results
print(result.chinchilla_table())
result.plot()
```

## Model Contract

Your model class (or factory function) must:

1. **Accept `model_size_param` as a constructor argument** — `find_optimal()` will call `model_cls(<size_param>=N)` to instantiate models at different sizes.
2. **Expose a `num_params() -> int` method** — used to measure actual parameter counts for sweep planning.

Everything else (architecture, output format, loss function) is up to you.

## IsoFLOPs Method

The IsoFLOPs approach from Chinchilla:

1. **Fix compute budget C** (in FLOPs)
2. **Vary model size N** and training tokens D such that `6 * N * D ≈ C`
3. **Train each configuration** and record final loss
4. **Find optimal N** for each compute budget (minimum loss)
5. **Fit power law**: `N_opt = k * C^a` across compute budgets

This reveals the optimal model size for any target compute budget.

## Result Object

`find_optimal()` returns a `Result` with three methods:

```python
# Chinchilla-style markdown table of optimal (N, D) pairs
result.chinchilla_table()

# Predict optimal config for a specific compute budget
config = result.predict(1e21)
# {'target_compute': 1e21, 'optimal_params': ..., 'optimal_tokens': ..., 'expected_loss': ...}

# Generate and save IsoFLOPs curves + scaling law plots
result.plot(show=True)
```

When `train=False` or `dataset`/`loss_fn` are omitted, `find_optimal()` returns a `SweepPlan` for inspection before committing to training.

## Examples

### GPT + TinyStories (language modeling)

```bash
python -m flops_fit.examples.example_programmatic          # mock mode (no GPU)
python -m flops_fit.examples.example_programmatic --real   # real training
```

### ViT + CIFAR-10 (image classification)

```bash
python -m flops_fit.examples.example_vit_cifar             # mock mode (no GPU)
python -m flops_fit.examples.example_vit_cifar --real      # real training
```

### CLI wrapper pattern

```bash
python -m flops_fit.examples.example_cli_wrapper --budgets 1e18 3e18 1e19 --layers 6
```

### Mock mode (no GPU or data download)

Pass `mode="mock"` to skip real training and generate synthetic losses — useful for testing the pipeline end-to-end:

```python
result = flops_fit.find_optimal(
    model_cls=make_gpt,
    model_size_param="d_model",
    dataset=synthetic_dataset,
    loss_fn=loss_fn,
    compute_budgets=[1e12, 3e12, 1e13],
    mode="mock",
)
```

## CLI Tools

The four pipeline stages are also available as standalone commands for Hydra-based workflows:

```bash
ff-plan --help      # generate sweep configurations
ff-train --help     # execute training runs
ff-analyze --help   # fit power laws
ff-visualize --help # generate plots
```

Configuration lives in `src/flops_fit/conf/`. Override any value on the command line:

```bash
ff-plan compute.min_flops=1e18 compute.max_flops=1e22
```

## Project Structure

```
flops-fit/
├── src/flops_fit/
│   ├── api.py          # find_optimal() entry point
│   ├── sweep.py        # IsoFLOP sweep planning
│   ├── trainer.py      # Training execution (local + mock modes)
│   ├── analyzer.py     # Power law fitting
│   ├── visualizer.py   # Plot generation
│   ├── result.py       # Result object
│   ├── conf/           # Hydra config files (CLI tools)
│   └── examples/       # GPT, ViT, TinyStories, CIFAR-10
├── tests/
└── pyproject.toml
```

## References

- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla / Hoffmann et al.)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al.)
