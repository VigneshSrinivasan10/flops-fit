# flops-fit

## What This Is

A public, open-source Python library for finding compute-optimal model size (N), data size (D), and predicted loss for any given compute budget. Users install flops-fit, pass their own model class, dataset, and loss function as Python objects, and the library orchestrates the sweep experiments, fits power laws, and returns Chinchilla-style predictions. Supports text and image data.

## Core Value

Given a compute budget, tell the user exactly how big their model should be and how much data to train on — for their specific architecture and dataset.

## Interface

**Primary:** Python library API

```python
import flops_fit
from my_project import MyModel, MyDataset, my_loss

result = flops_fit.find_optimal(
    model_cls=MyModel,
    model_size_param="d_model",
    model_kwargs={"num_layers": 12, "vocab_size": 50257},
    dataset=MyDataset("./data"),
    loss_fn=my_loss,
    compute_budgets=[1e17, 1e18, 1e19],
)

result.chinchilla_table()
result.plot()
```

**Secondary:** CLI exists only as an example of how to use the library.

## Requirements

### Validated

- ✓ IsoFLOP sweep planning with configurable compute budgets — existing
- ✓ Training execution with mock/local modes and resume support — existing
- ✓ Power law fitting (N_opt, D_opt, L_opt vs compute) via scipy — existing
- ✓ Scaling law visualization (IsoFLOP curves, scaling law plots) — existing
- ✓ GPT model with u-mup parameterization for scaling experiments — existing

### Active

- [ ] Python library API: `flops_fit.find_optimal()` takes model class, dataset, loss, compute budgets as Python objects
- [ ] Model interface: user passes a model class + size parameter name + kwargs; library creates models at different sizes
- [ ] Dataset interface: user passes a dataset object; library handles batching and iteration
- [ ] Loss interface: user passes a loss callable
- [ ] Results object: `.chinchilla_table()`, `.plot()`, `.predict(compute_budget)`
- [ ] Multi-GPU training support via data parallelism
- [ ] GPT + TinyStories as a built-in example (shows how to use the library)
- [ ] ViT + CIFAR as a second example (proves image support)
- [ ] Chinchilla table output: optimal N, D, and loss for a range of compute budgets
- [ ] Automatic outlier detection before fitting

### Out of Scope

- Non-text/non-image modalities (point clouds, audio, etc.) — defer to future
- Config-driven architecture (compose layers from YAML) — users write Python classes
- Distributed training beyond single-node data parallelism — too complex for v1
- Web dashboard — researchers use W&B/TensorBoard
- Cloud job submission — users have their own cluster infrastructure
- Auto HP tuning across scales — that's what u-mup solves

## Context

- Existing codebase is a working 4-stage pipeline (plan → train → analyze → visualize) hardcoded to GPT + TinyStories
- Uses Hydra for CLI configuration, PyTorch for models, scipy for curve fitting, matplotlib for plots
- The library API will be the primary interface; the existing CLI/Hydra becomes an example wrapper
- Python 3.11, uv package manager, pytest + ruff for testing/linting

## Constraints

- **Stack**: Python 3.11, PyTorch, scipy, matplotlib
- **Modalities**: Text and image data only for v1
- **Interface**: Python-first — users pass Python objects, not config file paths
- **Model contract**: Model class must accept a size parameter and expose `num_params()`

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Library-first, not CLI-first | Users embed flops-fit in their codebase, not the other way around | Decided |
| Python objects as input, not YAML/config | More Pythonic; config file is optional convenience, not primary API | Decided |
| User provides model class + size param name | Library varies the size param to create models at different scales | Decided |
| Multi-GPU via data parallelism | Covers most scaling law experiments without distributed complexity | Pending |
| Existing GPT + CLI becomes example, not core | Clean separation — library core has no hardcoded architectures | Decided |
| Text + image modalities for v1 | Covers most ML research; other modalities work if user handles data | Decided |

---
*Last updated: 2026-02-16 after pivot to library-first approach*
