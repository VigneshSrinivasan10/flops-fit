# flops-fit

## What This Is

A public, open-source Python library for finding compute-optimal model size (N), data size (D), and predicted loss for any given compute budget. Users install flops-fit, pass their own model class, dataset, and loss function as Python objects, and the library orchestrates the sweep experiments, fits power laws, and returns Chinchilla-style predictions. Supports text and image data. Scales to multiple GPUs via HuggingFace Accelerate with no user code changes.

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

- ✓ IsoFLOP sweep planning with configurable compute budgets — v1.0
- ✓ Training execution with mock/local modes and resume support — v1.0
- ✓ Power law fitting (N_opt, D_opt, L_opt vs compute) via scipy — v1.0
- ✓ Scaling law visualization (IsoFLOP curves, scaling law plots) — v1.0
- ✓ GPT model with u-mup parameterization for scaling experiments — v1.0
- ✓ Python library API: `flops_fit.find_optimal()` takes model class, dataset, loss, compute budgets as Python objects — v1.0
- ✓ Model interface: user passes a model class + size parameter name + kwargs; library creates models at different sizes — v1.0
- ✓ Dataset interface: user passes a dataset object; library handles batching and iteration — v1.0
- ✓ Loss interface: user passes a loss callable — v1.0
- ✓ Results object: `.chinchilla_table()`, `.plot()`, `.predict(compute_budget)` — v1.0
- ✓ Multi-GPU training support via data parallelism (HuggingFace Accelerate) — v1.0
- ✓ GPT + TinyStories as a built-in example — v1.0
- ✓ ViT + CIFAR as a second example proving image support — v1.0
- ✓ Chinchilla table output: optimal N, D, and loss for a range of compute budgets — v1.0
- ✓ Automatic outlier detection before fitting — v1.0

### Active

*(None — all v1.0 requirements shipped. Define next milestone with `/gsd:new-milestone`.)*

### Out of Scope

- Non-text/non-image modalities (point clouds, audio, etc.) — defer to future
- Config-driven architecture (compose layers from YAML) — users write Python classes
- Distributed training beyond single-node data parallelism — too complex for v1
- Web dashboard — researchers use W&B/TensorBoard
- Cloud job submission — users have their own cluster infrastructure
- Auto HP tuning across scales — that's what u-mup solves

## Context

**v1.0 shipped 2026-02-18.** ~7,300 LOC Python. 205 tests passing.

- Core library: `src/flops_fit/` — api.py, trainer.py, sweep.py, analyzer.py, result.py, data.py, loss.py, model_factory.py, planner.py, visualizer.py
- Examples: `src/flops_fit/examples/` — gpt.py, vit.py, example_gpt_tinystories.py, example_vit_cifar.py
- Tech stack: Python 3.11, PyTorch, scipy, matplotlib, HuggingFace Accelerate, uv, pytest

**Known issues / tech debt:**
- Accelerate version pin (>=1.0.0) not verified against latest — validate before next release
- Hydra + torchrun conflict needs Compose API workaround — relevant for any future CLI work
- Phases 3 and 4 are implemented but ROADMAP checkboxes were not updated during execution

## Constraints

- **Stack**: Python 3.11, PyTorch, scipy, matplotlib, HuggingFace Accelerate
- **Modalities**: Text and image data only for v1
- **Interface**: Python-first — users pass Python objects, not config file paths
- **Model contract**: Model class must accept a size parameter and expose `num_params()`

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Library-first, not CLI-first | Users embed flops-fit in their codebase, not the other way around | ✓ Good |
| Python objects as input, not YAML/config | More Pythonic; config file is optional convenience, not primary API | ✓ Good |
| User provides model class + size param name | Library varies the size param to create models at different scales | ✓ Good |
| Duck typing for model contract (no base class) | Easier adoption; probe-based validation catches missing methods early | ✓ Good |
| Linear-space NLS over log-space regression | Unbiased when loss has additive baseline (irreducible entropy) | ✓ Good |
| train=True default; mode='local' default | Training is the happy path; mode param preserves all existing tests | ✓ Good |
| Accelerator created per-experiment in _local_train | Avoids stale DDP gradient bucket state across different architectures | ✓ Good |
| Multi-GPU via Accelerate, activated by launch method | No user code changes; python script.py == single GPU, accelerate launch == multi | ✓ Good |
| Existing GPT + CLI becomes example, not core | Clean separation — library core has no hardcoded architectures | ✓ Good |
| Text + image modalities for v1 | Covers most ML research; other modalities work if user handles data | ✓ Good |

---
*Last updated: 2026-02-18 after v1.0 milestone*
