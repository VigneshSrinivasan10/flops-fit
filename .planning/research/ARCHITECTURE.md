# Architecture Research

**Domain:** Pluggable ML experiment framework (scaling law / compute-optimal tools)
**Researched:** 2026-02-15
**Confidence:** MEDIUM (training data only -- WebSearch/WebFetch unavailable; patterns are well-established but versions not verified against current docs)

## Current Architecture

The existing codebase is a linear pipeline with four independent stages connected by JSON files:

```
ff-plan          ff-train          ff-analyze          ff-visualize
   |                |                  |                    |
SweepPlanner -> TrainingRunner -> ScalingLawAnalyzer -> ScalingVisualizer
   |                |                  |                    |
   v                v                  v                    v
sweep.json      results.json    scaling_laws.json     plots/*.png
```

Each stage is a standalone class with a Hydra `@hydra.main` entry point. No shared runtime state. The model (`GPT`) is hardcoded in `model.py`. There is no plugin system, no GPU support, and a single modality (text via TinyStories).

### What Works and Should Be Preserved

- **File-based handoff between stages.** This is the right pattern for experiment pipelines -- stages can be re-run independently, results are inspectable, and there is no fragile in-memory coupling. Keep this.
- **Hydra configuration.** Already in place, and Hydra has a built-in `instantiate` system that is the natural basis for a plugin architecture. Keep and extend this.
- **Dataclass-based result objects** (`ExperimentConfig`, `TrainingResult`, `PowerLawFit`, `ScalingAnalysis`). Clean serialization boundary. Keep these.

### What Must Change

- **Hardcoded `GPT` model** must become one of many possible model plugins.
- **No dataset abstraction** -- dataset config exists in YAML but no code consumes it. Must become a plugin.
- **Loss is embedded in model's `forward()`** -- must be externalized for pluggability.
- **`TrainingRunner` has no real training loop** -- the `local` mode is a stub. The new training loop must be GPU-aware from day one.
- **No FLOP counting abstraction** -- `estimate_model_flops` uses `6*N*D` which is specific to dense transformers. Plugin models need to declare or measure their own FLOPs.

## Recommended Architecture

### System Overview

```
                         User-Provided Plugins
                    ┌──────────┬──────────┬──────────┐
                    │  Model   │ Dataset  │   Loss   │
                    │ Plugin   │ Plugin   │  Plugin  │
                    └────┬─────┴────┬─────┴────┬─────┘
                         │          │          │
                    ┌────┴──────────┴──────────┴─────┐
                    │         Plugin Registry         │
                    │   (Hydra instantiate + ABC)     │
                    └────────────────┬────────────────┘
                                    │
┌───────────┐    ┌─────────────────┴──────────────────┐    ┌────────────┐
│  Planner  │───>│           Training Engine           │───>│  Analyzer  │
│           │    │  ┌─────────┐  ┌──────────────────┐  │    │            │
│ sweep.json│    │  │ Device  │  │   Training Loop   │  │    │scaling_laws│
│           │    │  │ Manager │  │ (single/multi-GPU)│  │    │   .json    │
└───────────┘    │  └─────────┘  └──────────────────┘  │    └─────┬──────┘
                 │                                      │          │
                 │           results.json                │    ┌─────┴──────┐
                 └──────────────────────────────────────┘    │ Visualizer │
                                                             └────────────┘
```

### Component Responsibilities

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| **Plugin Registry** | Resolve `_target_` strings from YAML to Python classes; validate they implement required protocols | Hydra config, all pipeline stages |
| **Model Protocol** | Abstract interface: `forward()`, `num_params()`, `configure_optimizers()` | Training Engine, Planner (for param estimation) |
| **Dataset Protocol** | Abstract interface: returns `torch.utils.data.Dataset` or `IterableDataset` with configurable sequence/sample length | Training Engine |
| **Loss Protocol** | Abstract interface: `__call__(model_output, targets) -> scalar` | Training Engine |
| **Device Manager** | Handle device placement, DDP init/teardown, mixed precision context | Training Engine |
| **Training Engine** | Orchestrate: load plugins, build train loop, manage checkpoints, record results | All plugins, Device Manager, Planner output, Analyzer input |
| **Planner** | Generate sweep configs (unchanged, but now includes plugin class references) | YAML config, sweep.json |
| **Analyzer** | Fit power laws (unchanged logic, but reads from richer results schema) | results.json, scaling_laws.json |
| **Visualizer** | Generate plots (unchanged logic) | results.json, scaling_laws.json |

### Data Flow

```
YAML Config (user)
    │
    │  _target_: mypackage.models.ViT
    │  _target_: mypackage.data.ImageNetLoader
    │  _target_: mypackage.losses.CrossEntropy
    │
    v
Plugin Resolution (Hydra instantiate)
    │
    │  model_cls, dataset_cls, loss_cls
    │
    v
Planner: sweep.json
    │
    │  [{experiment_id, compute_budget, model_size, num_tokens, model_config_overrides}, ...]
    │
    v
Training Engine (per experiment):
    │
    │  1. Instantiate model from plugin with size params
    │  2. Instantiate dataset from plugin
    │  3. Instantiate loss from plugin
    │  4. DeviceManager: place model, set up DDP if multi-GPU
    │  5. Training loop: forward, loss, backward, step
    │  6. Record: final_loss, actual_flops, wall_time, metadata
    │
    v
results.json (enriched)
    │
    │  [{experiment_id, ..., final_loss, model_class, dataset_class, loss_class, device_info}, ...]
    │
    v
Analyzer: scaling_laws.json (unchanged format)
    │
    v
Visualizer: plots/*.png (unchanged)
```

## Recommended Project Structure

```
src/flops_fit/
├── __init__.py
├── protocols.py            # ABC / Protocol definitions for Model, Dataset, Loss
├── registry.py             # Plugin resolution via Hydra instantiate wrapper
├── planner.py              # Sweep planning (minor changes: model_config in ExperimentConfig)
├── engine/                 # New: training engine package
│   ├── __init__.py
│   ├── trainer.py          # Core training loop (replaces old trainer.py)
│   ├── device.py           # DeviceManager: CPU/GPU/DDP setup and teardown
│   └── checkpoint.py       # Checkpoint save/load for crash recovery
├── analyzer.py             # Scaling law fitting (minimal changes)
├── visualizer.py           # Plot generation (minimal changes)
├── plugins/                # Built-in plugins (example implementations)
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── gpt.py          # Existing GPT model, refactored as a plugin
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── text.py          # TinyStories / HF text datasets
│   │   └── image.py         # Image classification datasets
│   └── losses/
│       ├── __init__.py
│       ├── cross_entropy.py # Standard cross-entropy (text)
│       └── image_losses.py  # Image-specific losses
├── conf/
│   ├── planner.yaml
│   ├── trainer.yaml         # Now includes plugin _target_ references
│   ├── analyzer.yaml
│   ├── visualizer.yaml
│   ├── model/               # Hydra config group for model plugins
│   │   ├── gpt.yaml
│   │   └── vit.yaml         # Example image model config
│   ├── dataset/             # Hydra config group for dataset plugins
│   │   ├── tinystories.yaml
│   │   └── imagenet.yaml
│   ├── loss/                # Hydra config group for loss plugins
│   │   ├── cross_entropy.yaml
│   │   └── mse.yaml
│   └── presets/
│       ├── cpu_fast.yaml
│       └── gpu_full.yaml
└── utils/
    ├── __init__.py
    └── flops.py             # FLOP estimation utilities (extracted from model.py)
```

### Structure Rationale

- **`protocols.py` at top level:** Central contract definitions. Every plugin implementer reads this file first. Not buried in a subfolder.
- **`registry.py` at top level:** Thin wrapper around `hydra.utils.instantiate` with validation. One file, not a package.
- **`engine/` as a package:** The training engine is the most complex new component (loop, device management, checkpointing). Splitting into focused modules prevents a 500-line monolith.
- **`plugins/` for built-ins:** The existing GPT + TinyStories become reference implementations. Users see concrete examples of how to implement each protocol.
- **`conf/model/`, `conf/dataset/`, `conf/loss/`:** Hydra config groups. Users select plugins via `model=gpt` or `model=vit` on the command line. This is Hydra's native pattern for swappable component configs.

## Architectural Patterns

### Pattern 1: Protocol-Based Plugin Contracts

**What:** Define `typing.Protocol` (or `abc.ABC`) classes that specify the interface plugins must implement. Do NOT use a custom registry metaclass or decorator -- use Python's built-in structural typing.

**When to use:** Always. Every plugin type (model, dataset, loss) gets a protocol.

**Trade-offs:** Protocols enable duck typing (any class with the right methods works) but give weaker guarantees than ABC. Use `runtime_checkable` Protocol for validation at registration time.

**Example:**
```python
# protocols.py
from typing import Protocol, runtime_checkable, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset


@runtime_checkable
class ScalableModel(Protocol):
    """Interface for models used in scaling law experiments.

    Users implement this protocol. The tool calls these methods.
    """

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass. Returns dict with at least 'logits' key.

        Input batch comes from the dataset plugin. Model and dataset
        must agree on batch format.
        """
        ...

    def num_params(self, non_embedding: bool = True) -> int:
        """Count trainable parameters.

        Args:
            non_embedding: If True, exclude embedding parameters
                          (standard for Chinchilla-style analysis).
        """
        ...

    def configure_optimizers(
        self,
        learning_rate: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        """Return configured optimizer.

        Model owns optimizer config because parameter groups
        (e.g., u-mup scaling) are model-specific.
        """
        ...


@runtime_checkable
class ScalableDataset(Protocol):
    """Interface for datasets used in scaling law experiments."""

    def build(self, split: str = "train") -> Dataset:
        """Return a torch Dataset for the given split.

        The dataset yields dicts consumed by the model's forward().
        Model and dataset must agree on dict keys.
        """
        ...

    def token_count(self) -> Optional[int]:
        """Total tokens/samples in the training set.

        Used by planner to verify compute budgets are feasible.
        Returns None if count is unknown (e.g., streaming dataset).
        """
        ...


@runtime_checkable
class ScalableLoss(Protocol):
    """Interface for loss functions used in scaling law experiments."""

    def __call__(
        self,
        model_output: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute scalar loss from model output and batch.

        Both dicts come from model.forward() and dataset respectively.
        The loss function bridges their vocabularies.
        """
        ...
```

### Pattern 2: Hydra Instantiate for Plugin Resolution

**What:** Use `hydra.utils.instantiate()` with `_target_` fields in YAML to construct plugin objects. This is Hydra's native object construction pattern -- no custom registry needed.

**When to use:** For all plugin instantiation. Replace any manual `importlib` or string-to-class mapping with Hydra instantiate.

**Trade-offs:** Ties plugin loading to Hydra (acceptable since Hydra is already the config system). Users must specify fully-qualified Python paths in YAML, which is slightly verbose but explicit and debuggable.

**Example:**
```yaml
# conf/model/gpt.yaml
_target_: flops_fit.plugins.models.gpt.GPTModel
vocab_size: 50257
max_seq_len: 256
parametrization: u-mup
base_width: 32
# Size params (overridden per experiment by the planner):
d_model: 256
num_layers: 6
num_heads: 4
```

```yaml
# conf/dataset/tinystories.yaml
_target_: flops_fit.plugins.datasets.text.TextDataset
name: roneneldan/TinyStories
tokenizer: gpt2
seq_len: 256
cache_dir: .cache/datasets
```

```yaml
# conf/trainer.yaml
defaults:
  - model: gpt          # picks conf/model/gpt.yaml
  - dataset: tinystories # picks conf/dataset/tinystories.yaml
  - loss: cross_entropy  # picks conf/loss/cross_entropy.yaml

mode: local
# ... rest of trainer config
```

```python
# registry.py
from hydra.utils import instantiate
from flops_fit.protocols import ScalableModel, ScalableDataset, ScalableLoss


def resolve_model(cfg) -> ScalableModel:
    """Instantiate model plugin from Hydra config."""
    model = instantiate(cfg.model)
    if not isinstance(model, ScalableModel):
        raise TypeError(
            f"Model {type(model).__name__} does not implement ScalableModel protocol. "
            f"Required methods: forward(), num_params(), configure_optimizers()"
        )
    return model


def resolve_dataset(cfg) -> ScalableDataset:
    """Instantiate dataset plugin from Hydra config."""
    dataset = instantiate(cfg.dataset)
    if not isinstance(dataset, ScalableDataset):
        raise TypeError(
            f"Dataset {type(dataset).__name__} does not implement ScalableDataset protocol. "
            f"Required methods: build(), token_count()"
        )
    return dataset


def resolve_loss(cfg) -> ScalableLoss:
    """Instantiate loss plugin from Hydra config."""
    loss = instantiate(cfg.loss)
    if not isinstance(loss, ScalableLoss):
        raise TypeError(
            f"Loss {type(loss).__name__} does not implement ScalableLoss protocol. "
            f"Required methods: __call__(model_output, batch)"
        )
    return loss
```

### Pattern 3: Device Manager for GPU/DDP Abstraction

**What:** A single `DeviceManager` class that encapsulates all device-related logic: CPU/GPU selection, DDP initialization, model wrapping, mixed precision context. The training loop calls `DeviceManager` methods and never touches `torch.distributed` directly.

**When to use:** Always. Even for CPU-only runs, the DeviceManager provides a uniform interface.

**Trade-offs:** Adds a layer of indirection. Worth it because DDP boilerplate is error-prone and the training loop stays clean.

**Example:**
```python
# engine/device.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import contextmanager


class DeviceManager:
    """Manages device placement, DDP, and mixed precision."""

    def __init__(self, cfg):
        self.device_str = cfg.hardware.device  # "cpu", "cuda", "cuda:0"
        self.num_gpus = cfg.hardware.get("num_gpus", 1)
        self.dtype = getattr(torch, cfg.hardware.get("dtype", "float32"))
        self.use_ddp = self.num_gpus > 1
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0

    def setup(self):
        """Initialize device and DDP if needed."""
        if self.use_ddp:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.rank = int(os.environ.get("RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            dist.init_process_group(backend="nccl")
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        elif "cuda" in self.device_str:
            self.device = torch.device(self.device_str)
        else:
            self.device = torch.device("cpu")

    def teardown(self):
        """Clean up DDP."""
        if self.use_ddp:
            dist.destroy_process_group()

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Move model to device and wrap with DDP if needed."""
        model = model.to(self.device)
        if self.use_ddp:
            model = DDP(model, device_ids=[self.local_rank])
        return model

    def prepare_dataloader(self, dataset, batch_size, **kwargs):
        """Create DataLoader with DistributedSampler if needed."""
        sampler = None
        if self.use_ddp:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, shuffle=True)
            kwargs.pop("shuffle", None)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            **kwargs,
        )

    @property
    def is_main_process(self) -> bool:
        """True on rank 0 (use for logging, saving)."""
        return self.rank == 0

    @contextmanager
    def autocast(self):
        """Mixed precision context manager."""
        if self.dtype != torch.float32 and self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                yield
        else:
            yield
```

### Pattern 4: Batch Dict Convention for Multimodal Support

**What:** All data flows between dataset, model, and loss as `Dict[str, torch.Tensor]`. The dataset produces a dict, the model consumes and produces a dict, and the loss consumes both dicts. This decouples the three plugins from each other.

**When to use:** Always. This is the key pattern that makes multimodal work without framework changes.

**Trade-offs:** Dicts are less type-safe than named tuples or dataclasses. But they are maximally flexible -- a text dataset produces `{"input_ids": ..., "labels": ...}`, an image dataset produces `{"pixel_values": ..., "labels": ...}`, and the model/loss just look for the keys they need. No code changes in the training engine when adding a new modality.

**Why this beats alternatives:**
- Named tuples/dataclasses: Would require modality-specific types, breaking the generic pipeline.
- Separate model signatures per modality: Would require if/else branching in the training loop.
- Dict convention: The training engine does `model_output = model.forward(batch)` and `loss = loss_fn(model_output, batch)` regardless of modality. The contract is between the specific model/dataset/loss plugins, not the framework.

**Example:**
```python
# Text plugin batch: {"input_ids": [B, T], "labels": [B, T]}
# Image plugin batch: {"pixel_values": [B, C, H, W], "labels": [B]}
# Multimodal batch: {"input_ids": [B, T], "pixel_values": [B, C, H, W], "labels": [B]}

# Training engine -- modality-agnostic:
for batch in dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    model_output = model(batch)          # model knows what keys to use
    loss = loss_fn(model_output, batch)  # loss knows what keys to use
    loss.backward()
    optimizer.step()
```

## How the Existing Pipeline Should Evolve

### Migration Strategy: Incremental, Not Rewrite

The pipeline has the right bones. The evolution is:

1. **Extract protocols** -- define `ScalableModel`, `ScalableDataset`, `ScalableLoss` in `protocols.py`
2. **Wrap existing GPT as plugin** -- move `model.py` to `plugins/models/gpt.py`, adapt to protocol
3. **Build training engine** -- implement real training loop in `engine/trainer.py` with DeviceManager
4. **Add Hydra config groups** -- `conf/model/`, `conf/dataset/`, `conf/loss/` with `_target_` fields
5. **Wire registry** -- `registry.py` uses `hydra.utils.instantiate` + protocol validation
6. **Add image plugins** -- implement `ScalableModel` for a simple CNN/ViT, `ScalableDataset` for image data
7. **Add multi-GPU** -- DeviceManager with DDP support

### What Changes in Each Stage

**Planner (minor changes):**
- `ExperimentConfig` gains optional `model_config_overrides` dict for per-experiment model size params
- Planner must know how to map "target N params" to model-specific size params (d_model, num_layers). This is plugin-specific. Solution: the model plugin provides a `classmethod size_config(target_params: int) -> dict` that returns the config overrides needed to hit approximately that parameter count.

**Training Engine (replaces TrainingRunner):**
- Completely new training loop that uses plugins via protocols
- DeviceManager for CPU/GPU/DDP
- Checkpoint support for crash recovery
- Reads sweep.json, writes results.json (same file-based handoff)

**Analyzer (minimal changes):**
- No changes to core logic -- it operates on results.json which has the same key fields
- Add `model_class` and `dataset_class` to results metadata for provenance

**Visualizer (minimal changes):**
- No changes to core logic
- Optionally display plugin class names in plot titles/legends

## Anti-Patterns

### Anti-Pattern 1: Custom Registry with Decorators

**What people do:** Build a `@register_model("gpt")` decorator system with a global dict mapping names to classes.

**Why it's wrong:** Duplicates what Hydra instantiate already does. Creates a parallel naming system that can diverge from config. Requires importing the module to trigger registration, leading to import-order bugs.

**Do this instead:** Use Hydra's `_target_` field in YAML. The fully-qualified Python path IS the registration. No decorator needed. No global mutable state.

### Anti-Pattern 2: Modality-Specific Training Loops

**What people do:** Write `train_text()`, `train_image()`, `train_multimodal()` with separate loops.

**Why it's wrong:** Every new modality requires a new loop. Logic diverges. Bugs get fixed in one loop but not others.

**Do this instead:** One training loop that speaks `Dict[str, torch.Tensor]`. Modality differences are handled entirely within plugins.

### Anti-Pattern 3: Model Owns Loss Computation

**What people do:** Put loss computation inside `model.forward()` (as the current GPT does with `labels` parameter).

**Why it's wrong:** Couples model to a specific loss function. Cannot swap loss without modifying model. Cannot use the same model with different loss functions for different experiments.

**Do this instead:** Model returns `{"logits": ...}` (or whatever its output is). A separate loss plugin computes loss from model output and batch. The training engine wires them together.

### Anti-Pattern 4: DDP Logic Scattered Through Training Code

**What people do:** Sprinkle `if dist.is_initialized()` checks throughout the training loop, data loading, logging, and checkpointing.

**Why it's wrong:** Fragile, easy to miss a check, hard to test. Couples distributed awareness to every component.

**Do this instead:** Centralize all distributed logic in DeviceManager. Training loop calls `device_mgr.prepare_model()`, `device_mgr.prepare_dataloader()`, `device_mgr.is_main_process`. No distributed imports in the training loop itself.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| CPU, 1-2 experiments | DeviceManager on CPU. No DDP. Batch dict convention still applies. |
| Single GPU, full sweep | DeviceManager on CUDA. Mixed precision via autocast. Same loop. |
| Multi-GPU, single node | DeviceManager with DDP. torchrun launcher. DistributedSampler. |
| Multi-node (out of scope) | Would require FSDP or DeepSpeed. Not planned for v1. Document as future extension. |

### Scaling Priorities

1. **First bottleneck: Training speed on CPU.** Current estimates show 50-100 hours for a full sweep on CPU. Single GPU cuts this by 10-50x. Implement GPU support first.
2. **Second bottleneck: Large model memory.** For sweeps that include 50M+ parameter models, single GPU memory may be tight. DDP across GPUs solves this with data parallelism (each GPU holds a full model replica). For truly large models, would need FSDP -- but this is out of scope for scaling law experiments which use relatively small models.
3. **Third bottleneck: Sweep parallelism.** Different experiments in the sweep are independent. Future optimization: run multiple experiments concurrently on different GPUs rather than DDP on a single experiment.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Hugging Face Hub | `datasets` library for text data loading | Already a dependency. Dataset plugins can use it or not. |
| torchvision | Image dataset loading | Add as optional dependency for image plugins |
| Weights & Biases / TensorBoard | Optional experiment tracking | NOT in v1. Results.json is the tracking system. Can add later as a logging plugin. |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Planner -> Training Engine | sweep.json file (JSON) | Add `model_config_overrides` to experiment configs |
| Training Engine -> Analyzer | results.json file (JSON) | Add plugin metadata fields |
| Config -> Plugins | Hydra instantiate (`_target_`) | Users specify fully-qualified Python paths |
| DeviceManager -> Training Loop | Method calls (`prepare_model`, `is_main_process`) | Training loop never imports torch.distributed |
| Model <-> Dataset | Batch dict convention (`Dict[str, Tensor]`) | Model and dataset must agree on keys; loss bridges them |

## Build Order (Dependencies)

The components have clear dependencies that determine build order:

```
1. protocols.py              (no deps -- define interfaces first)
       |
2. plugins/models/gpt.py    (depends on protocols -- refactor existing GPT)
   plugins/datasets/text.py  (depends on protocols -- wrap existing dataset config)
   plugins/losses/cross_entropy.py (depends on protocols -- extract from GPT.forward)
       |
3. registry.py              (depends on protocols -- Hydra instantiate wrapper)
   conf/model/gpt.yaml      (depends on plugin existing)
   conf/dataset/tinystories.yaml
   conf/loss/cross_entropy.yaml
       |
4. engine/device.py         (depends on nothing -- standalone)
       |
5. engine/trainer.py        (depends on protocols, registry, device -- core training loop)
       |
6. Planner updates          (depends on protocols -- add size_config classmethod support)
       |
7. engine/checkpoint.py     (depends on engine/trainer -- adds crash recovery)
       |
8. Image plugins            (depends on protocols, proven working with text)
       |
9. Multi-GPU (DDP)          (depends on engine/device -- extend DeviceManager)
```

**Critical path:** Steps 1-5 must be sequential. Steps 6-9 can partially parallelize once step 5 is working.

**Key insight:** The plugin system (steps 1-3) and training engine (steps 4-5) should be built together and tested with the existing GPT+TinyStories before adding new plugins (step 8) or DDP (step 9). This keeps the "does the basic pipeline still work?" question answered at every step.

## FLOP Estimation Strategy

The current `estimate_model_flops` uses `C = 6*N*D` which is a dense-transformer approximation. For pluggable architectures:

**Recommended approach:** The `ScalableModel` protocol does NOT include a FLOP estimation method. Instead:

1. The training engine records `num_params` (from protocol) and `num_tokens_seen` (counted during training).
2. The default FLOP estimate remains `6*N*D` as a reasonable approximation for most architectures.
3. Optionally, a model plugin can override by providing a `estimate_flops(num_tokens: int) -> int` method. The registry checks for this optional method.

This avoids burdening every plugin author with FLOP estimation while allowing precise estimates when needed.

## Model Size Configuration Strategy

For the planner to generate experiments at different model sizes, it needs to know how to scale a given architecture. Different architectures scale differently (transformers scale width/depth, CNNs scale channels/layers).

**Recommended approach:** The `ScalableModel` protocol includes an optional classmethod:

```python
@classmethod
def size_configs(cls, target_params: list[int]) -> list[dict]:
    """Return config overrides to hit each target parameter count.

    Args:
        target_params: List of target parameter counts.

    Returns:
        List of config override dicts (same length as target_params).
        Each dict contains constructor kwargs that produce a model
        near the target size.
    """
    ...
```

The planner calls this to generate per-experiment model configs. If the method is not implemented, the planner falls back to asking the user to provide explicit size configs in YAML.

## Sources

- Hydra instantiate pattern: Based on Hydra documentation for `hydra.utils.instantiate` with `_target_` field (MEDIUM confidence -- well-established pattern, not verified against latest Hydra version)
- PyTorch DDP pattern: Based on PyTorch DistributedDataParallel documentation (MEDIUM confidence -- stable API since PyTorch 1.x, specifics not verified against PyTorch 2.x docs)
- Protocol-based plugin pattern: Standard Python `typing.Protocol` pattern (HIGH confidence -- core Python feature since 3.8)
- Batch dict convention: Common pattern in Hugging Face Transformers and other ML frameworks (MEDIUM confidence -- based on training data, not verified)
- `6*N*D` FLOP estimation: From Kaplan et al. (2020) and Hoffmann et al. (2022) scaling law papers (HIGH confidence -- well-established in literature)
- Existing codebase analysis: Direct inspection of source code (HIGH confidence)

---
*Architecture research for: flops-fit pluggable ML experiment framework*
*Researched: 2026-02-15*
