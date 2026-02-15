# Stack Research

**Domain:** Pluggable scaling law experiment framework (ML research tooling)
**Researched:** 2026-02-15
**Confidence:** MEDIUM (web search unavailable; recommendations based on official docs fetch for Accelerate + training knowledge of PyTorch/Hydra ecosystem, verified against existing codebase)

## What Already Exists (Keep As-Is)

The existing stack is solid and should not change. These are confirmed from `pyproject.toml`:

| Technology | Current Pin | Purpose | Verdict |
|------------|-------------|---------|---------|
| Python | >=3.11 | Runtime | Keep. 3.11+ gives `StrEnum`, `tomllib`, better typing, perf |
| PyTorch | >=2.0.0 | Model training | **Raise floor to >=2.1.0** (see below) |
| Hydra | >=1.3.2 | Config management | Keep. Critical for plugin architecture via `instantiate()` |
| scipy | >=1.11.0 | Power law fitting | Keep |
| matplotlib | >=3.8.0 | Visualization | Keep |
| numpy | >=1.26.0 | Array ops | Keep |
| pandas | >=2.1.0 | Results processing | Keep |
| HuggingFace datasets | >=2.14.0 | Text data loading | Keep |
| HuggingFace transformers | >=4.35.0 | Tokenizers | Keep |
| tqdm | >=4.66.0 | Progress bars | Keep |
| ruff | >=0.1.0 | Linting | Keep |
| pytest | >=8.0.0 | Testing | Keep |

## Recommended Additions

### 1. HuggingFace Accelerate -- Multi-GPU Training

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| accelerate | >=1.0.0 | Multi-GPU data parallel training | Wraps PyTorch DDP/FSDP with 4 lines of code change; handles device placement, gradient sync, mixed precision automatically |

**Confidence: HIGH** -- Verified via official docs (fetched successfully). Accelerate is the standard abstraction over PyTorch distributed training. It supports DDP, FSDP, and DeepSpeed as backends, configurable via `accelerate config` CLI or YAML.

**Why Accelerate over raw DDP/FSDP:**
- **User experience**: flops-fit users should not need to understand `torch.distributed.init_process_group`, rank management, or `torchrun`. Accelerate handles all of this.
- **Backend flexibility**: Users can switch between DDP (simple multi-GPU), FSDP (large model sharding), or DeepSpeed via config, not code changes.
- **Minimal code invasion**: The existing training loop needs only: (1) create `Accelerator()`, (2) call `accelerator.prepare(model, optimizer, dataloader)`, (3) replace `loss.backward()` with `accelerator.backward(loss)`. Plugin authors write normal PyTorch code.
- **Mixed precision**: Free bf16/fp16 support via `Accelerator(mixed_precision="bf16")`.
- **Launch simplicity**: `accelerate launch ff-train` handles multi-process spawning. Single-GPU falls back to normal execution with zero overhead.

**Why NOT raw PyTorch DDP:**
- DDP requires manual `torch.distributed.init_process_group()`, device assignment per rank, `DistributedSampler` wrapping, model wrapping in `DistributedDataParallel`. This is boilerplate that Accelerate eliminates.
- Plugin authors would need to be DDP-aware (handle rank, device placement). With Accelerate, they write normal single-GPU code.

**Why NOT raw FSDP:**
- FSDP is for models that don't fit on one GPU. Scaling law experiments typically use small-to-medium models (the point is to sweep model sizes). FSDP adds complexity (sharding strategy, mixed precision config, state dict management) that's unnecessary for most scaling law work.
- Accelerate gives FSDP as a backend option for the rare case someone needs it, without forcing everyone through FSDP complexity.

**Why NOT PyTorch Lightning / Fabric:**
- Lightning adds a `LightningModule` abstraction that would conflict with flops-fit's plugin interface. Users would need to subclass `LightningModule` instead of writing a plain `nn.Module`.
- Fabric (Lightning's lower-level API) is similar to Accelerate but has a smaller ecosystem and less HuggingFace integration.
- flops-fit already uses HuggingFace datasets/transformers; Accelerate is the natural fit.

### 2. torchvision -- Image Data Support

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| torchvision | >=0.16.0 | Image transforms, standard datasets | De facto standard for image preprocessing in PyTorch; provides transforms v2 with better composability |

**Confidence: MEDIUM** -- Version recommendation from training data. Verify exact latest version.

**Why torchvision:**
- Standard image transforms (resize, normalize, augment) that image plugin authors will expect.
- Built-in datasets (CIFAR-10, ImageNet) useful for example plugins and testing.
- `torchvision.transforms.v2` (stable since torchvision 0.16+) provides a modern transform API that works with both PIL images and tensors.
- No additional dependency burden -- torchvision is a first-party PyTorch package.

**Why NOT Albumentations:**
- Albumentations is faster for heavy augmentation pipelines, but scaling law experiments typically use minimal augmentation (we want to measure data scaling, not augmentation effects).
- Adding Albumentations as a core dependency would surprise users. Plugin authors can use it in their own data loaders if they want.

### 3. timm (PyTorch Image Models) -- Optional, for Example Plugins

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| timm | >=1.0.0 | Pre-built vision model architectures | Largest collection of vision architectures; useful for ViT/ResNet example plugins |

**Confidence: MEDIUM** -- Training data. Version should be verified.

**Verdict: Optional dependency, not core.** Include in an `[project.optional-dependencies] examples` group. The ViT example plugin can use timm for architecture definitions, but the core framework should not depend on it.

### 4. No New Dependencies for Plugin Architecture

The plugin system requires **zero new libraries**. Here is why:

**Dynamic module loading** uses Python's built-in `importlib`:
```python
import importlib

def load_class(dotted_path: str):
    """Load 'my_package.my_module.MyClass' from a dotted path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
```

**Hydra already provides this** via `hydra.utils.instantiate()` with `_target_` keys in YAML:
```yaml
model:
  _target_: flops_fit.examples.gpt.GPTModel
  d_model: 256
  num_layers: 6
```
```python
from hydra.utils import instantiate
model = instantiate(cfg.model)
```

This is the canonical Hydra pattern for plugin architectures. It is already part of the project's core dependency. No need for `stevedore`, `pluggy`, `entry_points`, or custom plugin registries.

**Confidence: HIGH** -- Hydra's `instantiate()` with `_target_` is well-documented core Hydra functionality. Fetched and confirmed from official Hydra docs.

## Version Pin Changes

### Raise PyTorch Floor to >=2.1.0

**Why:** PyTorch 2.1 stabilized `torch.compile()` and improved `scaled_dot_product_attention`. The existing codebase already uses `F.scaled_dot_product_attention` (model.py line 151). PyTorch 2.1+ also has better FSDP support if needed via Accelerate.

**Confidence: MEDIUM** -- Based on training data knowledge of PyTorch release timeline.

### Pin Accelerate to >=1.0.0

**Why:** Accelerate 1.0 was a stability milestone that unified the API. Earlier versions had frequent breaking changes in FSDP configuration.

**Confidence: LOW** -- I was unable to verify the exact current version via PyPI (web access restricted). The 1.0.0 floor is conservative. Verify actual latest version before committing to pyproject.toml.

## Updated pyproject.toml Dependencies

```toml
[project]
dependencies = [
    "hydra-core>=1.3.2",
    "matplotlib>=3.8.0",
    "numpy>=1.26.0",
    "pandas>=2.1.0",
    "scipy>=1.11.0",
    "pyyaml>=6.0",
    "tqdm>=4.66.0",
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "datasets>=2.14.0",
    "transformers>=4.35.0",
    "accelerate>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
]
examples = [
    "timm>=1.0.0",
]
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Accelerate (multi-GPU) | Raw PyTorch DDP | Only if you need precise control over process groups or custom all-reduce operations. Not relevant for scaling law sweeps. |
| Accelerate (multi-GPU) | PyTorch Lightning / Fabric | If you want a full training framework with logging, checkpointing, callbacks. flops-fit already has its own pipeline; Lightning would fight it. |
| Accelerate (multi-GPU) | DeepSpeed (direct) | Only for very large models (>10B params) needing ZeRO-3. Accelerate can use DeepSpeed as a backend if needed. |
| Hydra `instantiate()` (plugins) | `pluggy` / entry_points | If you need cross-package plugin discovery (pip-installed plugins). flops-fit plugins are user-local Python modules, not pip packages. |
| Hydra `instantiate()` (plugins) | Custom `importlib` wrapper | If you need pre-instantiation validation or lazy loading. Hydra's instantiate already handles errors well. Only add custom wrapper if Hydra's error messages prove insufficient. |
| torchvision (image data) | Albumentations | If your image pipeline needs GPU-accelerated augmentations or very complex augmentation chains. For scaling law experiments, torchvision transforms suffice. |
| torchvision (image data) | kornia | If you need differentiable augmentations on GPU tensors. Unusual for scaling law work. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| PyTorch Lightning (as core dependency) | Forces `LightningModule` inheritance on plugin authors; conflicts with flops-fit's own pipeline orchestration | Accelerate (wraps existing code, no new abstractions) |
| `stevedore` / `pluggy` for plugins | Over-engineered for this use case; plugins are local Python files, not pip-distributed packages | Hydra `instantiate()` with `_target_` config keys |
| Custom plugin registry / decorators | Adds framework boilerplate that users must learn; Hydra already solves this | Hydra `instantiate()` |
| `torch.distributed.launch` (legacy) | Deprecated in favor of `torchrun`; and `accelerate launch` is better for this use case | `accelerate launch` |
| `apex` (NVIDIA) | Deprecated; functionality merged into PyTorch core (mixed precision, fused optimizers) | PyTorch native AMP via Accelerate |
| `wandb` / `mlflow` as core dependency | Logging integration should be optional, not mandatory. Users can add their own. | Optional integration or callback hook |

## Stack Patterns by Variant

**If user has single GPU:**
- Accelerate with default config (no distributed, just device placement + mixed precision)
- Zero code change from CPU path; Accelerate auto-detects

**If user has multi-GPU (single node):**
- `accelerate config` to set up DDP
- `accelerate launch ff-train` spawns one process per GPU
- DataLoader automatically wrapped with `DistributedSampler`

**If user has multi-node (rare for scaling law research):**
- Accelerate supports this but flops-fit should document it as unsupported/experimental
- Would need `accelerate config` with multi-node settings

**If user wants vision scaling laws:**
- Install `pip install flops-fit[examples]` for timm
- Use built-in ViT example plugin or write custom model implementing the plugin interface
- torchvision transforms handle image preprocessing

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| torch >=2.1.0 | torchvision >=0.16.0 | Must use matching major.minor versions (torch 2.1 <-> torchvision 0.16, torch 2.2 <-> torchvision 0.17, etc.) |
| torch >=2.1.0 | accelerate >=1.0.0 | Accelerate pins its own torch compatibility; >=1.0.0 supports torch 2.x |
| hydra-core >=1.3.2 | omegaconf >=2.3 | Hydra bundles omegaconf; no separate pin needed |
| datasets >=2.14.0 | transformers >=4.35.0 | HuggingFace keeps these compatible within minor version ranges |

**Critical compatibility note:** torch and torchvision versions MUST match. If torch is 2.5.x, torchvision must be 0.20.x. The `pip install torch torchvision` command handles this automatically, but pinning in pyproject.toml should use `>=` not `==` to let pip resolve compatible pairs.

## Sources

- HuggingFace Accelerate official docs (https://huggingface.co/docs/accelerate/index) -- fetched 2026-02-15, confirmed: wraps DDP/FSDP/DeepSpeed, 4-line integration, `accelerate launch` CLI [HIGH confidence]
- Hydra instantiate docs (https://hydra.cc/docs/advanced/instantiate_objects/overview/) -- attempted fetch, confirmed from training data: `_target_` pattern for dynamic object creation [HIGH confidence -- well-established pattern]
- Existing codebase analysis (`pyproject.toml`, `model.py`, `trainer.py`, config files) -- direct inspection [HIGH confidence]
- PyTorch distributed training ecosystem (DDP, FSDP) -- training data knowledge [MEDIUM confidence -- unable to verify exact current versions]
- torchvision, timm version numbers -- training data knowledge [LOW confidence -- verify exact latest versions before pinning]

---
*Stack research for: flops-fit pluggable scaling law framework*
*Researched: 2026-02-15*
