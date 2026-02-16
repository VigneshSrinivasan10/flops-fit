# Phase 2: Dataset and Loss Interfaces - Research

**Researched:** 2026-02-16
**Domain:** PyTorch Dataset/DataLoader wrapping, loss callable validation, interface contracts
**Confidence:** HIGH

## Summary

Phase 2 adds two input interfaces to `find_optimal()`: a dataset parameter and a loss function parameter. The user passes a PyTorch `Dataset` or `DataLoader` and a callable loss function. The library must validate both at call time (not deep in training) and handle the Dataset-to-DataLoader wrapping internally.

The core technical challenges are: (1) accepting either a `Dataset` or a `DataLoader` and normalizing to a `DataLoader` internally, (2) validating that the loss callable has a compatible signature before training begins, and (3) providing clear error messages that tell users exactly what interface they need to implement. This phase does NOT build a training loop -- it builds the validation and wrapping layer that Phase 4 (Training Engine) will consume.

The existing `find_optimal()` stub (from Phase 1) already has `dataset=None` and `loss_fn=None` parameters. This phase fills in their validation logic and creates internal helpers for DataLoader wrapping. The existing `TrainingRunner` currently uses mock training with no real dataset -- Phase 4 will connect the new interfaces to actual training.

**Primary recommendation:** Create a `data.py` module with `wrap_dataset(dataset_or_loader, batch_size, ...)` that normalizes Dataset/DataLoader input, and a `validate_loss_fn(loss_fn)` function that checks callability and signature. Integrate validation into `find_optimal()` alongside the existing model validation. Keep validation lightweight -- create no probe batches, just check interfaces.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch.utils.data.Dataset | torch>=2.0.0 | Map-style dataset base class | Already a dependency; the standard PyTorch data interface |
| torch.utils.data.DataLoader | torch>=2.0.0 | Batching, shuffling, parallel loading | Already a dependency; handles all batching complexity |
| inspect | stdlib | Loss function signature validation | No new dependency needed |

### Supporting

No additional libraries needed. This phase uses only PyTorch's data utilities and Python stdlib.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Accept raw Dataset or DataLoader | Accept only DataLoader | Accepting only DataLoader is simpler but forces users to handle batching themselves. Accepting both is friendlier -- most scaling law users have a Dataset, not a pre-configured DataLoader. |
| `inspect.signature` for loss validation | Try/except on a probe call | `inspect.signature` catches problems without needing probe data. A probe call would require fabricating tensors of the right shape, which we don't know at validation time. |
| Duck typing (hasattr checks) | isinstance(dataset, Dataset) | isinstance is more precise and what PyTorch itself uses internally. Duck typing would accept objects that coincidentally have `__getitem__` but aren't real datasets. |

**Installation:**
```bash
# No new dependencies -- torch already in pyproject.toml
```

## Architecture Patterns

### Recommended Project Structure Changes

```
src/flops_fit/
    __init__.py          # UNCHANGED
    api.py               # MODIFY: add dataset/loss validation calls
    model_factory.py     # UNCHANGED (from Phase 1)
    data.py              # NEW: dataset wrapping + validation
    loss.py              # NEW: loss function validation
    model.py             # UNCHANGED (existing GPT)
    planner.py           # UNCHANGED
    trainer.py           # UNCHANGED
    analyzer.py          # UNCHANGED
    visualizer.py        # UNCHANGED
    conf/                # UNCHANGED
```

**Rationale:** Two new files (`data.py`, `loss.py`) keep concerns separated. `data.py` handles Dataset/DataLoader normalization. `loss.py` handles loss callable validation. Both are internal modules -- they are not exported from `__init__.py` because users don't interact with them directly. `api.py` calls them during `find_optimal()` validation.

**Alternative:** A single `validation.py` could hold both. However, `data.py` will grow in Phase 4 when actual DataLoader configuration (num_workers, pin_memory, device placement) is needed. Keeping it separate now avoids a future split.

### Pattern 1: Dataset/DataLoader Normalization

**What:** A function that accepts either a `torch.utils.data.Dataset` or a `torch.utils.data.DataLoader` and returns a `DataLoader`. If the input is already a DataLoader, it passes through. If it's a Dataset, the library wraps it in a DataLoader with sensible defaults.

**When to use:** At `find_optimal()` call time, before any training.

**Example:**
```python
# data.py
import torch.utils.data as data


def wrap_dataset(dataset_or_loader, batch_size=32, num_workers=0, shuffle=True):
    """Normalize a Dataset or DataLoader into a DataLoader.

    Args:
        dataset_or_loader: A torch.utils.data.Dataset or DataLoader
        batch_size: Batch size (ignored if input is already a DataLoader)
        num_workers: Number of data loading workers (ignored if DataLoader)
        shuffle: Whether to shuffle (ignored if DataLoader)

    Returns:
        A DataLoader ready for iteration

    Raises:
        TypeError: If input is neither a Dataset nor a DataLoader
    """
    if isinstance(dataset_or_loader, data.DataLoader):
        return dataset_or_loader

    if isinstance(dataset_or_loader, data.Dataset):
        return data.DataLoader(
            dataset_or_loader,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True,  # Important for consistent batch shapes
        )

    raise TypeError(
        f"dataset must be a torch.utils.data.Dataset or "
        f"torch.utils.data.DataLoader, got {type(dataset_or_loader).__name__}. "
        f"Wrap your data in a torch.utils.data.Dataset subclass with "
        f"__getitem__ and __len__ methods."
    )
```

### Pattern 2: Dataset Validation

**What:** Validate that a Dataset has the required interface before training. Check for `__getitem__` and `__len__` on map-style datasets. For IterableDataset, check for `__iter__`.

**When to use:** At `find_optimal()` call time.

**Example:**
```python
def validate_dataset(dataset_or_loader):
    """Validate that a dataset meets the required interface.

    Checks:
    1. Input is a Dataset or DataLoader (type check)
    2. Map-style datasets have __len__ (needed for epoch calculation)
    3. DataLoaders are usable (have a dataset attribute)

    Raises:
        TypeError: If input fails validation
    """
    if isinstance(dataset_or_loader, data.DataLoader):
        # DataLoader is already valid -- it was constructed with a Dataset
        return

    if isinstance(dataset_or_loader, data.IterableDataset):
        # IterableDataset is accepted but needs special handling
        # in the training loop (no __len__, no random access)
        return

    if isinstance(dataset_or_loader, data.Dataset):
        # Map-style dataset -- verify __len__ exists
        if not hasattr(dataset_or_loader, '__len__'):
            raise TypeError(
                f"{type(dataset_or_loader).__name__} is a Dataset but does "
                f"not implement __len__. flops_fit needs __len__ to "
                f"calculate training epochs from token budgets."
            )
        return

    raise TypeError(
        f"dataset must be a torch.utils.data.Dataset or "
        f"torch.utils.data.DataLoader, got {type(dataset_or_loader).__name__}."
    )
```

### Pattern 3: Loss Function Validation

**What:** Validate that the loss function is callable and has a compatible signature. The library needs loss functions that accept `(model_output, targets)` or similar -- but since models and loss functions vary widely, we validate minimally: the thing must be callable.

**When to use:** At `find_optimal()` call time.

**Example:**
```python
# loss.py
import inspect


def validate_loss_fn(loss_fn):
    """Validate that a loss function is usable.

    Checks:
    1. loss_fn is callable (function, nn.Module instance, or class with __call__)
    2. loss_fn accepts at least 2 positional arguments (predictions, targets)

    Raises:
        TypeError: If loss_fn is not callable or has wrong signature
    """
    if loss_fn is None:
        raise TypeError(
            "loss_fn is required. Pass a callable that computes loss from "
            "model outputs and targets.\n"
            "Examples:\n"
            "  loss_fn=torch.nn.CrossEntropyLoss()\n"
            "  loss_fn=lambda logits, labels: F.cross_entropy(logits, labels)\n"
            "  loss_fn=my_custom_loss  # any callable(predictions, targets) -> scalar"
        )

    if not callable(loss_fn):
        raise TypeError(
            f"loss_fn must be callable, got {type(loss_fn).__name__}. "
            f"Pass a function, nn.Module, or any object with a __call__ method."
        )

    # Check signature accepts at least 2 positional arguments
    # This catches common mistakes like passing a class instead of an instance
    try:
        sig = inspect.signature(loss_fn)
        # Count parameters that can be passed positionally
        positional_params = [
            p for p in sig.parameters.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        # Also count *args
        has_var_positional = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL
            for p in sig.parameters.values()
        )

        if len(positional_params) < 2 and not has_var_positional:
            raise TypeError(
                f"loss_fn must accept at least 2 positional arguments "
                f"(predictions, targets), but {_get_name(loss_fn)} accepts "
                f"{len(positional_params)}. "
                f"Expected signature: loss_fn(predictions, targets) -> scalar"
            )
    except (ValueError, TypeError):
        # Some callables (C extensions, some builtins) don't support
        # inspect.signature. That's fine -- we'll catch errors at training time.
        pass


def _get_name(loss_fn):
    """Get a human-readable name for a loss function."""
    if hasattr(loss_fn, '__name__'):
        return loss_fn.__name__
    return type(loss_fn).__name__
```

### Pattern 4: Integrating Validation into find_optimal()

**What:** Update `api.py` to validate dataset and loss_fn alongside the existing model validation.

**Example:**
```python
# api.py (updated)
def find_optimal(
    model_cls,
    model_size_param,
    model_kwargs=None,
    dataset=None,
    loss_fn=None,
    compute_budgets=None,
    **kwargs,
):
    if model_kwargs is None:
        model_kwargs = {}

    # Validate model contract (Phase 1)
    validate_model_contract(model_cls, model_size_param, model_kwargs)

    # Validate dataset interface (Phase 2)
    if dataset is not None:
        validate_dataset(dataset)

    # Validate loss function (Phase 2)
    if loss_fn is not None:
        validate_loss_fn(loss_fn)

    # Phase 3+: Actual sweep execution
    raise NotImplementedError(
        "find_optimal() validation passed. "
        "Full pipeline not yet implemented."
    )
```

**Note on None handling:** Phase 2 validates dataset and loss_fn IF they are provided, but does not require them yet. Phase 4 (Training Engine) will make them required for actual training. This keeps Phase 1's existing behavior intact -- calling `find_optimal()` with only model args still works.

### Anti-Patterns to Avoid

- **Requiring Dataset subclass inheritance:** Users should be able to pass any `torch.utils.data.Dataset` subclass, including `TensorDataset`, `ConcatDataset`, HuggingFace datasets wrapped with `.with_format("torch")`, etc. Do NOT require a specific base class beyond `Dataset`.

- **Creating a probe batch during validation:** Do NOT try to index into the dataset or iterate the DataLoader during validation. This could trigger data loading, downloading, or GPU operations. Validation should only check interfaces, not data.

- **Hardcoding loss function signatures:** Loss functions in the wild have wildly different signatures: `loss(logits, labels)`, `loss(output, target, weight)`, `nn.CrossEntropyLoss()(input, target)`. The library should validate "is callable with >= 2 positional args" and nothing more.

- **Wrapping DataLoader configuration in the validation phase:** Phase 2 validates; Phase 4 configures. Do NOT set num_workers, pin_memory, or device placement in Phase 2. Those belong in the training engine.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Batching | Custom batch iterator | `torch.utils.data.DataLoader` | Handles shuffling, multiprocess loading, memory pinning, collation, drop_last |
| Collation | Custom tensor stacking | `DataLoader(collate_fn=...)` | Default collate handles nested structures, variable types, None values |
| Parallel data loading | Threading/multiprocessing for data | `DataLoader(num_workers=N)` | Handles worker spawning, memory sharing, error propagation |
| Dataset length | Manual counting/caching | `len(dataset)` via `__len__` | Standard Python protocol; DataLoader relies on it for epoch boundaries |

**Key insight:** The PyTorch DataLoader already solves every data loading problem. This phase's job is to validate the user's input and wrap it in a DataLoader if needed -- not to build any data loading machinery.

## Common Pitfalls

### Pitfall 1: IterableDataset vs Map-Style Dataset

**What goes wrong:** User passes an `IterableDataset` (e.g., streaming from HuggingFace datasets), but the library assumes `__len__` exists for epoch/step calculation.

**Why it happens:** `IterableDataset` does not have `__len__`. The training engine needs to know how many tokens to process (from the compute budget), which requires knowing dataset size to calculate epochs.

**How to avoid:** Accept `IterableDataset` but document that the library will iterate it by token count, not by epochs. The training loop (Phase 4) will count tokens consumed and stop when the budget is reached. Validation in Phase 2 should accept IterableDataset without requiring `__len__`.

**Warning signs:** `TypeError: object of type 'MyIterableDataset' has no len()` during training setup.

### Pitfall 2: Loss Function Is a Class, Not an Instance

**What goes wrong:** User passes `torch.nn.CrossEntropyLoss` (the class) instead of `torch.nn.CrossEntropyLoss()` (an instance). The class is callable, but calling it creates an instance rather than computing loss.

**Why it happens:** Easy mistake. Both are callable. `CrossEntropyLoss(logits, targets)` creates a `CrossEntropyLoss` instance with logits as the first constructor arg -- silently wrong.

**How to avoid:** The signature check helps here -- `CrossEntropyLoss.__init__` has no required positional args (just `self`), so calling `CrossEntropyLoss(predictions, targets)` would actually work but produce wrong results. Better detection: check if `loss_fn` is a class (not an instance) via `inspect.isclass(loss_fn)`. If it is, warn: "Did you mean `loss_fn=nn.CrossEntropyLoss()` (with parentheses)?"

**Warning signs:** Loss values that don't decrease, or silent construction of a new loss module every forward pass.

### Pitfall 3: Dataset Returns Wrong Types

**What goes wrong:** Dataset `__getitem__` returns numpy arrays, PIL images, or plain Python lists instead of tensors. DataLoader's default collate function handles some conversions but not all.

**Why it happens:** Users often implement datasets that return raw data, expecting the training code to convert. This is valid -- DataLoader's default `collate_fn` converts numpy arrays to tensors automatically.

**How to avoid:** Do NOT validate tensor types in Phase 2. The default DataLoader collation handles numpy-to-tensor conversion. If a user's data is truly incompatible, the error will surface in Phase 4's training loop with a clear stack trace. Premature validation of return types would reject valid datasets.

**Warning signs:** None in Phase 2 -- this surfaces in Phase 4.

### Pitfall 4: HuggingFace Datasets Compatibility

**What goes wrong:** HuggingFace `datasets.Dataset` is NOT a `torch.utils.data.Dataset` subclass by default. Users need to call `.with_format("torch")` or wrap it.

**Why it happens:** HuggingFace datasets have their own `Dataset` class. `isinstance(hf_dataset, torch.utils.data.Dataset)` returns `False`.

**How to avoid:** The validation error message should mention this: "If using HuggingFace datasets, call `dataset.with_format('torch')` first, or wrap it in a torch Dataset subclass." This is purely a documentation/error-message concern -- the isinstance check correctly rejects non-torch datasets.

**Warning signs:** User reports "but it's a Dataset!" when validation rejects their HuggingFace dataset.

### Pitfall 5: inspect.signature Fails on Built-in/C-Extension Callables

**What goes wrong:** `inspect.signature()` raises `ValueError` for some built-in functions and C extensions (e.g., some compiled loss functions).

**Why it happens:** Not all Python callables have introspectable signatures.

**How to avoid:** Wrap `inspect.signature()` in a try/except. If signature inspection fails, skip the parameter count check and trust that the callable will work at training time. The fallback is a runtime error in Phase 4, which is acceptable.

**Warning signs:** `ValueError: no signature found for builtin type <class '...'>` during validation.

## Code Examples

### Example 1: User Passes a Dataset

```python
import torch
from torch.utils.data import Dataset
import flops_fit

class TextDataset(Dataset):
    def __init__(self, tokens, seq_len=256):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.tokens[start:start + self.seq_len]
        y = self.tokens[start + 1:start + self.seq_len + 1]
        return x, y

dataset = TextDataset(torch.randint(0, 50257, (1_000_000,)))
result = flops_fit.find_optimal(
    model_cls=MyModel,
    model_size_param="d_model",
    model_kwargs={"vocab_size": 50257},
    dataset=dataset,
    loss_fn=torch.nn.CrossEntropyLoss(),
)
```

### Example 2: User Passes a DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
result = flops_fit.find_optimal(
    model_cls=MyModel,
    model_size_param="d_model",
    model_kwargs={"vocab_size": 50257},
    dataset=loader,  # DataLoader is also accepted
    loss_fn=torch.nn.CrossEntropyLoss(),
)
```

### Example 3: Validation Error Messages

```python
# Not a Dataset
flops_fit.find_optimal(
    model_cls=MyModel,
    model_size_param="d_model",
    dataset=[1, 2, 3],  # plain list
    loss_fn=torch.nn.CrossEntropyLoss(),
)
# TypeError: dataset must be a torch.utils.data.Dataset or
# torch.utils.data.DataLoader, got list.

# Loss function is a class, not instance
flops_fit.find_optimal(
    model_cls=MyModel,
    model_size_param="d_model",
    dataset=dataset,
    loss_fn=torch.nn.CrossEntropyLoss,  # missing ()
)
# TypeError: loss_fn appears to be a class, not an instance.
# Did you mean loss_fn=CrossEntropyLoss() (with parentheses)?

# Not callable
flops_fit.find_optimal(
    model_cls=MyModel,
    model_size_param="d_model",
    dataset=dataset,
    loss_fn=42,
)
# TypeError: loss_fn must be callable, got int.
```

### Example 4: Custom Loss Function

```python
import torch.nn.functional as F

# Simple function
def my_loss(logits, labels):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

# nn.Module subclass
class WeightedLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, predictions, targets):
        ce = F.cross_entropy(predictions, targets)
        return self.alpha * ce

# Both work
result = flops_fit.find_optimal(..., loss_fn=my_loss)
result = flops_fit.find_optimal(..., loss_fn=WeightedLoss(alpha=0.3))
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hardcoded TinyStories dataset in config | User passes Dataset/DataLoader object | Phase 2 (this phase) | Any dataset works, not just text |
| Loss computed inside model forward() | User passes loss callable separately | Phase 2 (this phase) | Model and loss are decoupled; same model can be used with different losses |
| `trainer.py` mock mode with synthetic loss | Real dataset/loss interfaces | Phase 2 + Phase 4 | Enables actual training |

**Note:** The existing GPT model computes loss inside `forward()` when `labels` is provided. The library API decouples this -- users pass a separate loss function. The GPT example (Phase 7) will need to adapt by either: (a) using a loss function that calls `model(input_ids, labels)` and extracts the loss, or (b) restructuring the model to return logits only and letting the external loss function handle it. This is a Phase 7 concern, not Phase 2.

## Open Questions

1. **Should dataset and loss_fn be required or optional in Phase 2?**
   - What we know: Phase 1 made them optional (`None` defaults). Phase 4 needs them for training.
   - What's unclear: Should Phase 2 require them (breaking Phase 1 tests) or keep them optional?
   - Recommendation: Keep optional in Phase 2. Validate IF provided, skip if None. Phase 4 will make them required for the actual training path. This preserves backward compatibility and lets users incrementally build up their `find_optimal()` call.

2. **Should the library accept IterableDataset?**
   - What we know: IterableDataset lacks `__len__`, which complicates epoch/step calculation. Streaming datasets (HuggingFace, WebDataset) use IterableDataset.
   - What's unclear: Whether scaling law experiments (which need precise token counts) can work with streaming data.
   - Recommendation: Accept IterableDataset in validation (don't reject it). The training engine (Phase 4) will iterate by token count, not epochs. Document that IterableDataset users must ensure their dataset provides enough data for the planned token budgets.

3. **How does loss_fn interact with the model's existing forward() loss?**
   - What we know: The existing GPT `forward()` optionally computes loss when labels are provided. The library API introduces a separate `loss_fn`.
   - What's unclear: Whether Phase 4's training loop calls `model(input_ids)` then `loss_fn(logits, labels)`, or `model(input_ids, labels)` and ignores the external loss_fn.
   - Recommendation: The library API is the source of truth. Phase 4's training loop should call `model(batch_input)` to get predictions, then `loss_fn(predictions, targets)` to get loss. The model's internal loss computation is for the model's own convenience and is NOT used by the library. This is a Phase 4 design decision but informs Phase 2's validation: the loss_fn must work with model outputs, not model internals.

## Sources

### Primary (HIGH confidence)
- Existing codebase: `src/flops_fit/api.py`, `src/flops_fit/__init__.py`, `src/flops_fit/model.py`, `src/flops_fit/trainer.py` -- direct inspection
- [PyTorch torch.utils.data documentation](https://docs.pytorch.org/docs/stable/data.html) -- Dataset, DataLoader, IterableDataset interfaces
- [PyTorch Datasets & DataLoaders tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) -- standard usage patterns
- Phase 1 research and plan: `.planning/phases/01-library-skeleton-and-model-interface/01-RESEARCH.md`

### Secondary (MEDIUM confidence)
- [PyTorch custom loss functions guide](https://machinelearningmastery.com/creating-custom-layers-loss-functions-pytorch/) -- nn.Module vs function patterns
- [PyTorch Forums: custom loss functions](https://discuss.pytorch.org/t/custom-loss-functions/29387) -- community patterns

### Tertiary (LOW confidence)
- None. This phase uses well-established PyTorch interfaces with extensive documentation.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- uses only PyTorch's built-in data utilities, no new dependencies
- Architecture: HIGH -- Dataset/DataLoader and callable are standard PyTorch patterns
- Pitfalls: HIGH -- based on well-documented PyTorch gotchas (IterableDataset, HuggingFace compat, inspect.signature limits)

**Research date:** 2026-02-16
**Valid until:** Indefinite (torch.utils.data API is stable and backward-compatible)
