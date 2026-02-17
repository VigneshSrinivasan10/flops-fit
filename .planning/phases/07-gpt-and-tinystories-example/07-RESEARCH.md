# Phase 7: GPT + TinyStories Example - Research

**Researched:** 2026-02-17
**Domain:** Library examples, CLI wrappers, TinyStories integration, self-contained demonstration
**Confidence:** HIGH

## Summary

Phase 7 converts the existing GPT + TinyStories codebase into reusable library examples. The goal is to demonstrate how users integrate their own models and datasets with `flops_fit.find_optimal()`, not to build new functionality. Three key artifacts are needed:

1. **GPT Model Import:** Refactor the existing GPT implementation into `flops_fit.examples.gpt` so users can `from flops_fit.examples import GPT`
2. **TinyStories Wrapper:** Create a simple PyTorch Dataset wrapper around the HuggingFace TinyStories dataset that works end-to-end with the library
3. **Example Scripts:** Two self-contained Python scripts demonstrating (a) programmatic library usage and (b) CLI wrapper usage
4. **Mock/CPU Mode:** Existing `mode=mock` in trainer already enables testing without GPU; no new work needed

The key insight: Phase 7 is **refactoring + wrapping**, not building. All heavy lifting (find_optimal, Result, analyzer, visualizer, trainer) already exists from Phases 1-6.

**Primary recommendation:**

Create `src/flops_fit/examples/` package with:
- `gpt.py` — GPT model class and config (refactored from existing)
- `tinystories.py` — TinyStories dataset wrapper (new, minimal)
- `example_programmatic.py` — Demonstrates `find_optimal()` usage
- `example_cli_wrapper.py` — Demonstrates command-line integration

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `datasets` | 2.14.0+ | HuggingFace datasets (TinyStories hosted there) | Already in project dependencies; TinyStories official source |
| `transformers` | 4.35.0+ | Tokenizers (GPT-2 for TinyStories) | Already in project dependencies; standard for tokenization |
| `torch` | 2.0.0+ | Model definition and training | Already in project dependencies; core dependency |
| `flops_fit` | 0.1.0 | The library being demonstrated | This is what users import to use the example |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pathlib.Path` | stdlib | File I/O for dataset caching | Already used throughout codebase |
| `json` | stdlib | Configuration serialization | For example config files |
| `argparse` | stdlib | CLI argument parsing in wrapper example | Standard Python CLI tool |
| `hydra` | 1.3.2+ | CLI configuration (used in existing trainer, kept for example only) | Already available; good for showing Hydra wrapper alternative |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `datasets` library (HuggingFace) | Raw text file loading | HuggingFace: handles caching, streaming, multi-format. Raw files: simpler but requires custom batching. Use HF. |
| GPT-2 tokenizer | Custom BPE tokenizer | GPT-2: 50K vocab, proven on English text. Custom: control over encoding but reinvents the wheel. Use GPT-2. |
| Example scripts (Python) | Jupyter notebooks | Scripts: easier to test, version control, integrate into tests. Notebooks: better for exploration. Use scripts. |
| `argparse` for CLI wrapper | `click` library | argparse: stdlib, zero deps. click: more Pythonic but adds dependency. Use argparse. |

**Installation (already in pyproject.toml):**
```bash
pip install datasets transformers torch
# OR
uv sync
```

## Architecture Patterns

### Recommended Project Structure

```
src/flops_fit/
├── examples/                   # NEW: Examples package
│   ├── __init__.py            # Exports GPT, TinyStoriesDataset, example functions
│   ├── gpt.py                 # Refactored GPT + GPTConfig (from src/flops_fit/model.py)
│   ├── tinystories.py         # TinyStories dataset wrapper (NEW)
│   ├── example_programmatic.py # Demonstrates flops_fit.find_optimal() usage (NEW)
│   └── example_cli_wrapper.py  # Demonstrates Hydra/CLI integration (NEW)
│
├── model.py                    # Existing GPT stays here (or reference moved to examples)
├── trainer.py                  # Existing trainer (no changes)
├── api.py                      # Existing find_optimal (no changes)
└── ...                         # All other Phase 1-6 modules
```

### Pattern 1: GPT as Importable Example

**What:** The existing GPT model should be importable from `flops_fit.examples.gpt` so users see it as a reference implementation of the model contract.

**When to use:** When documenting how to write a model class for flops_fit

**Example:**
```python
# Source: flops_fit.examples.gpt
from flops_fit.examples import GPT, GPTConfig

config = GPTConfig(d_model=256, num_layers=6)
model = GPT(config)

# Satisfies the flops_fit contract:
print(model.num_params())  # Returns positive integer
```

**Key insight:** Don't duplicate GPT code. Move the existing implementation to examples, then import it back in tests if needed. Single source of truth.

### Pattern 2: TinyStories Dataset Wrapper

**What:** A minimal PyTorch Dataset that:
1. Downloads TinyStories from HuggingFace if not cached
2. Tokenizes text to token IDs using GPT-2 tokenizer
3. Yields (input_ids, labels) pairs for language modeling

**When to use:** When demonstrating text-based scaling law experiments

**Example:**
```python
# Source: flops_fit.examples.tinystories
from flops_fit.examples import TinyStoriesDataset

dataset = TinyStoriesDataset(
    split="train",
    seq_len=256,
    cache_dir=".cache/datasets",
)

# Returns (input_ids, labels) both shape (256,)
# Satisfies torch.utils.data.Dataset interface
for input_ids, labels in dataset:
    assert input_ids.shape == (256,)
    assert labels.shape == (256,)
```

**Key constraints:**
- Must be a proper `torch.utils.data.Dataset` (implement `__len__` and `__getitem__`)
- Must support batching via DataLoader
- Must cache downloaded data to disk to avoid re-downloading
- Should handle sequence length padding/truncation
- Should work in mock/CPU mode (no GPU requirement)

### Pattern 3: Programmatic Example Script

**What:** A standalone Python script demonstrating the full workflow:
1. Create GPT + TinyStories instances
2. Call `flops_fit.find_optimal()` with compute budgets
3. Display results (chinchilla_table, plots)

**When to use:** When users want to see "how do I use the library?" without CLI/Hydra

**Example:**
```python
# Source: flops_fit.examples.example_programmatic.py
import flops_fit
from flops_fit.examples import GPT, GPTConfig, TinyStoriesDataset

# 1. Define model
config = GPTConfig(d_model=256, num_layers=6)
model_cls = lambda d_model: GPT(GPTConfig(d_model=d_model, num_layers=6))

# 2. Load dataset
dataset = TinyStoriesDataset(split="train", seq_len=256)

# 3. Define loss
def loss_fn(logits, labels):
    return F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))

# 4. Run scaling law experiment
result = flops_fit.find_optimal(
    model_cls=model_cls,
    model_size_param="d_model",
    dataset=dataset,
    loss_fn=loss_fn,
    compute_budgets=[1e17, 1e18, 1e19],
    train=True,
    output_dir="outputs/gpt_tinystories",
)

# 5. Display results
print(result.chinchilla_table())
result.plot(show=True)
```

**Critical detail:** The `model_cls` must be a callable that returns an instance with `num_params()`. Don't pass the class directly; wrap it in a lambda or factory function that accepts `d_model`.

### Pattern 4: CLI Wrapper Example (Using Existing Hydra Setup)

**What:** A Python script showing how to integrate the library with Hydra CLI

**When to use:** When demonstrating "how to expose the library via command line"

**Example:**
```python
# Source: flops_fit.examples.example_cli_wrapper.py
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    import flops_fit
    from flops_fit.examples import GPT, GPTConfig, TinyStoriesDataset

    # Build model from config
    model_config = GPTConfig(**cfg.model)
    model_cls = lambda d_model: GPT(GPTConfig(d_model=d_model, **cfg.model))

    # Build dataset
    dataset = TinyStoriesDataset(**cfg.dataset)

    # Run find_optimal
    result = flops_fit.find_optimal(
        model_cls=model_cls,
        model_size_param="d_model",
        dataset=dataset,
        loss_fn=...,
        compute_budgets=cfg.compute_budgets,
        train=cfg.train,
        output_dir=cfg.output_dir,
    )

    print(result.chinchilla_table())

if __name__ == "__main__":
    main()
```

**Key insight:** This is NOT a new CLI tool. It's showing how users would build their own. The existing `ff-train`, `ff-plan`, etc. stay as-is; this just demonstrates the pattern.

### Anti-Patterns to Avoid

- **Duplicating GPT:** Don't copy model.py code into examples. Move it once, import everywhere.
- **Complex Dataset Logic:** TinyStories wrapper should be < 100 lines. No fancy preprocessing, just tokenize and return.
- **Hardcoding Paths:** Examples should use relative paths or configurable cache_dir.
- **GPU-Only Examples:** Both examples must work with `trainer.mode=mock` (synthetic training) for testing. Real training is optional.
- **Circular Imports:** Don't have examples import from trainer; keep them focused on the library API (find_optimal).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dataset download/caching | Custom download script | HuggingFace `datasets` library | Handles mirrors, checksums, resumable downloads; avoids reinventing package management |
| Tokenization for English text | Custom tokenizer | transformers library (GPT-2 tokenizer) | Proven on billions of tokens; handles edge cases; integrates with models |
| Model instantiation at different sizes | Manual size variations | Library's `model_factory.create_model()` | Already validates contract, handles size param; reuse existing validation |
| Loss computation for language modeling | Custom loss | PyTorch's `F.cross_entropy()` | Standard, efficient, differentiable; no reason to reinvent |
| Sweep execution | Manual loop | Existing `find_optimal()` from Phase 6 | Already orchestrates plan → train → analyze → visualize |

**Key insight:** Phase 7 is "show, don't tell." Examples should import and use library primitives, not rebuild them. Readers learn by seeing how to call the real API, not by watching custom code.

## Common Pitfalls

### Pitfall 1: GPT Import Conflicts

**What goes wrong:** Moving GPT to `examples/gpt.py` breaks existing imports in tests, trainer, and model_factory if not coordinated.

**Why it happens:** Old code imports `from flops_fit.model import GPT`. If GPT is moved, those imports fail silently or raise ImportError at runtime.

**How to avoid:**
1. Create `examples/gpt.py` with the GPT code
2. In `model.py`, add: `from flops_fit.examples.gpt import GPT` (re-export for backward compat)
3. Update `__init__.py` to export from both locations
4. Run full test suite to catch any missed imports

**Warning signs:** `ImportError: cannot import name 'GPT'` in tests, or tests suddenly failing after moving GPT.

### Pitfall 2: TinyStories Download at Import Time

**What goes wrong:** If `TinyStoriesDataset.__init__()` downloads the entire dataset synchronously, importing the class hangs or fails if the network is down.

**Why it happens:** Eager evaluation of HuggingFace `datasets.load_dataset()` blocks the thread.

**How to avoid:**
- Download/cache happens only in `TinyStoriesDataset.prepare_data()` or first `__getitem__()` call
- Log progress clearly: "Downloading TinyStories (X GB)..." so users know it's not hung
- Use `cache_dir` parameter to avoid repeated downloads

**Warning signs:** Import takes 30+ seconds; code hangs on `from flops_fit.examples import TinyStoriesDataset`.

### Pitfall 3: Mismatched Tokenizer Vocabulary

**What goes wrong:** GPT model has vocab_size=50257 (GPT-2), but tokenizer produces IDs outside that range.

**Why it happens:** Using a different tokenizer (e.g., BPE from another model) without matching vocabulary size.

**How to avoid:**
- Always use GPT-2 tokenizer (from transformers) for TinyStories dataset
- Assert: `tokenizer.vocab_size == model.config.vocab_size`
- Test: try tokenizing a sample story, verify max ID < vocab_size

**Warning signs:** `IndexError: index 50257 is out of bounds for dimension with size 50257` during embedding lookup.

### Pitfall 4: Dataset __len__ Mismatch with Actual Data

**What goes wrong:** `TinyStoriesDataset.__len__()` returns cached value, but actual HuggingFace dataset was updated after caching.

**Why it happens:** HuggingFace `datasets` caches locally, but if the remote version changes, __len__ can become incorrect.

**How to avoid:**
- Don't cache __len__; compute it from the actual dataset: `return len(self.dataset)`
- Log warnings if cached size differs from loaded size

**Warning signs:** DataLoader stops early; IndexError at the end of an epoch.

### Pitfall 5: Loss Function Shape Mismatch

**What goes wrong:** Trainer expects loss_fn(logits, labels) but example shows loss_fn(outputs, targets) with wrong shapes.

**Why it happens:** Language modeling loss expects (batch, seq_len, vocab) logits and (batch, seq_len) labels, not 1D.

**How to avoid:**
- Example loss_fn must reshape explicitly: `logits.view(-1, vocab_size)` and `labels.view(-1)`
- Document expected shapes in example code
- Add assertion in example: `assert logits.shape[-1] == config.vocab_size`

**Warning signs:** Loss computation raises `ValueError: view size mismatch` or produces NaN loss.

### Pitfall 6: Mock vs. Real Training Mode Confusion

**What goes wrong:** Example script works with `mode=mock` in trainer config, but users expect it to train real models.

**Why it happens:** `mode=mock` generates synthetic losses instead of running training. Users miss the note and think they have results.

**How to avoid:**
- Document clearly: "mock mode for testing, set trainer.mode=local for real training"
- Example script should explicitly note: `# Running in mock mode. Set train=False to skip training.`
- Show mock output differs from real (synthetic losses all ~2.5 vs. real losses decreasing)

**Warning signs:** Users report "results don't show the scaling law shape I expected."

## Code Examples

Verified patterns from existing codebase:

### GPT Model and Config (Already Exists, Just Reorganize)

```python
# Source: src/flops_fit/model.py (move to examples/gpt.py)
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class GPTConfig:
    """Configuration for GPT model."""
    vocab_size: int = 50257      # GPT-2 tokenizer vocab size
    d_model: int = 256           # Hidden dimension (width) — THIS VARIES IN SWEEP
    num_layers: int = 6          # Number of transformer blocks
    num_heads: int = 4           # Number of attention heads
    d_ff: Optional[int] = None   # FFN hidden dim (default: 4 * d_model)
    max_seq_len: int = 256       # Maximum sequence length
    dropout: float = 0.0         # Dropout rate (0 for scaling experiments)
    parametrization: str = "u-mup" # "u-mup" or "sp"
    base_width: int = 32         # Base width for u-mup LR scaling
    base_width_wd: int = 1024    # Base width for u-mup WD scaling
    base_std: float = 0.02       # Base std for initialization

class GPT(nn.Module):
    """GPT-style language model with u-mup parameterization."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # ... transformer blocks ...

    def num_params(self) -> int:
        """Return total number of parameters (flops_fit contract)."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """Forward pass."""
        # ... transformer logic ...
        # Returns (logits, loss) if labels provided
```

**Key detail:** `num_params()` method is non-negotiable. This is how the library validates the model contract.

### TinyStories Dataset Wrapper (NEW)

```python
# Source: examples/tinystories.py (NEW)
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path

class TinyStoriesDataset(Dataset):
    """PyTorch Dataset wrapper for HuggingFace TinyStories."""

    def __init__(
        self,
        split: str = "train",
        seq_len: int = 256,
        cache_dir: str = ".cache/datasets",
        tokenizer_name: str = "gpt2",
    ):
        """
        Args:
            split: "train" or "validation"
            seq_len: Sequence length for language modeling
            cache_dir: HuggingFace cache directory
            tokenizer_name: Name of HuggingFace tokenizer
        """
        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset (caches to cache_dir automatically)
        self.dataset = load_dataset(
            "roneneldan/TinyStories",
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        """Return (input_ids, labels) for language modeling."""
        story = self.dataset[idx]["story"]

        # Tokenize
        encoded = self.tokenizer(
            story,
            max_length=self.seq_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)

        # For language modeling, labels = input_ids (shifted by trainer)
        labels = input_ids.clone()

        return input_ids, labels
```

**Critical details:**
- Tokenizer cached automatically by HuggingFace
- Dataset cached to `cache_dir` to avoid re-downloading
- Returns tensors (not lists) for DataLoader compatibility
- Returns (input_ids, labels) tuples (standard for supervised learning)

### Programmatic Example (NEW)

```python
# Source: examples/example_programmatic.py (NEW)
#!/usr/bin/env python3
"""
Example: Use flops_fit.find_optimal() to run a scaling law experiment on GPT + TinyStories.

This demonstrates the library API directly (without CLI/Hydra).

To run:
    python -m flops_fit.examples.example_programmatic
"""

import torch
import torch.nn.functional as F
from pathlib import Path

import flops_fit
from flops_fit.examples import GPT, GPTConfig, TinyStoriesDataset


def loss_fn(outputs, labels):
    """Language modeling loss."""
    logits, _ = outputs
    return F.cross_entropy(logits.view(-1, 50257), labels.view(-1))


def main():
    # 1. Create model class (must accept d_model parameter)
    def create_model(d_model: int) -> GPT:
        config = GPTConfig(
            d_model=d_model,
            num_layers=6,
            num_heads=4,
            vocab_size=50257,
        )
        return GPT(config)

    # 2. Load dataset
    print("Loading TinyStories dataset...")
    dataset = TinyStoriesDataset(
        split="train",
        seq_len=256,
        cache_dir=".cache/datasets",
    )

    # 3. Run scaling law experiment
    print("Running scaling law experiment...")
    result = flops_fit.find_optimal(
        model_cls=create_model,
        model_size_param="d_model",
        dataset=dataset,
        loss_fn=loss_fn,
        compute_budgets=[1e17, 1e18, 1e19],  # Small budgets for demo
        train=True,
        output_dir="outputs/gpt_tinystories",
    )

    # 4. Display results
    print("\n" + "="*80)
    print("SCALING LAW RESULTS")
    print("="*80)
    print(result.chinchilla_table())

    print("\nGenerating plots...")
    figs = result.plot(show=False)
    print(f"Saved {len(figs)} figures to outputs/gpt_tinystories/plots/")


if __name__ == "__main__":
    main()
```

**Key lesson:** This is what users copy and modify. Keep it simple and clear.

### CLI Wrapper Example (NEW)

```python
# Source: examples/example_cli_wrapper.py (NEW)
#!/usr/bin/env python3
"""
Example: Use flops_fit with Hydra configuration.

This shows how to expose the library via command-line arguments
(as an alternative to the programmatic API).

To run:
    python -m flops_fit.examples.example_cli_wrapper \
        model.d_model=256 \
        dataset.seq_len=256 \
        compute_budgets=[1e17,1e18,1e19]
"""

import hydra
from omegaconf import DictConfig
import torch
import torch.nn.functional as F

import flops_fit
from flops_fit.examples import GPT, GPTConfig, TinyStoriesDataset


def loss_fn(outputs, labels):
    logits, _ = outputs
    return F.cross_entropy(logits.view(-1, 50257), labels.view(-1))


@hydra.main(version_base=None, config_path=None)
def main(cfg: DictConfig):
    # Build model from config
    def create_model(d_model: int) -> GPT:
        config = GPTConfig(
            d_model=d_model,
            num_layers=cfg.model.get("num_layers", 6),
            num_heads=cfg.model.get("num_heads", 4),
        )
        return GPT(config)

    # Load dataset
    dataset = TinyStoriesDataset(
        split=cfg.dataset.get("split", "train"),
        seq_len=cfg.dataset.get("seq_len", 256),
    )

    # Run scaling law experiment
    result = flops_fit.find_optimal(
        model_cls=create_model,
        model_size_param="d_model",
        dataset=dataset,
        loss_fn=loss_fn,
        compute_budgets=cfg.get("compute_budgets", [1e17, 1e18, 1e19]),
        train=cfg.get("train", True),
        output_dir=cfg.get("output_dir", "outputs/gpt_tinystories"),
    )

    print(result.chinchilla_table())


if __name__ == "__main__":
    main()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hardcoded GPT in trainer.py | GPT as importable example | Phase 7 (this phase) | Users can import GPT independently; library decoupled from specific model |
| Hydra config as primary API | Library API as primary | Phase 1 (library pivot) | Users can use flops_fit without CLI/Hydra; Hydra becomes optional example |
| Manual dataset loading | HuggingFace datasets integration | Phase 2 (dataset interface) | Automatic caching, streaming support; TinyStories just a wrapper |
| Hardcoded TinyStories path | Configurable dataset parameter | Phase 2 | Users can swap datasets (CIFAR, etc.) without modifying code |

**Deprecated/outdated:**
- **Direct Hydra usage as library primary interface:** Moved to examples only. Library is now pure Python API.
- **Custom tokenization:** Use transformers library tokenizers. Simpler, more reliable.

## Open Questions

1. **Should examples/ be installable as sub-package?**
   - What we know: examples/ is alongside api.py, analyzer.py, etc. in src/flops_fit/
   - What's unclear: Do we add `flops_fit.examples` to `__all__` in `__init__.py`? Or leave it discoverable but not exported?
   - Recommendation: Add to `__all__` with marker comment: `# Examples (reference implementations, not core library)`

2. **Where should example config files live?**
   - What we know: Trainer uses conf/trainer.yaml, conf/presets/cpu_fast.yaml
   - What's unclear: Should CLI wrapper example have its own conf/examples/ directory?
   - Recommendation: If using Hydra wrapper, create `src/flops_fit/examples/conf/` with minimal config files. But keep it light — the programmatic example is the primary one.

3. **How to handle backward compatibility with existing imports of GPT from model.py?**
   - What we know: Existing code does `from flops_fit.model import GPT`
   - What's unclear: Do we deprecate model.py's GPT, or keep a re-export?
   - Recommendation: Keep a re-export in model.py: `from flops_fit.examples.gpt import GPT` for backward compat, but document that examples/ is the canonical location.

4. **Should TinyStories dataset work without network access?**
   - What we know: HuggingFace datasets requires download on first run
   - What's unclear: What if user has no internet? Should we provide offline version or mock?
   - Recommendation: For tests, add a fixture that mocks HuggingFace; for real usage, document internet requirement. Phase 8+ can add offline dataset variants.

5. **What compute budgets should the example use?**
   - What we know: Trainer.yaml recommends 1e12 to 1e14 FLOPs for CPU experiments
   - What's unclear: Should example scripts use realistic 1e17-1e22 (for documentation) or 1e12-1e14 (for testing)?
   - Recommendation: Programmatic example uses small 1e12-1e13 budgets and sets `trainer.mode=mock` by default so it runs fast. CLI wrapper example shows realistic 1e17+ budgets as documentation.

## Verification Checklist

- [x] GPT model exists in codebase (src/flops_fit/model.py) — ready to move
- [x] GPT has `num_params()` method — satisfies library contract
- [x] TinyStories dataset on HuggingFace (roneneldan/TinyStories) — accessible via datasets library
- [x] Trainer supports mode=mock for testing without GPU — no new code needed
- [x] find_optimal() API is complete (Phase 6) — examples can use it directly
- [x] Result object has chinchilla_table(), plot(), predict() — all documented in Phase 6
- [x] Existing trainer.yaml has TinyStories config — shows the pattern

## Sources

### Primary (HIGH confidence)
- `src/flops_fit/model.py` — Existing GPT implementation with u-mup parameterization, num_params() method
- `src/flops_fit/api.py` — Complete find_optimal() implementation from Phase 6
- `src/flops_fit/result.py` — Result object with chinchilla_table(), plot(), predict() methods
- `src/flops_fit/trainer.py` — TrainingRunner with mode=mock support for testing
- `src/flops_fit/conf/trainer.yaml` — Existing TinyStories configuration in project
- `tests/test_api.py` — Integration tests showing find_optimal() usage patterns
- `tests/test_result.py` — Tests demonstrating Result object API
- [HuggingFace Datasets Library](https://huggingface.co/datasets) — roneneldan/TinyStories hosted here
- [Transformers Library](https://huggingface.co/transformers/) — GPT-2 tokenizer source

### Secondary (MEDIUM confidence)
- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) — Original paper describing dataset and use case
- [roneneldan/TinyStories on HuggingFace](https://huggingface.co/datasets/roneneldan/TinyStories) — Dataset card with download stats, versions, usage examples
- `.planning/REQUIREMENTS.md` — Project requirements linking EX-01 (GPT + TinyStories) and EX-03 (CLI wrapper) to Phase 7
- `.planning/PROJECT.md` — Project context confirming "existing CLI/Hydra becomes example, not core"

### Tertiary (LOW confidence)
- None; all critical information verified with high-confidence sources

## Metadata

**Confidence breakdown:**
- **Standard stack:** HIGH — All libraries already in project dependencies, verified in pyproject.toml
- **Architecture:** HIGH — Existing GPT, trainer, find_optimal all verified in codebase; HuggingFace APIs are well-documented
- **Integration:** HIGH — Trainer already has mode=mock; find_optimal already complete; no unexpected dependencies
- **Dataset:** MEDIUM — TinyStories is live on HuggingFace, but dataset versions may change; verified as of Feb 2025 in HuggingFace dataset card

**Research date:** 2026-02-17
**Valid until:** 2026-03-17 (30 days; TinyStories dataset is stable, but library code evolves)

**Key assumptions verified:**
- GPT model has `num_params()` method ✓
- find_optimal() is complete and tested ✓
- Result object has required methods (chinchilla_table, plot, predict) ✓
- TinyStories dataset is publicly available on HuggingFace ✓
- Trainer supports mode=mock (no GPU training required) ✓
