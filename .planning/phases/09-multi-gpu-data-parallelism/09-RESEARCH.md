# Phase 9: Multi-GPU Data Parallelism - Research

**Researched:** 2026-02-18
**Domain:** Distributed training / HuggingFace Accelerate / PyTorch DDP
**Confidence:** HIGH

## Summary

Phase 9 adds multi-GPU data parallelism to the flops-fit library using HuggingFace Accelerate. The core challenge is integrating distributed training into a **library-first** design where users call `find_optimal()` from their own Python code -- they do not write training scripts themselves. This constrains the integration pattern significantly: the library must handle Accelerate setup internally, and the user's model class and dataset must work without modification.

HuggingFace Accelerate (v1.12.0, latest stable) wraps PyTorch's DistributedDataParallel (DDP) with a minimal API: create an `Accelerator`, call `prepare()` on model/optimizer/dataloader, replace `loss.backward()` with `accelerator.backward(loss)`, and launch via `accelerate launch`. The key insight is that **multi-GPU requires process spawning** -- you cannot simply call `Accelerator()` in a single process and get multi-GPU. The user must either: (a) launch their script with `accelerate launch`, or (b) the library uses `notebook_launcher` to spawn processes programmatically.

For flops-fit's library-first design, the recommended approach is a **two-tier strategy**: (1) the library's `_local_train` method uses Accelerate internally so the training loop itself is Accelerate-aware and works correctly in both single-GPU and multi-GPU contexts, and (2) the user is instructed to launch their script with `accelerate launch` when they want multi-GPU. This matches HuggingFace's own design philosophy and avoids fighting the distributed training paradigm.

**Primary recommendation:** Modify `TrainingRunner._local_train()` to use Accelerate's `Accelerator` for device placement, data sharding, and gradient sync. Add `accelerate>=1.0.0` as a dependency. Single-GPU remains the default (works with `python script.py`); multi-GPU activates when the user runs `accelerate launch script.py`.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| accelerate | >=1.0.0 | Multi-GPU data parallelism | Official HuggingFace library; wraps PyTorch DDP with minimal API; 5 lines to integrate; handles device placement, data sharding, gradient sync |
| torch | >=2.0.0 | Already a dependency | DDP backend; NCCL for GPU-GPU communication |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| accelerate.utils.set_seed | (part of accelerate) | Reproducible seeding across processes | Always, when reproducibility matters |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Accelerate | Raw PyTorch DDP | More boilerplate (init_process_group, DistributedSampler, cleanup); Accelerate wraps all of this |
| Accelerate | PyTorch FSDP | Overkill for data parallelism; FSDP shards model weights across GPUs, needed only when model does not fit in one GPU |
| Accelerate | DeepSpeed | Much heavier dependency; designed for very large models; Accelerate can use DeepSpeed as backend if needed later |

**Installation:**
```bash
pip install "accelerate>=1.0.0"
```

Or add to pyproject.toml dependencies:
```toml
"accelerate>=1.0.0",
```

## Architecture Patterns

### How Accelerate Integrates Into _local_train

The key change is in `TrainingRunner._local_train()`. Currently it manually does `.to(device)`, creates an optimizer, and runs a standard training loop. With Accelerate:

```python
from accelerate import Accelerator

def _local_train(self, experiment, model_cls, size_param, model_kwargs,
                 dataset_or_loader, loss_fn, epochs=1, batch_size=32,
                 learning_rate=0.01):
    accelerator = Accelerator()

    model = create_model(model_cls, size_param, experiment.size_param_value, model_kwargs)
    dataloader = wrap_dataset(dataset_or_loader, batch_size=batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Accelerate handles: device placement, DDP wrapping, dataloader sharding
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    start_time = time.time()
    model.train()
    total_loss = 0.0
    total_batches = 0

    for _epoch in range(epochs):
        for _batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:
                inputs = batch
                targets = batch
            # No .to(device) needed -- prepare() handled it
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            accelerator.backward(loss)  # replaces loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    wall_time = time.time() - start_time
    final_loss = total_loss / total_batches if total_batches > 0 else float("nan")

    # Get underlying model for num_params (prepare wraps in DDP)
    unwrapped_model = accelerator.unwrap_model(model)
    actual_n = unwrapped_model.num_params()
    actual_flops = 6 * actual_n * experiment.num_tokens

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(final_loss), float(actual_flops), float(wall_time)
```

### Single-GPU vs Multi-GPU: Zero Code Difference

When run with `python script.py`, Accelerate detects a single process and acts as a thin no-op wrapper -- `prepare()` just moves to GPU, `backward()` calls `loss.backward()`. No overhead.

When run with `accelerate launch --num_processes=N script.py`, Accelerate detects DDP environment variables set by the launcher and activates data parallelism automatically.

This means **the same code path works for both cases**.

### Launch Pattern for Users

```python
# user_script.py
from flops_fit import find_optimal
from my_models import GPT
from my_data import TinyStoriesDataset
import torch.nn.functional as F

result = find_optimal(
    model_cls=GPT,
    model_size_param="d_model",
    dataset=TinyStoriesDataset(),
    loss_fn=F.cross_entropy,
    compute_budgets=[1e15, 1e16, 1e17],
)
```

Single GPU: `python user_script.py`
Multi-GPU: `accelerate launch --num_processes=4 user_script.py`

### Process Architecture for Sweep

**Important design consideration:** In a sweep, each experiment trains a different model. Data parallelism means sharding batches of the *same* experiment across GPUs. The sweep loop itself should run on all processes identically (Accelerate handles the data sharding per experiment).

```
Process 0 (main)     Process 1           Process 2           Process 3
  |                    |                   |                   |
  +-- exp_0000 --------+--- exp_0000 ------+--- exp_0000 -----+--- exp_0000
  |   (1/4 batches)    |   (1/4 batches)   |   (1/4 batches)  |   (1/4 batches)
  |   gradient sync    |   gradient sync   |   gradient sync  |   gradient sync
  |                    |                   |                   |
  +-- exp_0001 --------+--- exp_0001 ------+--- exp_0001 -----+--- exp_0001
  ...                  ...                 ...                 ...
```

### Where Results Should Be Written

Only the main process (rank 0) should write results.json. Accelerate provides `accelerator.is_main_process` for this:

```python
if accelerator.is_main_process:
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
```

### Logging/Progress Bars

Only the main process should print progress. Use `accelerator.is_main_process` or `accelerator.is_local_main_process` to gate tqdm and logging.

### Recommended Project Structure Changes

```
src/flops_fit/
    trainer.py          # Modified: _local_train uses Accelerator
    api.py              # Minimal change: possibly pass num_processes hint
    data.py             # No change needed
    sweep.py            # No change needed
    model_factory.py    # No change needed
```

The changes are concentrated in `trainer.py`. The rest of the library is unaffected.

### Anti-Patterns to Avoid
- **Creating Accelerator at module level:** Accelerator must be created inside the training function, not at import time. Module-level creation breaks process spawning.
- **Manual .to(device) after prepare():** Accelerate handles device placement. Adding manual `.to(device)` after `prepare()` can cause device mismatches in DDP.
- **Calling model.num_params() on wrapped model:** After `prepare()`, model is wrapped in DDP. Use `accelerator.unwrap_model(model)` to access the original model's methods.
- **Writing files from all processes:** Only rank 0 should write results.json. Other ranks writing simultaneously corrupts files.
- **Different control flow per process:** All processes must execute the same forward/backward calls. Branching logic that only some processes execute causes deadlocks.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Data sharding across GPUs | Custom DistributedSampler management | `accelerator.prepare(dataloader)` | Accelerate handles DistributedSampler, padding for uneven batches, drop_last coordination |
| Gradient synchronization | Manual all_reduce calls | `accelerator.backward(loss)` | DDP sync is automatic; manual reduction is error-prone |
| Process group initialization | `torch.distributed.init_process_group()` boilerplate | `Accelerator()` constructor | Accelerate reads env vars from launcher, handles NCCL backend selection |
| Device placement | Manual `.to(device)` per tensor | `accelerator.prepare()` | Handles GPU assignment per rank automatically |
| Multi-process file I/O | Lock files or rank-based if statements | `accelerator.is_main_process` | Clean API, no race conditions |
| Seed synchronization | Manual seed setting on each device | `accelerate.utils.set_seed(42)` | Sets random, numpy, torch, cuda, and XLA seeds consistently |

**Key insight:** Accelerate exists precisely to eliminate the boilerplate of distributed training. Every piece of DDP ceremony (process groups, samplers, gradient sync, device assignment) is handled by the 5-line integration pattern.

## Common Pitfalls

### Pitfall 1: Batch Size Semantics Change
**What goes wrong:** User sets `batch_size=32`, expects 32 samples per step globally, but with 4 GPUs gets 128 samples per step (32 per GPU x 4).
**Why it happens:** Accelerate's `prepare(dataloader)` treats the dataloader's batch_size as per-GPU batch size.
**How to avoid:** Document clearly that `batch_size` is per-GPU. Optionally, divide by `accelerator.num_processes` to keep effective batch size constant. For scaling law experiments, the per-GPU interpretation is actually fine since we care about total tokens processed, not batch size dynamics.
**Warning signs:** Loss curves differ between single-GPU and multi-GPU runs.

### Pitfall 2: Loss Values Differ Between Single and Multi-GPU
**What goes wrong:** Users expect identical loss values but get different numbers.
**Why it happens:** (1) Effective batch size changes (see above), (2) DDP averages gradients across processes but loss.item() reports the local process's loss, (3) floating point non-associativity.
**How to avoid:** Accept numerical tolerance (1e-3 to 1e-2 range). For the scaling law use case, small loss differences are acceptable -- we are fitting curves, not chasing exact loss values. To get closer values, use `accelerator.gather()` to average loss across processes for reporting.
**Warning signs:** Multi-GPU loss is consistently offset from single-GPU loss.

### Pitfall 3: Model Methods Unavailable After prepare()
**What goes wrong:** `model.num_params()` fails or returns wrong value after `accelerator.prepare(model)`.
**Why it happens:** `prepare()` wraps the model in `DistributedDataParallel`, which proxies `__getattr__` but may not expose custom methods correctly.
**How to avoid:** Always use `accelerator.unwrap_model(model)` to access custom methods like `num_params()`.
**Warning signs:** `AttributeError` on custom model methods.

### Pitfall 4: Deadlocks from Uneven Control Flow
**What goes wrong:** Training hangs indefinitely.
**Why it happens:** DDP requires all processes to execute the same collective operations. If one process skips a forward/backward pass (e.g., due to fewer batches), it deadlocks waiting for gradient sync.
**How to avoid:** Use `drop_last=True` in DataLoader (already done in `wrap_dataset`). Ensure all processes run the same experiment sequence. Accelerate's prepared DataLoader handles padding/dropping.
**Warning signs:** Process hangs with no error message.

### Pitfall 5: File I/O Corruption
**What goes wrong:** results.json is garbled or contains duplicate entries.
**Why it happens:** Multiple processes write to the same file simultaneously.
**How to avoid:** Gate all file writes with `accelerator.is_main_process`. Use `accelerator.wait_for_everyone()` before file operations that other processes depend on.
**Warning signs:** JSON parse errors, duplicate experiment_ids in results.

### Pitfall 6: Memory Not Freed Between Experiments
**What goes wrong:** OOM errors on later experiments in a sweep.
**Why it happens:** DDP-wrapped models hold references. The current `del model; torch.cuda.empty_cache()` may not fully clean up DDP state.
**How to avoid:** Call `accelerator.free_memory()` between experiments, or create a fresh `Accelerator` per experiment. Also ensure `del model` targets the prepared (wrapped) model.
**Warning signs:** GPU memory grows monotonically during sweep.

### Pitfall 7: Creating Accelerator Per Experiment vs Once
**What goes wrong:** Either stale DDP state (if reused) or slow startup (if recreated).
**Why it happens:** DDP initialization has overhead; but reusing across different model sizes may leave stale gradient buckets.
**How to avoid:** Create one `Accelerator` instance per `run_sweep_from_plan` call but call `accelerator.free_memory()` between experiments and re-`prepare()` each new model/optimizer/dataloader. The Accelerator itself can be reused; it is the prepared objects that must be refreshed.
**Warning signs:** Gradient shape mismatches, slow sweeps.

## Code Examples

### Minimal Integration (Verified Pattern from Official Docs)
```python
# Source: https://huggingface.co/docs/accelerate/quicktour
from accelerate import Accelerator

accelerator = Accelerator()

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    inputs, targets = batch
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

### Gathering Loss Across Processes for Reporting
```python
# Source: https://huggingface.co/docs/accelerate/concept_guides/performance
import torch

# After computing local loss
loss_tensor = torch.tensor([loss.item()], device=accelerator.device)
gathered_loss = accelerator.gather(loss_tensor)
avg_loss = gathered_loss.mean().item()

# Only main process logs
if accelerator.is_main_process:
    logger.info(f"Average loss across GPUs: {avg_loss}")
```

### Unwrapping Model for Custom Methods
```python
# After prepare(), model is DDP-wrapped
unwrapped = accelerator.unwrap_model(model)
n_params = unwrapped.num_params()  # Access custom method
```

### Seed Setting for Reproducibility
```python
# Source: https://huggingface.co/docs/accelerate/concept_guides/performance
from accelerate.utils import set_seed
set_seed(42)  # Sets random, numpy, torch, cuda seeds on all processes
```

### Conditional File I/O
```python
if accelerator.is_main_process:
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
accelerator.wait_for_everyone()  # Ensure file is written before others read
```

### Cleanup Between Experiments
```python
# Between experiments in a sweep:
accelerator.free_memory()  # Releases prepared objects
del model, optimizer, dataloader
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.nn.DataParallel` | `torch.nn.parallel.DistributedDataParallel` via Accelerate | PyTorch 1.x -> 2.x era | DataParallel is deprecated in practice; DDP is faster (no GIL) |
| Manual `init_process_group` + `DistributedSampler` | `Accelerator().prepare()` | Accelerate 0.x -> 1.0 (2024) | 5 lines vs 30+ lines of boilerplate |
| `accelerate config` interactive wizard | Still the recommended way, but CLI flags work too | Stable since 0.12 | Users can skip config and use `--num_processes=N` |

**Deprecated/outdated:**
- `torch.nn.DataParallel`: Single-process, multi-thread. GIL bottleneck. Do not use.
- Manual DDP setup: Still works but Accelerate is strictly better for this use case.

## Open Questions

1. **Should Accelerator be created once per sweep or once per experiment?**
   - What we know: Accelerator can be reused, but prepared objects (model, optimizer, dataloader) must be refreshed per experiment since model architecture changes.
   - What's unclear: Whether calling `prepare()` multiple times on different models with the same Accelerator causes issues with DDP gradient bucket caching.
   - Recommendation: Create Accelerator once at sweep level, call `free_memory()` + re-`prepare()` per experiment. Test this pattern explicitly.

2. **Should the library auto-scale learning rate for multi-GPU?**
   - What we know: HuggingFace recommends linear LR scaling with num_processes. Current default is SGD with lr=0.01.
   - What's unclear: For scaling law experiments specifically, whether LR scaling matters (we're measuring loss at convergence, not training speed).
   - Recommendation: Do NOT auto-scale LR. Document the tradeoff. Users doing serious scaling experiments will tune LR separately.

3. **How to test multi-GPU without multi-GPU hardware?**
   - What we know: Accelerate on single GPU is a no-op wrapper. `accelerate.debug_launcher` can simulate multi-process on CPU.
   - What's unclear: Whether debug_launcher faithfully reproduces DDP behavior.
   - Recommendation: Test the Accelerate integration on single GPU (verifying the code path works). Add a `@pytest.mark.skipif(torch.cuda.device_count() < 2)` marker for true multi-GPU tests that run only in CI or on multi-GPU machines.

4. **Should find_optimal() accept a `num_gpus` parameter?**
   - What we know: Accelerate auto-detects from environment. Adding a parameter would fight this.
   - Recommendation: No. Let Accelerate auto-detect. The user controls GPU count via `accelerate launch --num_processes=N` or `CUDA_VISIBLE_DEVICES`. This is the standard pattern.

## Sources

### Primary (HIGH confidence)
- [HuggingFace Accelerate Quicktour](https://huggingface.co/docs/accelerate/quicktour) - Core integration pattern, prepare/backward API
- [Accelerate Performance Guide](https://huggingface.co/docs/accelerate/concept_guides/performance) - Batch size semantics, LR scaling, reproducibility
- [Accelerate Launchers API](https://huggingface.co/docs/accelerate/en/package_reference/launchers) - notebook_launcher, debug_launcher signatures
- [Accelerate Launch Tutorial](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch) - CLI launch patterns, config files
- [PyPI accelerate](https://pypi.org/project/accelerate/) - Version 1.12.0, Python >=3.10.0

### Secondary (MEDIUM confidence)
- [PyTorch DDP to Accelerate Blog](https://huggingface.co/blog/pytorch-ddp-accelerate-transformers) - Full conversion example, verified against quicktour
- [Accelerate GitHub](https://github.com/huggingface/accelerate) - Repository, issues, development version 1.13.0.dev0

### Tertiary (LOW confidence)
- Community forum discussions on loss differences between single/multi-GPU (anecdotal, consistent with official docs)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Accelerate is the canonical choice for this; well-documented, stable API since 1.0
- Architecture: HIGH - The integration pattern is well-established and minimal; fits library-first design
- Pitfalls: HIGH - Batch size semantics, file I/O, unwrap_model are all documented in official guides
- Reproducibility: MEDIUM - Exact loss matching is documented as not guaranteed; tolerance level is empirical

**Research date:** 2026-02-18
**Valid until:** 2026-04-18 (stable library, slow-moving API)
