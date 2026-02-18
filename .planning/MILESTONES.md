# Milestones

## v1.0 MVP (Shipped: 2026-02-18)

**Phases:** 9 (Phases 1-9) | **Plans:** 19 | **Tests:** 205 passing | **LOC:** ~7,300 Python
**Timeline:** 2026-02-14 → 2026-02-18 (4 days)

**Delivered:** Python library where users pass their model class, dataset, and loss function to get Chinchilla-style scaling law predictions for any architecture.

**Key accomplishments:**
- Model factory with `num_params()` contract — library scales any user-defined model class by varying a size parameter
- Dataset and loss validation wired into `find_optimal()` with fail-fast ordering and clear error messages
- IsoFLOP sweep planning with probe-based grid generation and inspectable `SweepPlan` dataclasses
- Real PyTorch training loop (SGD, device-aware) using Chinchilla FLOPs formula integrated into sweep pipeline
- Linear-space NLS power law fitting with irreducible loss (`l_inf`) term — unbiased vs log-space regression
- `Result` facade with `chinchilla_table()`, `plot()`, `predict()` — full end-to-end `find_optimal()` API
- GPT + TinyStories and ViT + CIFAR-10 examples proving architecture-agnostic, multi-modality design
- HuggingFace Accelerate multi-GPU integration — no user code changes, activated via `accelerate launch`

---

