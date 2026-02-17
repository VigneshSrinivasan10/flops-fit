# Phase 8: ViT + CIFAR Example - Research

**Researched:** 2026-02-17
**Domain:** Vision Transformer implementation + image classification scaling laws
**Confidence:** HIGH

## Summary

Phase 8 requires implementing a Vision Transformer (ViT) model that proves flops_fit is truly architecture-agnostic. Unlike Phase 7 (text-only GPT), this phase introduces a different modality (images) and a different pattern of output reshaping for loss computation.

The core challenge is not ViT itself—torchvision provides a standard implementation—but rather:
1. Creating a ViT wrapper class that matches the flops_fit model contract (size parameter + `num_params()` method)
2. Implementing an image-specific loss function pattern (image classification returns logits directly, not logits + auxiliary values)
3. Choosing a dataset that's small enough for mock/CPU testing but representative of real image classification
4. Handling the architectural differences from GPT (patch projection instead of embedding, no sequence shifting needed)

**Primary recommendation:** Implement a custom ViT class (patch size 4 or 8, scalable embedding dim) with CIFAR-10 as the dataset. Use torchvision's pretrained ViT as reference but build a trainable variant. Pattern loss function directly (no tuple wrapping) to differentiate from GPT's two-element return.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.0+ | Deep learning framework | Industry standard, already in project |
| torchvision | 0.15+ | Vision models and datasets | Official PyTorch vision library, provides pretrained ViT reference |
| CIFAR-10 | Built-in | Image classification dataset | 32×32 images, 10 classes, 60k training samples—perfect for scaling experiments (small, reproducible) |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| timm (PyTorch Image Models) | 0.9+ | Alternative ViT implementations | For reference; ecosystem standard but not required for Phase 8 |
| Hugging Face Transformers | 4.0+ | Pretrained ViT models | For architecture reference only; use torchvision for lightweight implementation |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| torchvision ViT | timm or HF Transformers | timm/HF offer more variants but add dependencies; torchvision is lighter and sufficient |
| CIFAR-10 | ImageNet-1k | ImageNet requires more storage/bandwidth; CIFAR-10 is practical for CPU testing |
| Custom ViT | torchvision pretrained models | Pretrained weights won't initialize correctly for arbitrary embedding dims; custom is necessary |

**Installation:**
```bash
# Already in project via requirements or poetry
# torchvision.datasets.CIFAR10 requires torchvision
pip install torch torchvision
```

## Architecture Patterns

### Recommended Project Structure

```
src/flops_fit/examples/
├── vit.py                    # ViT model class (patch projection, transformer, classification head)
├── cifar.py                  # CIFAR10 dataset wrapper (optional, torchvision built-in suffices)
├── example_vit_cifar.py      # Programmatic example (parallel to example_programmatic.py)
└── (existing gpt.py, tinystories.py, example_*.py)

tests/
└── test_examples.py          # Add ViT contract tests alongside GPT tests
```

### Pattern 1: ViT Model Contract

**What:** Custom ViT implementation that accepts an embedding dimension (`embed_dim`, `d_model`, or equivalent) as a constructor parameter and exposes `num_params() -> int`.

**When to use:** All ViT scaling experiments. The size parameter controls model width (embedding dimension), which scales the patch projection, transformer blocks, and attention layers proportionally.

**Example:**
```python
# From torchvision.models.vision_transformer source as reference
# but adapted for scalable embedding dimension

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 10,
        embed_dim: int = 256,          # Size parameter (varies in sweep)
        num_layers: int = 12,
        num_heads: int = 8,
        mlp_dim: int = None,           # Default: 4 * embed_dim
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        if mlp_dim is None:
            mlp_dim = 4 * embed_dim

        # Patch embedding: projects patches to embed_dim
        self.patch_embed = nn.Linear(3 * patch_size * patch_size, embed_dim)

        # Classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Classification head: takes [CLS] token output
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 3, 32, 32)
        B = x.shape[0]

        # Extract and flatten patches
        # Reshape: (B, 3, H, W) -> (B, num_patches, patch_dim)
        patches = x.reshape(
            B,
            3,
            self.image_size // self.patch_size,
            self.patch_size,
            self.image_size // self.patch_size,
            self.patch_size,
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
            B, self.num_patches, 3 * self.patch_size * self.patch_size
        )

        # Project patches to embedding
        x = self.patch_embed(patches)  # (B, num_patches, embed_dim)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)  # (B, num_patches+1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed

        # Transformer encoder
        x = self.transformer(x)  # (B, num_patches+1, embed_dim)

        # Classification: use [CLS] token only
        cls_output = x[:, 0]  # (B, embed_dim)
        logits = self.head(cls_output)  # (B, num_classes)

        return logits

    def num_params(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())
```

### Pattern 2: Image Classification Loss Function

**What:** Loss function that directly accepts (logits, labels) without tuple unpacking, unlike GPT's (logits, auxiliary) -> logits pattern.

**When to use:** All image classification tasks where the model returns logits directly.

**Example:**
```python
def vit_loss_fn(logits, labels):
    """Image classification loss for ViT outputs.

    Args:
        logits: Model output, shape (B, num_classes)
        labels: Target class indices, shape (B,)

    Returns:
        Scalar cross-entropy loss.
    """
    return F.cross_entropy(logits, labels)
```

**Key difference from GPT:** GPT returns `(logits, loss_or_None)` tuple and requires unpacking. ViT returns logits directly. This is cleaner and more standard for image classification.

### Pattern 3: Model Factory for Sweep

**What:** Callable that creates ViT instances at different embedding dimensions, fixing other architecture parameters.

**When to use:** Integrating ViT with flops_fit's sweeping mechanism.

**Example:**
```python
def make_vit_factory(num_layers=12, num_heads=8, patch_size=4):
    """Return factory function that creates ViT instances for a given embed_dim."""
    def create_vit(embed_dim: int) -> VisionTransformer:
        return VisionTransformer(
            image_size=32,
            patch_size=patch_size,
            num_classes=10,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
    return create_vit
```

### Pattern 4: CIFAR-10 Dataset Integration

**What:** Use torchvision's built-in CIFAR-10 dataset with image normalization transforms.

**When to use:** Direct data loading without custom wrappers (unlike TinyStories, which required HF lazy loading).

**Example:**
```python
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize

transforms = Compose([
    ToTensor(),
    Normalize(
        mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 channel means
        std=[0.2470, 0.2435, 0.2616],   # CIFAR-10 channel stds
    )
])

dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transforms,
)
```

### Anti-Patterns to Avoid

- **Assuming ViT scales the same way as GPT:** ViT's attention is over patches, not tokens. Sequence length = (image_size / patch_size)². For CIFAR-10 (32×32), patch size 4 gives 64 tokens; patch size 8 gives 16. This is drastically different from GPT's variable sequence length.
- **Not scaling embedding dim uniformly:** If embed_dim changes but num_heads stays fixed, ensure embed_dim is divisible by num_heads, or dynamically adjust heads.
- **Forgetting positional embeddings are learnable and dimension-dependent:** Can't reuse positional embeddings across different embed_dim values—must recreate.
- **Using image-level batch normalization in transformer blocks:** ViT typically uses layer norm, not batch norm. Transformer blocks expect layer norm.
- **Not normalizing CIFAR-10 inputs:** Raw pixel values [0, 255] will destabilize training. Always apply standard mean/std normalization.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Vision Transformer architecture | Custom attention, patching, position embeddings from scratch | torchvision.models.vision_transformer or timm | Subtle pitfalls: positional encoding shape, patch flattening order, causal vs. non-causal attention, output layer initialization |
| CIFAR-10 data loading | Manual image normalization, train/test split logic | torchvision.datasets.CIFAR10 | Built-in, correct mean/std, handles download and caching automatically |
| Cross-entropy loss | Custom softmax + log + NLLLoss | torch.nn.CrossEntropyLoss or F.cross_entropy | Numerically stable implementation critical for convergence |
| Image augmentation (future) | Manual cropping, flipping transforms | torchvision.transforms.Compose | Composable, efficient, standard in ecosystem |

**Key insight:** Vision Transformers are deceptively complex. Positional embeddings, patch projection order, and attention mask construction are easy to get wrong. Using reference implementations and torch.nn utilities is essential.

## Common Pitfalls

### Pitfall 1: Patch Size Incompatibility with CIFAR-10

**What goes wrong:** CIFAR-10 images are 32×32. If patch size is 3, you get (32/3)² ≈ 115 patches (non-integer). If patch size is 5, similar issue. Only patch sizes that evenly divide 32 (1, 2, 4, 8, 16, 32) work.

**Why it happens:** Authors of ViT papers typically assume 224×224 (ImageNet) or 256×256 images. CIFAR-10's small size requires deliberate patch size choice.

**How to avoid:** Use patch_size=4 (64 tokens) or patch_size=8 (16 tokens). Validate: `assert image_size % patch_size == 0`.

**Warning signs:** Runtime errors about tensor shape mismatch when reshaping patches; non-integer division warnings.

### Pitfall 2: Embedding Dimension Not Divisible by Number of Heads

**What goes wrong:** If embed_dim=256 but num_heads=7, head_dim = 256 / 7 ≈ 36.57 (non-integer). Transformer layers fail or silently produce wrong output shapes.

**Why it happens:** num_heads is fixed while embed_dim varies in sweeps. Easy to pick incompatible pairs.

**How to avoid:** Ensure `embed_dim % num_heads == 0` in all model instantiations. For fixed num_heads=8, use embed_dim values like 64, 128, 192, 256, 320, 384, etc.

**Warning signs:** Shape mismatch errors in transformer layer; "head_dim is not an integer" from PyTorch.

### Pitfall 3: Image Normalization Mismatch

**What goes wrong:** CIFAR-10 has specific per-channel statistics (mean [0.4914, 0.4822, 0.4465], std [0.2470, 0.2435, 0.2616]). Using ImageNet stats or no normalization causes poor convergence.

**Why it happens:** Developers copy transforms from ImageNet tutorials without adjusting for CIFAR-10.

**How to avoid:** Always include channel-wise normalization with CIFAR-10 statistics. Verify mean/std are applied in both train and validation transforms.

**Warning signs:** Training loss is stuck high or oscillating; validation accuracy remains near random (10%); model completely fails to learn.

### Pitfall 4: Positional Embedding Dimension Must Match embed_dim

**What goes wrong:** If pos_embed is initialized for embed_dim=256, but model is instantiated with embed_dim=512, shape mismatch on forward pass: (B, num_tokens+1, 512) + (1, num_tokens+1, 256) fails.

**Why it happens:** Positional embeddings are often initialized as a constant tensor. If they're shared or cached, they don't adapt to new sizes.

**How to avoid:** Always initialize positional embeddings inside `__init__` after embed_dim is set. Never reuse pos_embed across instances with different embed_dim.

**Warning signs:** Runtime shape errors when adding positional embeddings; errors mentioning position embedding size.

### Pitfall 5: Sequence Length Interpretation

**What goes wrong:** Code assumes variable sequence length like GPT. But ViT sequence length = (H / patch_size) * (W / patch_size) + 1 (for CLS token). For CIFAR-10 32×32 with patch_size=4, it's always 64 + 1 = 65 tokens.

**Why it happens:** Developers coming from NLP expect variable-length sequences (text). Image patches are fixed per image size.

**How to avoid:** Explicitly document that sequence length is derived from image and patch size, not from input. Validate all images are same size in dataset.

**Warning signs:** Dimension errors in batching; unexpected memory usage spikes.

### Pitfall 6: FLOPs Calculation Must Account for Patch Computation

**What goes wrong:** Standard Chinchilla formula (C = 6*N*T) assumes N tokens, but ViT's patch projection and attention are not identical to GPT's embedding. Underestimating FLOPs can skew scaling law fits.

**Why it happens:** Phase 3 documented flops_per_param_per_token=6 as default. This assumes transformer attention, which holds for ViT, but patch projection adds O(3 * patch_size^2) operations per patch.

**How to avoid:** For initial Phase 8, use default flops_per_param_per_token=6 (it's close enough for scaling law purposes). For precise analysis, compute FLOPs as: 6*N*T + (3 * patch_size^2 * num_patches). Validate with a small model.

**Warning signs:** Scaling laws fit poorly; predicted optimal sizes are wildly off; actual wall time doesn't match FLOP predictions.

## Code Examples

Verified patterns from official sources:

### Minimal ViT Implementation

```python
# Source: adapted from torchvision.models.vision_transformer
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, embed_dim=256, num_layers=6, num_heads=8, num_classes=10, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (32 // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Linear(3 * patch_size * patch_size, embed_dim, bias=True)

        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        # Patch extraction: (B, 3, 32, 32) -> (B, num_patches, 3*patch_size*patch_size)
        patches = x.reshape(
            B, 3,
            32 // self.patch_size, self.patch_size,
            32 // self.patch_size, self.patch_size
        ).permute(0, 2, 4, 1, 3, 5).reshape(B, self.num_patches, -1)

        # Embed patches
        x = self.patch_embed(patches)  # (B, num_patches, embed_dim)

        # Prepend class token
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)  # (B, 1+num_patches, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed

        # Transformer
        x = self.encoder(x)

        # Extract class token and classify
        x = self.norm(x[:, 0])
        return self.head(x)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
```

### CIFAR-10 Data Loading

```python
# Source: https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# For mock mode: create TensorDataset
import torch
images, labels = [], []
for i in range(128):  # Mini dataset
    img, label = train_dataset[i]
    images.append(img)
    labels.append(label)
mock_dataset = torch.utils.data.TensorDataset(
    torch.stack(images),
    torch.tensor(labels)
)
```

### ViT Loss Function

```python
import torch.nn.functional as F

def vit_loss_fn(logits, labels):
    """Classification loss for ViT.

    Args:
        logits: (B, num_classes) model output
        labels: (B,) integer class indices

    Returns:
        Scalar loss
    """
    return F.cross_entropy(logits, labels)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Convolutional networks for image classification | Vision Transformers (ViT) | 2020 (Dosovitskiy et al.) | Transformers match or exceed CNN accuracy without inductive biases; enable scaling laws analysis |
| Fixed sequence length assumptions | Adaptive/dynamic patch sizes (DeiT, APT) | 2021-2025 | Improved efficiency; for Phase 8, keep fixed (CIFAR-10 size fixed) |
| Per-image normalization tuning | Standardized channel-wise statistics | 2015+ (ImageNet standardization) | CIFAR-10 has canonical stats; use them |
| Manual patch extraction in models | Built-in torchvision.models.VisionTransformer | 2021+ | Standardized, tested implementation available |

**Deprecated/outdated:**
- Strided convolutions as first layer: ViT uses linear projection of patches instead (more flexible, scales better).
- Fixed positional encoding (sinusoidal): Learnable positional embeddings now standard (better expressiveness).

## Open Questions

1. **Should ViT use num_heads dynamic with embed_dim, or fixed?**
   - What we know: Fixed num_heads (e.g., 8) requires embed_dim divisible by 8.
   - What's unclear: Whether sweep should vary heads with embed_dim to keep head_dim constant.
   - Recommendation: Keep num_heads fixed at 8 (standard for small models). Ensure embed_dim is a multiple of 8 in size generation (Phase 3).

2. **Should patch size be configurable or fixed?**
   - What we know: Patch size affects sequence length; smaller patches = more tokens = more compute.
   - What's unclear: Whether sweep should test patch_size=4 vs. patch_size=8 to show ViT sensitivity.
   - Recommendation: Fix patch_size=4 for Phase 8 (most information-dense for 32×32 images). Future: add patch size as sweep variable.

3. **How to account for patch projection FLOPs in scaling law?**
   - What we know: Patch projection is O(batch * num_patches * 3*patch_size^2 * embed_dim).
   - What's unclear: Whether default flops_per_param_per_token=6 (designed for transformers) is accurate.
   - Recommendation: Use default for now (close enough for scaling laws). Document the approximation. Validate with profiling on one model size.

4. **Is image resizing needed for variable-size experiments?**
   - What we know: CIFAR-10 is fixed 32×32.
   - What's unclear: Whether all scales should use same image size, or if larger models should see larger images.
   - Recommendation: Keep image size fixed at 32×32 for Phase 8 (simpler, matches Chinchilla principle of isolating one variable). Document that scaling laws assume fixed image resolution.

## Sources

### Primary (HIGH confidence)

- [PyTorch Vision Transformer API](https://docs.pytorch.org/vision/main/models/vision_transformer.html) — official torchvision documentation, constructor parameters, available variants
- [PyTorch CIFAR-10 Tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) — dataset loading, transforms, normalization statistics
- [PyTorch Lightning ViT Tutorial](https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html) — ViT architecture, patch extraction, forward pass structure
- [Existing flops_fit codebase](file:///home/viggie/Projects/flops-fit/src/flops_fit/) — model contract (num_params), loss function interface, trainer integration patterns
- [PyTorch Cross Entropy Loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — official documentation on loss function behavior, shape expectations

### Secondary (MEDIUM confidence)

- [Hugging Face Vision Transformer Documentation](https://huggingface.co/docs/transformers/model_doc/vit) — architecture overview, pretrained models (reference only, not used directly)
- [timm Vision Transformer Models](https://github.com/huggingface/pytorch-image-models) — reference implementations, model variants (ecosystem standard but optional dependency)

### Tertiary (LOW confidence)

- [WebSearch on ViT FLOPs](https://arxiv.org/html/2502.10120v1) — recent optimization papers discuss FLOP computation but are research-stage, not production-ready

## Metadata

**Confidence breakdown:**

- **Standard stack (HIGH):** PyTorch + torchvision + CIFAR-10 are official, stable, documented.
- **Architecture patterns (HIGH):** ViT model structure verified against torchvision source; loss function pattern derived from flops_fit's existing trainer code.
- **Pitfalls (MEDIUM-HIGH):** Based on common ViT implementation mistakes documented in tutorials and research papers; patch size and embedding dimension constraints verified against math.
- **FLOPs calculation (MEDIUM):** Phase 3 default formula is well-documented; ViT-specific adjustments are estimates from research papers, not validated against flops_fit's trainer profiling yet.

**Research date:** 2026-02-17
**Valid until:** 2026-03-17 (30 days; torchvision/PyTorch are stable, CIFAR-10 is immutable, ViT architecture is not expected to change significantly)
