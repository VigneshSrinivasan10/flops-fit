"""Vision Transformer (ViT) for image classification scaling experiments.

Implements a custom ViT class with embed_dim as the size parameter,
suitable for scaling law experiments with CIFAR-10.

Key differences from GPT example:
- forward() returns logits DIRECTLY (not a tuple)
- vit_loss_fn receives logits directly (no tuple unpacking)
- Uses patch projection instead of token embedding
- Sequence length is fixed (derived from image_size and patch_size)

References:
- Dosovitskiy et al. (2020): "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- torchvision.models.vision_transformer (reference implementation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionTransformer(nn.Module):
    """Vision Transformer for image classification.

    Uses embed_dim as the primary size parameter for scaling experiments.
    forward() returns logits directly -- not a tuple -- distinguishing it
    from GPT whose forward() returns (logits, loss).

    Args:
        embed_dim: Embedding dimension (the size parameter for sweeps). Default: 256.
        image_size: Input image spatial dimension (assumes square). Default: 32.
        patch_size: Patch size (must divide image_size evenly). Default: 4.
        num_classes: Number of output classes. Default: 10 (CIFAR-10).
        num_layers: Number of transformer encoder layers. Default: 6.
        num_heads: Number of attention heads (embed_dim must be divisible). Default: 8.
        mlp_dim: Feed-forward hidden dimension. Defaults to 4 * embed_dim.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        image_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 10,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_dim: int = None,
    ):
        super().__init__()

        assert image_size % patch_size == 0, (
            f"patch_size {patch_size} must divide image_size {image_size}"
        )
        assert embed_dim % num_heads == 0, (
            f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        )

        if mlp_dim is None:
            mlp_dim = 4 * embed_dim

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2

        # Patch projection: flattened patch -> embed_dim
        self.patch_embed = nn.Linear(3 * patch_size * patch_size, embed_dim, bias=True)

        # [CLS] token prepended to patch sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable positional embeddings (num_patches + 1 for CLS token)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Transformer encoder with pre-norm (norm_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final norm applied to [CLS] token before classification head
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head: [CLS] -> num_classes logits
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for image classification.

        Args:
            x: Input images, shape (B, 3, image_size, image_size).

        Returns:
            logits: Class logits, shape (B, num_classes). DIRECT return (not a tuple).
        """
        B = x.shape[0]

        # Patch extraction: (B, 3, H, W) -> (B, num_patches, 3*patch_size*patch_size)
        patches = (
            x.reshape(
                B,
                3,
                self.image_size // self.patch_size,
                self.patch_size,
                self.image_size // self.patch_size,
                self.patch_size,
            )
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(B, self.num_patches, -1)
        )

        # Project patches to embedding dimension
        x = self.patch_embed(patches)  # (B, num_patches, embed_dim)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)          # (B, num_patches+1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed

        # Transformer encoder
        x = self.encoder(x)

        # Extract [CLS] token, apply norm, classify
        x = self.norm(x[:, 0])   # (B, embed_dim)
        return self.head(x)       # (B, num_classes) -- logits DIRECTLY

    def num_params(self) -> int:
        """Return total trainable parameter count (flops_fit model contract)."""
        return sum(p.numel() for p in self.parameters())


def vit_loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Image classification loss for ViT.

    Logits shape: (B, num_classes). Labels shape: (B,). Returns scalar cross-entropy.

    NOTE: unlike gpt_loss_fn, logits are direct (not a tuple). VisionTransformer.forward()
    returns logits directly, so this function receives them as-is with no tuple unpacking.

    Args:
        logits: Model output, shape (B, num_classes).
        labels: Target class indices, shape (B,).

    Returns:
        Scalar cross-entropy loss.
    """
    return F.cross_entropy(logits, labels)
