#!/usr/bin/env python3
"""
Example: Programmatic use of flops_fit.find_optimal() with ViT + CIFAR-10.

This script demonstrates that flops_fit is architecture-agnostic: the same
find_optimal() API call works for image classification (ViT) just as it does
for language modeling (GPT), even though the output format and loss function
differ structurally.

Key differences from the GPT example:
- Size parameter is ``embed_dim`` (not ``d_model``)
- VisionTransformer.forward() returns logits DIRECTLY (not a tuple)
- vit_loss_fn receives logits directly -- no tuple unpacking needed
- Dataset yields (image_tensor, label) pairs, not (input_ids, labels)

Usage (mock mode, no GPU or network needed):
    python -m flops_fit.examples.example_vit_cifar

Usage (real training, requires CIFAR-10 download via torchvision):
    python -m flops_fit.examples.example_vit_cifar --real

The model factory wraps VisionTransformer so find_optimal() can vary embed_dim.
Note: embed_dim must be divisible by num_heads; ensure sweep sizes are multiples of 8.
"""

import argparse
import torch
from torch.utils.data import TensorDataset

import flops_fit
from flops_fit.examples import VisionTransformer, vit_loss_fn, CIFAR10Dataset


def make_vit_factory(num_layers: int = 6, num_heads: int = 8, patch_size: int = 4):
    """Return a factory function that creates VisionTransformer instances for a given embed_dim.

    find_optimal() calls model_cls(embed_dim=N) to create models at different
    sizes. This factory wraps VisionTransformer instantiation so the caller
    only needs to vary embed_dim.

    Args:
        num_layers: Number of transformer layers (fixed across sweep).
        num_heads: Number of attention heads (fixed across sweep).
            embed_dim must be divisible by num_heads; ensure sweep sizes
            are multiples of 8.
        patch_size: Patch size for image tokenization. Must divide image_size (32) evenly.

    Returns:
        Callable: (embed_dim: int) -> VisionTransformer instance with num_params() method.
    """
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


def _make_synthetic_cifar_dataset(size: int = 128):
    """Create a tiny synthetic CIFAR-like dataset for mock mode.

    Returns a torch TensorDataset yielding (image, label) pairs of random
    float32 image tensors and integer class labels in [0, 10). Used when
    CIFAR-10 download is not desired (testing, CI, quick demos).

    Args:
        size: Number of examples in the dataset.

    Returns:
        torch.utils.data.TensorDataset yielding (image, label) tensors.
    """
    images = torch.randn(size, 3, 32, 32)       # float32 already normalized
    labels = torch.randint(0, 10, (size,)).long()  # integer class labels [0..9]
    return TensorDataset(images, labels)


def run(use_real_data: bool = False, output_dir: str = "outputs/vit_cifar"):
    """Run the ViT + CIFAR-10 scaling law experiment.

    Args:
        use_real_data: If True, downloads CIFAR-10 from torchvision and
                       runs real local training (mode="local").
                       If False (default), uses trainer mock mode with
                       a minimal synthetic dataset (mode="mock").
        output_dir: Directory for results.json and plots.
    """
    print("=" * 60)
    print("flops_fit: ViT + CIFAR-10 Scaling Law Example")
    print("=" * 60)

    # 1. Model factory — varies embed_dim, fixes other architecture params
    model_cls = make_vit_factory(num_layers=4, num_heads=8, patch_size=4)

    # 2. Dataset and training mode
    if use_real_data:
        print("Loading CIFAR-10 dataset (requires torchvision + internet)...")
        dataset = CIFAR10Dataset(train=True)
        trainer_mode = "local"
    else:
        # Synthetic dataset for mock mode — avoids any network access
        print("Using synthetic dataset (mock mode, no download needed).")
        dataset = _make_synthetic_cifar_dataset(size=128)
        trainer_mode = "mock"

    # 3. Small compute budgets for a quick demo (use 1e17+ for real experiments)
    compute_budgets = [1e12, 3e12, 1e13, 3e13, 1e14]

    print(f"\nRunning sweep over {len(compute_budgets)} compute budgets...")
    print(f"Trainer mode: {trainer_mode}")
    print(f"Output directory: {output_dir}\n")

    result = flops_fit.find_optimal(
        model_cls=model_cls,
        model_size_param="embed_dim",    # KEY: embed_dim is the size parameter (not d_model)
        dataset=dataset,
        loss_fn=vit_loss_fn,             # KEY: takes (logits, labels) directly — no tuple unpacking
        compute_budgets=compute_budgets,
        train=True,
        mode=trainer_mode,
        output_dir=output_dir,
    )

    # 4. Display results
    print("\n" + "=" * 60)
    print("SCALING LAW RESULTS (Chinchilla Table)")
    print("=" * 60)
    print(result.chinchilla_table())

    print("\nGenerating scaling law plots...")
    result.plot(show=False)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="ViT + CIFAR-10 scaling law example",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real CIFAR-10 data + local training (requires internet). Default: synthetic mock data.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/vit_cifar",
        help="Directory for results and plots.",
    )
    args = parser.parse_args()
    run(use_real_data=args.real, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
