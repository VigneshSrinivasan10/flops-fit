"""flops_fit.examples: Reference implementations for library users.

Provides GPT + TinyStories and VisionTransformer + CIFAR-10 as ready-made
model + dataset pairs for demonstrating flops_fit.find_optimal() usage.
"""
from flops_fit.examples.gpt import GPT, GPTConfig, create_model_for_scaling
from flops_fit.examples.tinystories import TinyStoriesDataset
from flops_fit.examples.vit import VisionTransformer, vit_loss_fn
from flops_fit.examples.cifar import CIFAR10Dataset

__all__ = [
    "GPT",
    "GPTConfig",
    "create_model_for_scaling",
    "TinyStoriesDataset",
    "VisionTransformer",
    "vit_loss_fn",
    "CIFAR10Dataset",
]
