"""flops_fit.examples: Reference implementations for library users.

Provides GPT + TinyStories as ready-made model + dataset for
demonstrating flops_fit.find_optimal() usage.
"""
from flops_fit.examples.gpt import GPT, GPTConfig, create_model_for_scaling
from flops_fit.examples.tinystories import TinyStoriesDataset

__all__ = [
    "GPT",
    "GPTConfig",
    "create_model_for_scaling",
    "TinyStoriesDataset",
]
