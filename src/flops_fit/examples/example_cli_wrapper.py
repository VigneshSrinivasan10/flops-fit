#!/usr/bin/env python3
"""
Example: CLI wrapper for flops_fit.find_optimal() using argparse.

This demonstrates how to expose the library via command-line arguments.
Users can adapt this pattern for their own model + dataset combinations.

All library configuration is passed as arguments; no YAML files required.

Usage:
    python -m flops_fit.examples.example_cli_wrapper --help
    python -m flops_fit.examples.example_cli_wrapper --budgets 1e12 3e12 1e13
    python -m flops_fit.examples.example_cli_wrapper --real --layers 6 --output-dir results/

The key pattern:
    1. Parse args with argparse
    2. Build model_cls factory from arch params
    3. Build dataset from data params
    4. Call flops_fit.find_optimal() with compute_budgets from args
    5. Print results
"""

import argparse
import torch
import torch.nn.functional as F

import flops_fit
from flops_fit.examples import GPT, GPTConfig, TinyStoriesDataset


VOCAB_SIZE = 50257


def make_model_factory(num_layers: int, num_heads: int):
    """Return a GPT factory callable for find_optimal().

    Args:
        num_layers: Fixed number of layers across the sweep.
        num_heads: Fixed number of attention heads.

    Returns:
        Callable: (d_model: int) -> GPT instance with num_params() method.
    """
    def create_gpt(d_model: int) -> GPT:
        config = GPTConfig(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            vocab_size=VOCAB_SIZE,
            max_seq_len=256,
        )
        return GPT(config)
    return create_gpt


def gpt_loss_fn(outputs, labels):
    """Cross-entropy loss over GPT's (B, T, V) logits."""
    logits, _ = outputs
    return F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1))


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(
        description="flops_fit: GPT + TinyStories scaling law CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Architecture
    p.add_argument("--layers", type=int, default=4, help="Number of transformer layers")
    p.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    # Dataset
    p.add_argument("--real", action="store_true",
                   help="Use real TinyStories + local training (requires internet). Default: synthetic mock.")
    p.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    p.add_argument("--cache-dir", default=".cache/datasets", help="HuggingFace cache directory")
    # Sweep
    p.add_argument(
        "--budgets", nargs="+", type=float,
        default=[1e12, 3e12, 1e13, 3e13, 1e14],
        help="Compute budgets in FLOPs",
    )
    # Output
    p.add_argument("--output-dir", default="outputs/gpt_tinystories", help="Results directory")
    return p


def main():
    args = build_parser().parse_args()

    print("=" * 60)
    print("flops_fit CLI: GPT + TinyStories")
    print("=" * 60)
    print(f"Architecture: {args.layers} layers, {args.heads} heads")
    print(f"Budgets: {args.budgets}")
    print(f"Dataset: {'TinyStories (real)' if args.real else 'Synthetic (mock)'}")
    print()

    # Model factory
    model_cls = make_model_factory(num_layers=args.layers, num_heads=args.heads)

    # Dataset and training mode
    if args.real:
        print("Loading TinyStories dataset...")
        dataset = TinyStoriesDataset(
            split="train",
            seq_len=args.seq_len,
            cache_dir=args.cache_dir,
        )
        dataset.prepare_data()
        trainer_mode = "local"
    else:
        from torch.utils.data import TensorDataset
        print("Using synthetic dataset (no download).")
        data = torch.randint(0, VOCAB_SIZE, (128, args.seq_len))
        dataset = TensorDataset(data, data.clone())
        trainer_mode = "mock"

    # Run find_optimal — pass mode so TrainingRunner uses the right backend
    result = flops_fit.find_optimal(
        model_cls=model_cls,
        model_size_param="d_model",
        dataset=dataset,
        loss_fn=gpt_loss_fn,
        compute_budgets=args.budgets,
        train=True,
        mode=trainer_mode,
        output_dir=args.output_dir,
    )

    print(result.chinchilla_table())


if __name__ == "__main__":
    main()
