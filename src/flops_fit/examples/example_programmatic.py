#!/usr/bin/env python3
"""
Example: Programmatic use of flops_fit.find_optimal() with GPT + TinyStories.

This script demonstrates the full scaling law workflow:
1. Define a model class using GPT from flops_fit.examples
2. Load TinyStories dataset (or use mock mode without network)
3. Call find_optimal() with compute budgets
4. Display the Chinchilla scaling table

Usage (mock mode, no GPU or network needed):
    python -m flops_fit.examples.example_programmatic

Usage (real training, requires TinyStories download):
    python -m flops_fit.examples.example_programmatic --real

The model factory wraps GPT so find_optimal() can vary d_model.
The loss function reshapes GPT's (B, T, V) logits for cross_entropy.
"""

import argparse
import torch
import torch.nn.functional as F

import flops_fit
from flops_fit.examples import GPT, GPTConfig, TinyStoriesDataset


VOCAB_SIZE = 50257  # GPT-2 tokenizer vocab size


def make_model_factory(num_layers: int = 4, num_heads: int = 4):
    """Return a factory function that creates GPT instances for a given d_model.

    find_optimal() calls model_cls(d_model=N) to create models at different
    sizes. This factory wraps GPTConfig instantiation so the caller only
    needs to vary d_model.

    Args:
        num_layers: Number of transformer layers (fixed across sweep).
        num_heads: Number of attention heads (fixed across sweep).

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
    """Language modeling loss for GPT outputs.

    GPT.forward() returns (logits, loss_or_None). This loss_fn is called
    by the trainer with (model_output, labels) where model_output is the
    raw return value of model.forward(input_ids).

    Args:
        outputs: Tuple of (logits, _) from GPT.forward(), logits shape (B, T, V).
        labels: Target token IDs, shape (B, T).

    Returns:
        Scalar cross-entropy loss.
    """
    logits, _ = outputs
    # Reshape for cross_entropy: (B*T, V) and (B*T,)
    return F.cross_entropy(
        logits.view(-1, VOCAB_SIZE),
        labels.view(-1),
    )


def run(use_real_data: bool = False, output_dir: str = "outputs/gpt_tinystories"):
    """Run the scaling law experiment.

    Args:
        use_real_data: If True, downloads TinyStories from HuggingFace and
                       runs real local training (mode="local").
                       If False (default), uses trainer mock mode with
                       a minimal synthetic dataset (mode="mock").
        output_dir: Directory for results.json and plots.
    """
    print("=" * 60)
    print("flops_fit: GPT + TinyStories Scaling Law Example")
    print("=" * 60)

    # 1. Model factory — varies d_model, fixes other architecture params
    model_cls = make_model_factory(num_layers=4, num_heads=4)

    # 2. Dataset and training mode
    if use_real_data:
        print("Loading TinyStories dataset (requires internet)...")
        dataset = TinyStoriesDataset(split="train", seq_len=256)
        dataset.prepare_data()
        trainer_mode = "local"
    else:
        # Synthetic dataset for mock mode — avoids any network access
        print("Using synthetic dataset (mock mode, no download needed).")
        dataset = _make_synthetic_dataset(seq_len=256, size=128)
        trainer_mode = "mock"

    # 3. Small compute budgets for a quick demo (use 1e17+ for real experiments)
    compute_budgets = [1e12, 3e12, 1e13, 3e13, 1e14]

    print(f"\nRunning sweep over {len(compute_budgets)} compute budgets...")
    print(f"Trainer mode: {trainer_mode}")
    print(f"Output directory: {output_dir}\n")

    result = flops_fit.find_optimal(
        model_cls=model_cls,
        model_size_param="d_model",
        dataset=dataset,
        loss_fn=gpt_loss_fn,
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
    figs = result.plot(show=False)
    print(f"Saved {len(figs)} figure(s) to {output_dir}/plots/")

    return result


def _make_synthetic_dataset(seq_len: int = 256, size: int = 128):
    """Create a tiny synthetic dataset for mock/CPU mode.

    Returns a torch Dataset yielding (input_ids, labels) pairs of random
    token IDs in [0, VOCAB_SIZE). Used when TinyStories download is not
    desired (testing, CI, quick demos).

    Args:
        seq_len: Sequence length for each example.
        size: Number of examples in the dataset.

    Returns:
        torch.utils.data.Dataset yielding (input_ids, labels) tensors.
    """
    from torch.utils.data import TensorDataset
    data = torch.randint(0, VOCAB_SIZE, (size, seq_len))
    return TensorDataset(data, data.clone())


def main():
    parser = argparse.ArgumentParser(
        description="GPT + TinyStories scaling law example",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real TinyStories data + local training (requires internet). Default: synthetic mock data.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/gpt_tinystories",
        help="Directory for results and plots.",
    )
    args = parser.parse_args()
    run(use_real_data=args.real, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
