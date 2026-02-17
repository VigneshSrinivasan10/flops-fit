"""TinyStories dataset wrapper for flops_fit examples.

Provides a PyTorch Dataset wrapper around the HuggingFace TinyStories dataset
(roneneldan/TinyStories) for use with flops_fit.find_optimal().

Usage:
    dataset = TinyStoriesDataset(split="train", seq_len=256)
    dataset.prepare_data()  # Downloads and caches HuggingFace dataset
    input_ids, labels = dataset[0]  # Returns (seq_len,) tensors

Note: Requires internet access on first run to download from HuggingFace.
      Data is cached to cache_dir (default: .cache/datasets) for subsequent runs.
"""

from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset


class TinyStoriesDataset(Dataset):
    """PyTorch Dataset wrapper for HuggingFace roneneldan/TinyStories.

    Lazy-loads dataset from HuggingFace on first access. Caches to disk
    to avoid repeated downloads.

    Args:
        split: Dataset split -- "train" or "validation".
        seq_len: Sequence length for language modeling. Sequences are
            truncated to seq_len tokens. Shorter sequences are left-padded
            with the EOS token ID.
        cache_dir: HuggingFace cache directory for downloaded datasets
            and tokenizers. Defaults to ".cache/datasets".
        tokenizer_name: HuggingFace tokenizer name. Defaults to "gpt2"
            (vocab_size=50257, matches GPT model default).
    """

    def __init__(
        self,
        split: str = "train",
        seq_len: int = 256,
        cache_dir: str = ".cache/datasets",
        tokenizer_name: str = "gpt2",
    ):
        self.split = split
        self.seq_len = seq_len
        self.cache_dir = cache_dir
        self.tokenizer_name = tokenizer_name
        # Lazy: do NOT load dataset or tokenizer here
        self._dataset = None
        self._tokenizer = None

    def prepare_data(self) -> None:
        """Download and cache the dataset and tokenizer.

        Call this explicitly before training to ensure data is available.
        If not called, __getitem__ calls it on first access.
        Prints progress so users know it is not hung.
        """
        if self._dataset is not None:
            return
        # Import lazily to avoid import-time side effects
        from datasets import load_dataset
        from transformers import AutoTokenizer

        print(f"Loading tokenizer '{self.tokenizer_name}'...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            cache_dir=self.cache_dir,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        print(f"Loading TinyStories ({self.split} split) from HuggingFace...")
        self._dataset = load_dataset(
            "roneneldan/TinyStories",
            split=self.split,
            cache_dir=self.cache_dir,
            trust_remote_code=False,
        )
        print(f"TinyStories loaded: {len(self._dataset)} examples.")

    def __len__(self) -> int:
        """Return number of examples. Loads dataset if not yet loaded."""
        if self._dataset is None:
            self.prepare_data()
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, labels) tensors for language modeling.

        Both tensors have shape (seq_len,). Labels equal input_ids (the
        trainer is responsible for next-token shifting if needed).

        Args:
            idx: Example index.

        Returns:
            Tuple of (input_ids, labels), both shape (seq_len,), dtype torch.long.
        """
        if self._dataset is None:
            self.prepare_data()

        story = self._dataset[idx]["story"]

        encoded = self._tokenizer(
            story,
            max_length=self.seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)  # (seq_len,)
        labels = input_ids.clone()

        return input_ids, labels
