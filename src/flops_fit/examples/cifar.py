"""CIFAR-10 dataset wrapper for flops_fit examples.

Provides a PyTorch Dataset wrapper around torchvision's CIFAR-10 dataset
for use with flops_fit.find_optimal() and VisionTransformer.

Usage:
    dataset = CIFAR10Dataset(train=True, data_dir="./data")
    img, label = dataset[0]  # Triggers lazy download on first access
    # img shape: (3, 32, 32), label: int

Note: Requires internet access on first access to download from torchvision.
      torchvision is imported lazily inside _prepare_data() to avoid
      import-time overhead when users are not running image experiments.
"""

import torch
from torch.utils.data import Dataset


class CIFAR10Dataset(Dataset):
    """PyTorch Dataset wrapper for torchvision CIFAR-10.

    Lazy-loads dataset from torchvision on first access. Download is
    automatic (download=True). torchvision is imported lazily inside
    _prepare_data() -- zero import-time overhead.

    Returns (image_tensor, label) pairs:
      - image_tensor: (3, 32, 32) float tensor, normalized with CIFAR-10 channel statistics
      - label: int in [0, 9]

    Args:
        train: If True, use training split (50000 images). If False, test split (10000 images).
        data_dir: Directory for dataset storage. Downloads here on first access.
    """

    def __init__(self, train: bool = True, data_dir: str = "./data"):
        self.train = train
        self.data_dir = data_dir
        self._dataset = None  # lazy: not loaded until first access

    def _prepare_data(self) -> None:
        """Download and initialize the CIFAR-10 dataset.

        Called on first access via __len__ or __getitem__. Imports torchvision
        lazily here (not at module top) to avoid import-time overhead.
        """
        if self._dataset is not None:
            return

        # Lazy-import torchvision inside this method only
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ])

        self._dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=self.train,
            download=True,
            transform=transform,
        )

    def __len__(self) -> int:
        """Return number of examples. Downloads dataset if not yet loaded."""
        self._prepare_data()
        return len(self._dataset)

    def __getitem__(self, idx: int):
        """Return (image_tensor, label) for example at idx.

        Args:
            idx: Example index.

        Returns:
            Tuple of (image_tensor, label) where image_tensor has shape (3, 32, 32)
            and label is an integer in [0, 9].
        """
        self._prepare_data()
        return self._dataset[idx]
