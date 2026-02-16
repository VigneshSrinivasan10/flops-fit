"""Dataset Validation and DataLoader Wrapping

Validates user-provided dataset inputs and normalizes them to DataLoader
instances for use in the training pipeline. Supports map-style Datasets,
IterableDatasets, and existing DataLoaders.

Provides clear error messages when invalid inputs are given, including
a hint for HuggingFace dataset users.
"""

import logging

from torch.utils.data import DataLoader, Dataset, IterableDataset

logger = logging.getLogger(__name__)


def validate_dataset(dataset_or_loader: Dataset | DataLoader) -> None:
    """Validate that the input is a torch Dataset or DataLoader.

    Args:
        dataset_or_loader: The dataset or data loader to validate.

    Raises:
        TypeError: If the input is not a Dataset or DataLoader.
    """
    if dataset_or_loader is None:
        raise TypeError(
            "dataset is required. Expected a torch.utils.data.Dataset "
            "or torch.utils.data.DataLoader."
        )

    if isinstance(dataset_or_loader, (DataLoader, IterableDataset, Dataset)):
        return

    type_name = type(dataset_or_loader).__name__
    raise TypeError(
        f"Expected a torch.utils.data.Dataset or torch.utils.data.DataLoader, "
        f"got {type_name}. "
        f"If using HuggingFace datasets, call `dataset.with_format('torch')` first."
    )


def wrap_dataset(
    dataset_or_loader: Dataset | DataLoader,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Normalize a Dataset or DataLoader into a DataLoader.

    If given an existing DataLoader, returns it as-is. If given a Dataset or
    IterableDataset, wraps it in a new DataLoader with the specified parameters.

    Args:
        dataset_or_loader: The dataset or data loader to wrap.
        batch_size: Batch size for the DataLoader.
        num_workers: Number of worker processes for data loading.
        shuffle: Whether to shuffle the data. Ignored for IterableDataset.

    Returns:
        A DataLoader wrapping the given dataset.

    Raises:
        TypeError: If the input is not a Dataset or DataLoader.
    """
    if isinstance(dataset_or_loader, DataLoader):
        return dataset_or_loader

    if isinstance(dataset_or_loader, IterableDataset):
        return DataLoader(
            dataset_or_loader,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=True,
        )

    if isinstance(dataset_or_loader, Dataset):
        return DataLoader(
            dataset_or_loader,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=True,
        )

    type_name = type(dataset_or_loader).__name__
    raise TypeError(
        f"Expected a torch.utils.data.Dataset or torch.utils.data.DataLoader, "
        f"got {type_name}. "
        f"If using HuggingFace datasets, call `dataset.with_format('torch')` first."
    )
