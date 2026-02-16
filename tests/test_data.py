"""Tests for dataset validation and DataLoader wrapping."""

import pytest
import torch
import torch.utils.data
from torch.utils.data import DataLoader

from flops_fit.data import validate_dataset, wrap_dataset


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, size=100):
        self.data = torch.randn(size, 10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SimpleIterableDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        for i in range(100):
            yield torch.randn(10)


class TestValidateDataset:
    """Test suite for validate_dataset."""

    def test_accepts_map_dataset(self):
        """validate_dataset accepts a map-style Dataset without error."""
        validate_dataset(SimpleDataset())

    def test_accepts_dataloader(self):
        """validate_dataset accepts a DataLoader without error."""
        validate_dataset(DataLoader(SimpleDataset()))

    def test_accepts_iterable_dataset(self):
        """validate_dataset accepts an IterableDataset without error."""
        validate_dataset(SimpleIterableDataset())

    def test_rejects_plain_list(self):
        """validate_dataset rejects a plain list with TypeError."""
        with pytest.raises(TypeError):
            validate_dataset([1, 2, 3])

    def test_rejects_dict(self):
        """validate_dataset rejects a dict with TypeError."""
        with pytest.raises(TypeError):
            validate_dataset({"a": 1})

    def test_rejects_none(self):
        """validate_dataset rejects None with TypeError."""
        with pytest.raises(TypeError):
            validate_dataset(None)

    def test_error_message_mentions_huggingface(self):
        """Error message for rejected type contains HuggingFace hint."""
        with pytest.raises(TypeError, match="with_format"):
            validate_dataset([1, 2, 3])

    def test_error_message_includes_type_name(self):
        """Error for list includes 'list' in message."""
        with pytest.raises(TypeError, match="list"):
            validate_dataset([1, 2, 3])


class TestWrapDataset:
    """Test suite for wrap_dataset."""

    def test_wraps_dataset_to_dataloader(self):
        """wrap_dataset returns a DataLoader when given a Dataset."""
        result = wrap_dataset(SimpleDataset())
        assert isinstance(result, DataLoader)

    def test_wrapped_dataloader_is_iterable(self):
        """Iterating one batch from wrapped DataLoader yields a tensor."""
        loader = wrap_dataset(SimpleDataset())
        batch = next(iter(loader))
        assert isinstance(batch, torch.Tensor)

    def test_passthrough_existing_dataloader(self):
        """wrap_dataset passes through an existing DataLoader unchanged."""
        original = DataLoader(SimpleDataset())
        result = wrap_dataset(original)
        assert result is original

    def test_respects_batch_size(self):
        """Wrapping with batch_size=16 produces batches of that size."""
        loader = wrap_dataset(SimpleDataset(size=64), batch_size=16)
        batch = next(iter(loader))
        assert batch.shape[0] == 16

    def test_rejects_non_dataset(self):
        """wrap_dataset rejects a string with TypeError."""
        with pytest.raises(TypeError):
            wrap_dataset("not a dataset")

    def test_wraps_iterable_dataset(self):
        """wrap_dataset wraps an IterableDataset into a DataLoader."""
        result = wrap_dataset(SimpleIterableDataset())
        assert isinstance(result, DataLoader)
