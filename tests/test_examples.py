"""Tests for flops_fit.examples package.

TinyStories tests mock HuggingFace to avoid network access in CI.
Example script tests use synthetic datasets (no mocking needed for data).
"""

import pytest
import torch
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# GPT contract
# ---------------------------------------------------------------------------

class TestGPTContract:
    """GPT in examples satisfies the flops_fit model contract."""

    def test_num_params_method_exists(self):
        """GPT.num_params() exists and returns a positive integer.

        num_params() is the primary contract method checked by
        model_factory.validate_model_contract(). NOT n().
        """
        from flops_fit.examples import GPT, GPTConfig
        config = GPTConfig(d_model=64, num_layers=2, num_heads=2, vocab_size=100, max_seq_len=32)
        model = GPT(config)
        result = model.num_params()
        assert isinstance(result, int)
        assert result > 0

    def test_num_params_increases_with_d_model(self):
        """Larger d_model produces larger num_params()."""
        from flops_fit.examples import GPT, GPTConfig
        small = GPT(GPTConfig(d_model=64, num_layers=2, num_heads=2, vocab_size=100, max_seq_len=32))
        large = GPT(GPTConfig(d_model=128, num_layers=2, num_heads=2, vocab_size=100, max_seq_len=32))
        assert large.num_params() > small.num_params()

    def test_backward_compat_import_from_model_py(self):
        """GPT can still be imported from flops_fit.model (backward compat)."""
        from flops_fit.model import GPT, GPTConfig
        config = GPTConfig(d_model=64, num_layers=2, num_heads=2, vocab_size=100, max_seq_len=32)
        model = GPT(config)
        assert model.num_params() > 0


# ---------------------------------------------------------------------------
# TinyStoriesDataset: lazy loading
# ---------------------------------------------------------------------------

class TestTinyStoriesDatasetLazyLoad:
    """TinyStoriesDataset does not trigger network access at import or instantiation."""

    def test_import_is_instant(self):
        """TinyStoriesDataset imports without HuggingFace side effects."""
        # If this import triggered a download, CI would hang.
        from flops_fit.examples.tinystories import TinyStoriesDataset
        assert TinyStoriesDataset is not None

    def test_instantiation_does_not_call_load_dataset(self):
        """TinyStoriesDataset() does not call load_dataset at __init__."""
        from flops_fit.examples.tinystories import TinyStoriesDataset
        with patch("flops_fit.examples.tinystories.TinyStoriesDataset.prepare_data") as mock_prepare:
            ds = TinyStoriesDataset(split="train", seq_len=128)
            mock_prepare.assert_not_called()

    def test_attributes_stored(self):
        """Constructor stores all attributes for later use."""
        from flops_fit.examples.tinystories import TinyStoriesDataset
        ds = TinyStoriesDataset(split="validation", seq_len=64, cache_dir="/tmp/cache", tokenizer_name="gpt2")
        assert ds.split == "validation"
        assert ds.seq_len == 64
        assert ds.cache_dir == "/tmp/cache"
        assert ds.tokenizer_name == "gpt2"
        assert ds._dataset is None
        assert ds._tokenizer is None


# ---------------------------------------------------------------------------
# TinyStoriesDataset: mocked HuggingFace
# ---------------------------------------------------------------------------

def _make_mock_hf_dataset(stories, seq_len):
    """Build a minimal mock HuggingFace dataset-like object."""
    records = [{"story": s} for s in stories]
    mock_ds = MagicMock()
    mock_ds.__len__ = MagicMock(return_value=len(records))
    mock_ds.__getitem__ = MagicMock(side_effect=lambda i: records[i])
    return mock_ds


def _make_mock_tokenizer(seq_len):
    """Build a mock HuggingFace tokenizer."""
    mock_tok = MagicMock()
    mock_tok.pad_token = mock_tok.eos_token  # already set
    # When called as tokenizer(text, ...) returns encoded dict
    def tokenize(text, max_length=seq_len, truncation=True, padding="max_length", return_tensors="pt"):
        ids = torch.zeros(1, max_length, dtype=torch.long)
        return {"input_ids": ids}
    mock_tok.side_effect = tokenize
    return mock_tok


class TestTinyStoriesDatasetWithMock:
    """TinyStoriesDataset behavior with mocked HuggingFace."""

    @pytest.fixture
    def ds_with_mock(self):
        """Return a TinyStoriesDataset with HuggingFace mocked."""
        from flops_fit.examples.tinystories import TinyStoriesDataset

        stories = ["Once upon a time.", "The end.", "A story."]
        seq_len = 32

        mock_hf_ds = _make_mock_hf_dataset(stories, seq_len)
        mock_tok = _make_mock_tokenizer(seq_len)

        ds = TinyStoriesDataset(split="train", seq_len=seq_len)

        # Directly inject mocked objects (bypasses HuggingFace)
        ds._dataset = mock_hf_ds
        ds._tokenizer = mock_tok

        return ds, stories, seq_len

    def test_len_returns_dataset_length(self, ds_with_mock):
        ds, stories, seq_len = ds_with_mock
        assert len(ds) == len(stories)

    def test_getitem_returns_tensor_pair(self, ds_with_mock):
        ds, stories, seq_len = ds_with_mock
        input_ids, labels = ds[0]
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert input_ids.shape == (seq_len,)
        assert labels.shape == (seq_len,)

    def test_labels_equal_input_ids(self, ds_with_mock):
        ds, stories, seq_len = ds_with_mock
        input_ids, labels = ds[0]
        assert torch.equal(input_ids, labels)

    def test_dtype_is_long(self, ds_with_mock):
        ds, stories, seq_len = ds_with_mock
        input_ids, labels = ds[0]
        assert input_ids.dtype == torch.long
        assert labels.dtype == torch.long


# ---------------------------------------------------------------------------
# Example scripts: smoke tests (no HuggingFace, no GPU)
# ---------------------------------------------------------------------------

class TestProgrammaticExample:
    """Smoke tests for the programmatic example script."""

    def test_make_model_factory_creates_gpt_with_num_params(self):
        """make_model_factory returns a callable that produces GPT with num_params()."""
        from flops_fit.examples.example_programmatic import make_model_factory
        factory = make_model_factory(num_layers=2, num_heads=2)
        model = factory(d_model=64)
        assert hasattr(model, "num_params")
        assert model.num_params() > 0

    def test_make_synthetic_dataset_shape(self):
        """_make_synthetic_dataset returns Dataset with correct shape."""
        from flops_fit.examples.example_programmatic import _make_synthetic_dataset
        ds = _make_synthetic_dataset(seq_len=16, size=8)
        assert len(ds) == 8
        ids, labels = ds[0]
        assert ids.shape == (16,)
        assert labels.shape == (16,)

    def test_gpt_loss_fn_shape(self):
        """gpt_loss_fn returns scalar given correct shapes."""
        from flops_fit.examples.example_programmatic import gpt_loss_fn, VOCAB_SIZE
        B, T = 2, 8
        logits = torch.randn(B, T, VOCAB_SIZE)
        labels = torch.randint(0, VOCAB_SIZE, (B, T))
        outputs = (logits, None)
        loss = gpt_loss_fn(outputs, labels)
        assert loss.shape == ()  # scalar
        assert loss.item() > 0


class TestCLIWrapperExample:
    """Smoke tests for the CLI wrapper example script."""

    def test_build_parser(self):
        """build_parser() creates a valid ArgumentParser with expected args."""
        from flops_fit.examples.example_cli_wrapper import build_parser
        parser = build_parser()
        # Parse empty args — should use defaults
        args = parser.parse_args([])
        assert args.layers == 4
        assert args.heads == 4
        assert not args.real
        assert args.seq_len == 256

    def test_make_model_factory(self):
        """make_model_factory in CLI wrapper creates GPT with num_params()."""
        from flops_fit.examples.example_cli_wrapper import make_model_factory
        factory = make_model_factory(num_layers=2, num_heads=2)
        model = factory(d_model=64)
        assert model.num_params() > 0

    def test_gpt_loss_fn_cli(self):
        """gpt_loss_fn in CLI wrapper returns scalar."""
        from flops_fit.examples.example_cli_wrapper import gpt_loss_fn, VOCAB_SIZE
        B, T = 2, 8
        logits = torch.randn(B, T, VOCAB_SIZE)
        labels = torch.randint(0, VOCAB_SIZE, (B, T))
        loss = gpt_loss_fn((logits, None), labels)
        assert loss.ndim == 0  # scalar


# ---------------------------------------------------------------------------
# ViT contract
# ---------------------------------------------------------------------------

class TestViTContract:
    """VisionTransformer in examples satisfies the flops_fit model contract."""

    def test_num_params_method_exists(self):
        """VisionTransformer.num_params() exists and returns a positive integer."""
        from flops_fit.examples import VisionTransformer
        model = VisionTransformer(embed_dim=64, num_layers=2, num_heads=8)
        result = model.num_params()
        assert isinstance(result, int)
        assert result > 0

    def test_num_params_increases_with_embed_dim(self):
        """Larger embed_dim produces larger num_params()."""
        from flops_fit.examples import VisionTransformer
        small = VisionTransformer(embed_dim=64, num_layers=2, num_heads=8)
        large = VisionTransformer(embed_dim=128, num_layers=2, num_heads=8)
        assert large.num_params() > small.num_params()

    def test_forward_returns_correct_shape(self):
        """forward(batch) returns shape (B, num_classes)."""
        from flops_fit.examples import VisionTransformer
        model = VisionTransformer(embed_dim=64, num_layers=2, num_heads=8, num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)

    def test_forward_returns_tensor_not_tuple(self):
        """VisionTransformer.forward() returns a Tensor directly, not a tuple.

        This is the structural contrast with GPT, which returns (logits, loss).
        """
        from flops_fit.examples import VisionTransformer
        model = VisionTransformer(embed_dim=64, num_layers=2, num_heads=8)
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert type(output) is torch.Tensor


# ---------------------------------------------------------------------------
# ViT loss function
# ---------------------------------------------------------------------------

class TestViTLossFunction:
    """vit_loss_fn accepts direct logits (no tuple unpacking)."""

    def test_vit_loss_fn_accepts_direct_logits(self):
        """vit_loss_fn(logits, labels) returns scalar given correct shapes."""
        from flops_fit.examples import vit_loss_fn
        logits = torch.randn(2, 10)
        labels = torch.randint(0, 10, (2,))
        loss = vit_loss_fn(logits, labels)
        assert loss.ndim == 0

    def test_vit_loss_fn_returns_scalar(self):
        """vit_loss_fn output has ndim == 0 (scalar tensor)."""
        from flops_fit.examples import vit_loss_fn
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss = vit_loss_fn(logits, labels)
        assert loss.ndim == 0

    def test_vit_loss_fn_positive(self):
        """vit_loss_fn output is a positive scalar (cross-entropy >= 0)."""
        from flops_fit.examples import vit_loss_fn
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss = vit_loss_fn(logits, labels)
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# CIFAR10Dataset: lazy loading
# ---------------------------------------------------------------------------

class TestCIFAR10DatasetLazyLoad:
    """CIFAR10Dataset does not trigger network access at import or instantiation."""

    def test_import_is_instant(self):
        """CIFAR10Dataset imports without torchvision side effects."""
        from flops_fit.examples import CIFAR10Dataset
        assert CIFAR10Dataset is not None

    def test_instantiation_does_not_load_data(self):
        """CIFAR10Dataset() does not download or load data at __init__."""
        from flops_fit.examples import CIFAR10Dataset
        ds = CIFAR10Dataset(train=False, data_dir="/tmp/test")
        assert ds._dataset is None

    def test_attributes_stored(self):
        """Constructor stores train and data_dir for later use."""
        from flops_fit.examples import CIFAR10Dataset
        ds = CIFAR10Dataset(train=False, data_dir="/tmp/test")
        assert ds.train is False
        assert ds.data_dir == "/tmp/test"


# ---------------------------------------------------------------------------
# ViT example script: smoke tests (no download, no GPU)
# ---------------------------------------------------------------------------

class TestViTExampleScript:
    """Smoke tests for the ViT + CIFAR example script."""

    def test_make_vit_factory_creates_vit_with_num_params(self):
        """make_vit_factory returns a callable producing VisionTransformer with num_params() > 0."""
        from flops_fit.examples.example_vit_cifar import make_vit_factory
        factory = make_vit_factory(num_layers=2, num_heads=8)
        model = factory(embed_dim=64)
        assert hasattr(model, "num_params")
        assert model.num_params() > 0

    def test_make_synthetic_cifar_dataset_shape(self):
        """_make_synthetic_cifar_dataset returns Dataset with correct shape."""
        from flops_fit.examples.example_vit_cifar import _make_synthetic_cifar_dataset
        ds = _make_synthetic_cifar_dataset(size=8)
        assert len(ds) == 8
        img, label = ds[0]
        assert img.shape == (3, 32, 32)
        assert isinstance(label.item(), int)

    def test_vit_loss_fn_imported_correctly(self):
        """vit_loss_fn from example_vit_cifar behaves identically to the one in flops_fit.examples.vit."""
        from flops_fit.examples.example_vit_cifar import vit_loss_fn as example_loss_fn
        from flops_fit.examples.vit import vit_loss_fn as vit_module_loss_fn
        # Both should produce the same result for the same inputs
        logits = torch.randn(3, 10)
        labels = torch.randint(0, 10, (3,))
        loss_a = example_loss_fn(logits, labels)
        loss_b = vit_module_loss_fn(logits, labels)
        assert torch.allclose(loss_a, loss_b)
        assert loss_a.ndim == 0
