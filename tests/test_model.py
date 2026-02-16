"""Tests for the GPT model module."""

import pytest
import torch

from flops_fit.model import GPT, GPTConfig, create_model_for_scaling, estimate_params_from_config


class TestGPTConfig:
    """Test suite for GPTConfig."""

    def test_default_d_ff(self):
        """d_ff defaults to 4 * d_model."""
        config = GPTConfig(d_model=256)
        assert config.d_ff == 1024

    def test_custom_d_ff(self):
        """Custom d_ff overrides the default."""
        config = GPTConfig(d_model=256, d_ff=512)
        assert config.d_ff == 512

    def test_head_divisibility_assertion(self):
        """d_model must be divisible by num_heads."""
        with pytest.raises(AssertionError):
            GPTConfig(d_model=256, num_heads=3)


class TestGPT:
    """Test suite for GPT model."""

    def test_forward_output_shape(self):
        """Forward pass produces correct logits shape, no loss without labels."""
        config = GPTConfig(
            vocab_size=100, d_model=64, num_layers=2, num_heads=2, max_seq_len=32
        )
        model = GPT(config)
        input_ids = torch.randint(0, 100, (2, 16))

        with torch.no_grad():
            logits, loss = model(input_ids)

        assert logits.shape == (2, 16, 100)
        assert loss is None

    def test_forward_with_labels_computes_loss(self):
        """Forward pass with labels returns a finite positive loss."""
        config = GPTConfig(
            vocab_size=100, d_model=64, num_layers=2, num_heads=2, max_seq_len=32
        )
        model = GPT(config)
        input_ids = torch.randint(0, 100, (2, 16))
        labels = torch.randint(0, 100, (2, 16))

        logits, loss = model(input_ids, labels=labels)

        assert loss is not None
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_count_parameters_non_embedding(self):
        """Non-embedding param count is less than total; difference equals embedding size."""
        config = GPTConfig(
            vocab_size=100, d_model=64, num_layers=2, num_heads=2, max_seq_len=32
        )
        model = GPT(config)

        non_emb = model.count_parameters(non_embedding=True)
        total = model.count_parameters(non_embedding=False)

        assert non_emb > 0
        assert non_emb < total
        assert total - non_emb == config.vocab_size * config.d_model

    def test_count_parameters_total(self):
        """Total param count matches estimate_params_from_config."""
        config = GPTConfig(
            vocab_size=100, d_model=128, num_layers=4, num_heads=4, max_seq_len=32
        )
        model = GPT(config)

        actual_total = model.count_parameters(non_embedding=False)
        estimate = estimate_params_from_config(
            d_model=128, num_layers=4, vocab_size=100
        )

        assert actual_total == estimate["total"]

    def test_umup_initialization_lm_head_is_zero(self):
        """u-mup parametrization initializes lm_head weights to zero."""
        config = GPTConfig(
            vocab_size=100, d_model=64, num_layers=2, num_heads=2,
            parametrization="u-mup",
        )
        model = GPT(config)
        assert torch.all(model.lm_head.weight == 0)

    def test_umup_initialization_embedding_nonzero(self):
        """u-mup parametrization initializes embedding weights to nonzero."""
        config = GPTConfig(
            vocab_size=100, d_model=64, num_layers=2, num_heads=2,
            parametrization="u-mup",
        )
        model = GPT(config)
        assert not torch.all(model.token_emb.weight == 0)

    def test_sp_initialization(self):
        """SP parametrization creates model without error; lm_head is zero."""
        config = GPTConfig(
            vocab_size=100, d_model=64, num_layers=2, num_heads=2,
            parametrization="sp",
        )
        model = GPT(config)
        assert torch.all(model.lm_head.weight == 0)

    def test_from_config_classmethod(self):
        """from_config creates a GPT instance with correct config."""
        model = GPT.from_config(d_model=64, num_layers=2, num_heads=2, vocab_size=100, max_seq_len=32)
        assert isinstance(model, GPT)
        assert model.config.d_model == 64


class TestCreateModelForScaling:
    """Test suite for create_model_for_scaling helper."""

    def test_creates_model_near_target(self):
        """Model param count is within 10x of target.

        Characterization: uses 12*L*d^2 approximation but actual per-layer
        params are 16*d^2 (4*d^2 attention + 12*d^2 SwiGLU FFN) plus
        d_model is rounded to nearest multiple of 64. This produces
        significantly more params than target, especially at small scales.
        """
        model = create_model_for_scaling(target_params=1_000_000, num_layers=4)
        actual = model.count_parameters(non_embedding=True)

        assert actual > 0
        # The approximation is very rough at small scales due to d_model rounding
        assert actual < 10.0 * 1_000_000

    def test_d_model_is_multiple_of_64(self):
        """d_model is always a multiple of 64 for hardware efficiency."""
        for target in [500_000, 1_000_000, 5_000_000, 10_000_000]:
            model = create_model_for_scaling(target_params=target, num_layers=4)
            assert model.config.d_model % 64 == 0

    def test_d_model_divisible_by_num_heads(self):
        """d_model is always divisible by num_heads."""
        model = create_model_for_scaling(target_params=1_000_000, num_layers=4)
        assert model.config.d_model % model.config.num_heads == 0


class TestEstimateParamsFromConfig:
    """Test suite for estimate_params_from_config."""

    def test_estimate_matches_actual_model(self):
        """Estimate total matches actual GPT model param count exactly."""
        config = GPTConfig(
            vocab_size=100, d_model=128, num_layers=4, num_heads=4, max_seq_len=32
        )
        model = GPT(config)

        actual_total = model.count_parameters(non_embedding=False)
        estimate = estimate_params_from_config(d_model=128, num_layers=4, vocab_size=100)

        assert actual_total == estimate["total"]
