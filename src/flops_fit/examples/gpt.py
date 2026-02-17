"""
flops-fit GPT Example Implementation

GPT-style transformer with u-mup (Unit-Scaled Maximal Update Parametrization) for scaling law experiments.

The u-mup recipe ensures hyperparameters transfer across model widths:
- Initialization: std = base_std * (fan_in) ** -0.5
- Learning rate: scales inversely with width (base_width / hidden_dim)
- Weight decay: scales with width (hidden_dim / base_width)
- Output layer: initialized to zero

References:
- Blake et al. (2024): "u-μP: The Unit-Scaled Maximal Update Parametrization" (arXiv:2407.17465)
- Yang et al. (2022): "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"
- https://github.com/VigneshSrinivasan10/scaling-recipes
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Configuration for GPT model."""

    # Model architecture
    vocab_size: int = 50257      # GPT-2 tokenizer vocab size
    d_model: int = 256           # Hidden dimension (width)
    num_layers: int = 6          # Number of transformer blocks
    num_heads: int = 4           # Number of attention heads
    d_ff: Optional[int] = None   # FFN hidden dim (default: 4 * d_model)
    max_seq_len: int = 256       # Maximum sequence length

    # Regularization
    dropout: float = 0.0         # Dropout rate (0 for scaling experiments)

    # u-mup settings
    parametrization: str = "u-mup" # "u-mup" or "sp" (standard parameterization)
    base_width: int = 32         # Base width for u-mup LR scaling
    base_width_wd: int = 1024    # Base width for u-mup WD scaling
    base_std: float = 0.02       # Base std for initialization

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = max_seq_len
        self._cos_cached = None
        self._sin_cached = None
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        self._seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos()[None, None, :, :]
        self._sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[2]
        if seq_len > self._seq_len_cached:
            self._build_cache(seq_len)
        return (
            self._cos_cached[:, :, :seq_len, :].to(x.device),
            self._sin_cached[:, :, :seq_len, :].to(x.device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to queries and keys."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with u-mup."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads

        # QKV projection (combined for efficiency)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)

        # Output projection
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Rotary embeddings
        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        # QKV projection
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        # Reshape to (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary(q)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention with causal mask
        # Using PyTorch 2.0+ scaled_dot_product_attention for efficiency
        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.config.dropout if self.training else 0.0)
        else:
            # Fallback for older PyTorch
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q @ k.transpose(-2, -1)) * scale

            # Causal mask
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal_mask, float('-inf'))

            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            y = attn @ v

        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.dropout(y)

        return y


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)  # Gate
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (x * W1) * SiLU(x * W3), then W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GPT(nn.Module):
    """
    GPT-style language model with u-mup parameterization.

    u-mup ensures hyperparameters (especially learning rate) transfer across
    model widths, which is crucial for scaling law experiments.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Final norm and output projection
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (optional but common)
        # self.lm_head.weight = self.token_emb.weight

        # Initialize parameters
        if config.parametrization == "u-mup":
            self.reset_parameters_umup()
        elif config.parametrization == "sp":
            self.reset_parameters_sp()
        else:
            raise ValueError(f"Invalid parametrization: {config.parametrization}")

    def reset_parameters_umup(self) -> None:
        """
        u-mup initialization:
        - Hidden layers: std = base_std * (fan_in) ** -0.5
        - Output layer (lm_head): zeros
        """
        base_std = self.config.base_std

        for name, param in self.named_parameters():
            if param.dim() < 2:
                # Bias and 1D params: skip or small init
                continue

            # Output layer: initialize to zero
            if "lm_head" in name:
                nn.init.zeros_(param)
            # Embedding: normal with base_std
            elif "token_emb" in name:
                nn.init.normal_(param, mean=0.0, std=base_std)
            # Hidden layers: scale with fan_in
            else:
                fan_in = param.shape[1] if param.dim() >= 2 else param.shape[0]
                std = base_std * (fan_in ** -0.5)
                nn.init.normal_(param, mean=0.0, std=std)

    def reset_parameters_sp(self) -> None:
        """Standard parameterization (Xavier/Kaiming)."""
        for name, param in self.named_parameters():
            if param.dim() < 2:
                continue
            if "lm_head" in name:
                nn.init.zeros_(param)
            else:
                nn.init.xavier_normal_(param)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: (B, T) token indices
            labels: (B, T) target token indices for loss computation

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar loss if labels provided
        """
        B, T = input_ids.shape

        # Token embeddings (no position embedding with RoPE)
        x = self.token_emb(input_ids)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def configure_optimizers(
        self,
        learning_rate: float,
        weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.95),
    ) -> Tuple[torch.optim.Optimizer, Dict[str, Dict]]:
        """
        Configure optimizer with u-mup learning rate and weight decay scaling.

        u-mup scaling rules:
        - LR: base_lr * (base_width / hidden_dim)
        - WD: base_wd * (hidden_dim / base_width_wd)
        - No decay for bias, norm, and embedding layers

        Args:
            learning_rate: Base learning rate
            weight_decay: Base weight decay
            betas: Adam betas

        Returns:
            optimizer: Configured AdamW optimizer
            settings: Dict of per-parameter settings for logging
        """
        no_decay_names = ["bias", "norm", "token_emb"]

        param_groups = defaultdict(
            lambda: {"params": [], "weight_decay": None, "lr": None}
        )
        final_settings = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Determine LR and WD based on parameter type
            if any(nd in name for nd in no_decay_names):
                # No decay params: reduced LR, no WD
                lr_value = learning_rate * 0.1
                wd_value = 0.0
            elif "lm_head" in name:
                # Output layer: base LR (it's initialized to zero anyway)
                lr_value = learning_rate
                wd_value = weight_decay
            else:
                # Hidden layers: u-mup scaling
                hidden_dim = param.shape[-1] if param.dim() >= 2 else self.config.d_model

                # LR scales inversely with width
                lr_value = learning_rate * (self.config.base_width / hidden_dim)

                # WD scales with width
                wd_value = weight_decay * (hidden_dim / self.config.base_width_wd)

            # Group parameters by (lr, wd) for efficiency
            group_key = (lr_value, wd_value)
            param_groups[group_key]["params"].append(param)
            param_groups[group_key]["weight_decay"] = wd_value
            param_groups[group_key]["lr"] = lr_value

            final_settings[name] = {
                "lr": lr_value,
                "wd": wd_value,
                "shape": list(param.shape),
            }

        optimizer_groups = list(param_groups.values())
        optimizer = torch.optim.AdamW(optimizer_groups, betas=betas)

        return optimizer, final_settings

    def count_parameters(self, non_embedding: bool = True) -> int:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            emb_params = self.token_emb.weight.numel()
            return total - emb_params
        return total

    def num_params(self) -> int:
        """Return total number of parameters (flops_fit model contract).

        This is the primary method checked by model_factory.validate_model_contract().
        Must return a positive integer equal to total trainable parameter count.
        """
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(
        cls,
        d_model: int,
        num_layers: int,
        num_heads: int,
        vocab_size: int = 50257,
        max_seq_len: int = 256,
        **kwargs,
    ) -> "GPT":
        """Create a GPT model from basic config parameters."""
        config = GPTConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            **kwargs,
        )
        return cls(config)


def estimate_model_flops(
    num_params: int,
    num_tokens: int,
    seq_len: int = 256,
) -> int:
    """
    Estimate training FLOPs for a transformer model.

    Uses the standard approximation: C ≈ 6 * N * D
    where N = parameters, D = tokens.

    This accounts for forward + backward pass (3x forward)
    and the ~2 FLOPs per parameter per token.
    """
    return 6 * num_params * num_tokens


def estimate_params_from_config(
    d_model: int,
    num_layers: int,
    vocab_size: int = 50257,
    d_ff_ratio: int = 4,
) -> Dict[str, int]:
    """
    Estimate parameter count for a GPT model.

    Components:
    - Embedding: vocab_size * d_model
    - Attention (per layer): 4 * d_model^2 (QKV + proj)
    - FFN (per layer): 3 * d_model * d_ff (w1, w2, w3 for SwiGLU)
    - Norm (per layer): 2 * d_model
    - Final norm: d_model
    - LM head: vocab_size * d_model (often tied with embedding)
    """
    d_ff = d_model * d_ff_ratio

    embedding = vocab_size * d_model
    attention_per_layer = 4 * d_model * d_model  # QKV + proj
    ffn_per_layer = 3 * d_model * d_ff  # SwiGLU has 3 weight matrices
    norm_per_layer = 2 * d_model  # 2 RMSNorm per block

    per_layer = attention_per_layer + ffn_per_layer + norm_per_layer
    total_layers = num_layers * per_layer

    final_norm = d_model
    lm_head = vocab_size * d_model

    # Non-embedding parameters (what Chinchilla uses)
    non_embedding = total_layers + final_norm

    # Total with embedding (no weight tying)
    total = embedding + total_layers + final_norm + lm_head

    return {
        "total": total,
        "non_embedding": non_embedding,
        "embedding": embedding,
        "per_layer": per_layer,
        "lm_head": lm_head,
    }


# Convenience function for scaling experiments
def create_model_for_scaling(
    target_params: int,
    num_layers: int = 6,
    vocab_size: int = 50257,
    max_seq_len: int = 256,
    parametrization: str = "u-mup",
) -> GPT:
    """
    Create a model targeting a specific parameter count.

    Solves for d_model given target non-embedding params and num_layers.
    Uses: N_non_emb ≈ 12 * num_layers * d_model^2 (approximate)
    """
    # Approximate: N ≈ 12 * L * d^2
    d_model_approx = int(math.sqrt(target_params / (12 * num_layers)))

    # Round to nearest multiple of 64 for efficiency
    d_model = max(64, (d_model_approx // 64) * 64)

    # Ensure d_model is divisible by common head counts
    num_heads = max(2, d_model // 64)
    while d_model % num_heads != 0 and num_heads > 1:
        num_heads -= 1

    config = GPTConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        parametrization=parametrization,
    )

    return GPT(config)
