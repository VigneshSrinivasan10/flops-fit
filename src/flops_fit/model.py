"""flops-fit Model Implementation - backward compatibility re-export.

The canonical GPT implementation lives in flops_fit.examples.gpt.
This module re-exports all public symbols for backward compatibility.
"""
from flops_fit.examples.gpt import (  # noqa: F401
    GPTConfig,
    RMSNorm,
    RotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
    CausalSelfAttention,
    FeedForward,
    TransformerBlock,
    GPT,
    estimate_model_flops,
    estimate_params_from_config,
    create_model_for_scaling,
)
