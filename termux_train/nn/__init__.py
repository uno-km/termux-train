"""
termux_train.nn
===============
Neural Network layers, containers, modules, activations, losses, embeddings, transformers, utils, RoPE, quantization, and LoRA adapters.
"""

from .module import Module
from .parameter import Parameter
from .linear import Linear
from .embedding import Embedding
from .layernorm import LayerNorm
from .attention import MultiHeadAttention
from .rope import RotaryEmbedding
from .transformer import FeedForward, TransformerBlock, TinyTransformerLM
from .quantization import QuantizedLinear, quantize_linear_int8
from .sequential import Sequential
from .activations import ReLU, Sigmoid, Tanh
from .loss import (
    mse_loss,
    MSELoss,
    bce_loss,
    BCELoss,
    bce_with_logits_loss,
    BCEWithLogitsLoss,
    cross_entropy_loss,
    CrossEntropyLoss,
)
from .lora import (
    LoRALinear,
    adapter_parameters,
    named_adapter_parameters,
    adapter_state_dict,
    load_adapter_state_dict,
    merge_lora_adapters,
    unmerge_lora_adapters,
)
from .utils import clip_grad_norm_

# Alias
binary_cross_entropy_with_logits = bce_with_logits_loss = BCEWithLogitsLoss

__all__ = [
    "Module",
    "Parameter",
    "Linear",
    "Embedding",
    "LayerNorm",
    "MultiHeadAttention",
    "RotaryEmbedding",
    "QuantizedLinear",
    "quantize_linear_int8",
    "FeedForward",
    "TransformerBlock",
    "TinyTransformerLM",
    "Sequential",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "mse_loss",
    "MSELoss",
    "bce_loss",
    "BCELoss",
    "bce_with_logits_loss",
    "binary_cross_entropy_with_logits",
    "BCEWithLogitsLoss",
    "cross_entropy_loss",
    "CrossEntropyLoss",
    "LoRALinear",
    "adapter_parameters",
    "named_adapter_parameters",
    "adapter_state_dict",
    "load_adapter_state_dict",
    "merge_lora_adapters",
    "unmerge_lora_adapters",
    "clip_grad_norm_",
]
