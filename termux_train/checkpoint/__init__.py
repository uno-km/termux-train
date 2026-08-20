"""
termux_train.checkpoint
=======================
Serialization engines: SafeTensors zero-copy binary format and atomic rollback checkpoints.
"""

from .safetensors import save_safetensors, load_safetensors

__all__ = [
    "save_safetensors",
    "load_safetensors",
]
