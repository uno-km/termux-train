"""
termux_train.vision
===================
On-device Vision VLM Fine-Tuning and LoRA Adaptation Engine.
Open-Source under Apache License 2.0.
"""

from .vlm_lora import (
    MultimodalProjectorLoRA,
    VLMAttentionLoRA,
    VLMTransformerBlockLoRA,
    VisionLanguageModelLoRA,
)
from .trainer import train_vision_vlm_lora

__all__ = [
    "MultimodalProjectorLoRA",
    "VLMAttentionLoRA",
    "VLMTransformerBlockLoRA",
    "VisionLanguageModelLoRA",
    "train_vision_vlm_lora",
]
