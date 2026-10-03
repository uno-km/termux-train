"""
termux_train.adapters
=====================
Target-Aware Adapters & Export Hub for the 6 Termux AI Runtimes:
llamacpp, bitnet, diffusion, vision, tts, stt.
"""

from .hub import TargetEcosystem, export_target_adapter

__all__ = [
    "TargetEcosystem",
    "export_target_adapter",
]
