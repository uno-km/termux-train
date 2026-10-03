"""
termux_train.rl
===============
On-Device Reinforcement Learning & Preference Optimization Package.
Includes Direct Preference Optimization (DPO), Group Relative Policy Optimization (GRPO),
PPO, and rule-based verifiers for self-contained, low-VRAM reasoning and alignment.
"""

from .dpo import DPOLoss
from .grpo import GRPOLoss
from .ppo import PPOLoss
from .reward import xml_format_reward, accuracy_reward, length_penalty_reward, extract_answer_content

__all__ = [
    "DPOLoss",
    "GRPOLoss",
    "PPOLoss",
    "xml_format_reward",
    "accuracy_reward",
    "length_penalty_reward",
    "extract_answer_content",
]
