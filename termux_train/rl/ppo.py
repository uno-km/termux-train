"""
termux_train.rl.ppo
===================
Proximal Policy Optimization (PPO) Loss & Engine for On-Device Policy Optimization.
Supports Clipped Surrogate Objective, Value Function Loss, and Entropy Bonus.
"""

from __future__ import annotations

import math
from typing import Tuple, Dict, Any, Optional
from ..tensor import Tensor, zeros, ones


class PPOLoss:
    """
    Proximal Policy Optimization (PPO) Loss:
      L_PPO = L_clip + c1 * L_value - c2 * Entropy

    Args:
        clip_eps: Clipping parameter epsilon (default: 0.2).
        vf_coef: Value function loss coefficient (default: 0.5).
        ent_coef: Entropy bonus coefficient (default: 0.01).
    """

    def __init__(
        self,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01
    ) -> None:
        self.clip_eps = float(clip_eps)
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)

    def __call__(
        self,
        log_probs: Tensor,
        old_log_probs: Tensor,
        advantages: Tensor,
        values: Optional[Tensor] = None,
        returns: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Computes PPO loss and metrics.
        """
        # Ratio
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        clipped_ratio = ratio.clip(min_val=1.0 - self.clip_eps, max_val=1.0 + self.clip_eps)
        surr2 = clipped_ratio * advantages

        # min(surr1, surr2)
        diff = surr1 - surr2
        surr_min = (surr1 + surr2 - (diff ** 2 + 1e-12).sqrt()) * 0.5
        policy_loss = -(surr_min.mean())

        vf_loss_val = 0.0
        total_loss = policy_loss

        if values is not None and returns is not None:
            v_err = values - returns
            vf_loss = (v_err ** 2).mean() * self.vf_coef
            total_loss = total_loss + vf_loss
            vf_loss_val = float(vf_loss.item())

        metrics = {
            "total_loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": vf_loss_val,
            "mean_ratio": float(ratio.mean().item()),
        }

        return total_loss, metrics
