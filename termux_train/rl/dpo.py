"""
termux_train.rl.dpo
===================
Direct Preference Optimization (DPO) Loss & Engine for On-Device Alignment.
Bypasses the complexity and VRAM footprint of explicit reward modeling by directly
optimizing policy parameters from pairwise (chosen vs rejected) preferences.
"""

from __future__ import annotations

import math
from typing import Dict, Any, Tuple, Optional
from ..tensor import Tensor, zeros, ones


class DPOLoss:
    """
    Direct Preference Optimization Loss:
      L_DPO(pi_theta; pi_ref) = -E[log sigma(beta * log(pi_theta(yw|x)/pi_ref(yw|x)) - beta * log(pi_theta(yl|x)/pi_ref(yl|x)))]

    Args:
        beta: Temperature scaling parameter controlling deviation from reference model (default: 0.1).
        label_smoothing: Optional label smoothing parameter in [0.0, 0.5) (default: 0.0).
    """

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0) -> None:
        if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta <= 0.0:
            raise ValueError(f"beta must be a finite positive number, got {beta}")
        if not (0.0 <= label_smoothing < 0.5):
            raise ValueError(f"label_smoothing must be in [0.0, 0.5), got {label_smoothing}")

        self.beta = float(beta)
        self.label_smoothing = float(label_smoothing)

    def __call__(
        self,
        policy_chosen_logps: Tensor,
        policy_rejected_logps: Tensor,
        ref_chosen_logps: Tensor,
        ref_rejected_logps: Tensor,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Computes DPO loss and returns (loss_tensor, metrics_dict).

        Args:
            policy_chosen_logps: Log probabilities of chosen completions under policy (shape: (batch,)).
            policy_rejected_logps: Log probabilities of rejected completions under policy (shape: (batch,)).
            ref_chosen_logps: Log probabilities of chosen completions under frozen ref policy (shape: (batch,)).
            ref_rejected_logps: Log probabilities of rejected completions under frozen ref policy (shape: (batch,)).
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = (pi_logratios - ref_logratios) * self.beta

        # log_sigmoid(x) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
        # For numerical stability:
        # if x >= 0: -log(1 + exp(-x))
        # if x < 0: x - log(1 + exp(x))
        # Using Tensor.sigmoid().log() or stable elementwise:
        sig = logits.sigmoid()
        # safe clip for log
        sig_safe = sig.clip(min_val=1e-8, max_val=1.0 - 1e-8)
        log_sig = sig_safe.log()

        if self.label_smoothing > 0.0:
            # (1 - s) * (-log(sigmoid(logits))) + s * (-log(sigmoid(-logits)))
            neg_sig = (-logits).sigmoid().clip(min_val=1e-8, max_val=1.0 - 1e-8)
            neg_log_sig = neg_sig.log()
            losses = -(log_sig * (1.0 - self.label_smoothing) + neg_log_sig * self.label_smoothing)
        else:
            losses = -log_sig

        loss = losses.mean()

        # Compute implicit rewards for telemetry (detached)
        chosen_rewards = ((policy_chosen_logps - ref_chosen_logps) * self.beta).detach()
        rejected_rewards = ((policy_rejected_logps - ref_rejected_logps) * self.beta).detach()
        
        c_flat = chosen_rewards.tolist()
        r_flat = rejected_rewards.tolist()
        if not isinstance(c_flat, list):
            c_flat = [c_flat]
        if not isinstance(r_flat, list):
            r_flat = [r_flat]
        acc = sum(1.0 for c, r in zip(c_flat, r_flat) if c > r) / max(len(c_flat), 1)

        metrics = {
            "loss": float(loss.item()),
            "chosen_reward_mean": float(chosen_rewards.mean().item()),
            "rejected_reward_mean": float(rejected_rewards.mean().item()),
            "reward_margin": float((chosen_rewards - rejected_rewards).mean().item()),
            "accuracy": acc,
        }

        return loss, metrics
