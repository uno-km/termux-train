"""
termux_train.rl.grpo
====================
Group Relative Policy Optimization (GRPO) Engine (DeepSeek-R1 Style).
Eliminates the separate Critic/Value neural network entirely by calculating relative
advantages across groups of generated outputs per prompt.
Minimizes VRAM requirements to allow on-device Reinforcement Learning on Termux.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Dict, Any, Optional, Union
from ..tensor import Tensor, zeros, ones


class GRPOLoss:
    """
    Group Relative Policy Optimization (GRPO) Loss:
      For each question/prompt q, a group of G responses {o_1, ..., o_G} is evaluated.
      Normalized group advantage: A_i = (r_i - mean(r)) / (std(r) + eps)
      Surrogate policy loss with clipping + KL divergence penalty to reference model.

    Args:
        epsilon: PPO-style clipping threshold (default: 0.2).
        beta: KL divergence penalty weight against frozen reference policy (default: 0.04).
        group_size: Number of candidate outputs sampled per prompt (default: 4).
    """

    def __init__(
        self,
        epsilon: float = 0.2,
        beta: float = 0.04,
        group_size: int = 4
    ) -> None:
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if isinstance(beta, bool) or not isinstance(beta, (int, float)) or beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 2:
            raise ValueError(f"group_size must be >= 2, got {group_size}")

        self.epsilon = float(epsilon)
        self.beta = float(beta)
        self.group_size = int(group_size)

    def compute_group_advantages(self, rewards: Union[List[float], Tensor]) -> Tensor:
        """
        Computes mean-centered and variance-normalized advantages across the group:
          A_i = (r_i - mean(r)) / (std(r) + 1e-8)
        """
        if isinstance(rewards, list):
            r_vals = [float(x) for x in rewards]
            n = len(r_vals)
            if n < 2:
                return Tensor([0.0] * n, dtype="float32")
            mean_r = sum(r_vals) / n
            var_r = sum((x - mean_r) ** 2 for x in r_vals) / n
            std_r = math.sqrt(var_r) + 1e-8
            advs = [(x - mean_r) / std_r for x in r_vals]
            return Tensor(advs, dtype="float32")

        # Tensor input
        mean_r = rewards.mean()
        diff = rewards - mean_r
        var_r = (diff ** 2).mean()
        std_r = (var_r + 1e-8).sqrt()
        return diff / std_r

    def __call__(
        self,
        log_probs: Tensor,
        old_log_probs: Tensor,
        rewards: Union[List[float], Tensor],
        ref_log_probs: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Computes GRPO loss:
          L_GRPO = - (1/G) * sum( min( ratio * A, clip(ratio, 1-eps, 1+eps) * A ) ) + beta * D_KL

        Args:
            log_probs: Log probabilities under current policy pi_theta (shape: (G,)).
            old_log_probs: Log probabilities under previous policy pi_old (shape: (G,)).
            rewards: Scalar reward scores for each sample in the group (len G).
            ref_log_probs: Optional log probabilities under reference policy pi_ref (shape: (G,)).
        """
        advantages = self.compute_group_advantages(rewards).detach()

        # Ratio = exp(log_prob - old_log_prob)
        log_ratio = log_probs - old_log_probs
        ratio = log_ratio.exp()

        surr1 = ratio * advantages
        clipped_ratio = ratio.clip(min_val=1.0 - self.epsilon, max_val=1.0 + self.epsilon)
        surr2 = clipped_ratio * advantages

        # Elementwise minimum between surr1 and surr2:
        # min(a, b) = 0.5 * (a + b - |a - b|)
        diff = surr1 - surr2
        # smooth approximation or exact min via sign
        # Here: surr1 and surr2 only differ when ratio is clipped.
        # When advantage > 0: ratio is clipped from above -> surr2 <= surr1 -> min is surr2
        # When advantage < 0: ratio is clipped from below -> surr2 >= surr1 -> min is surr1
        # Thus min(surr1, surr2) is directly computable without branching:
        surr_min = (surr1 + surr2 - (diff ** 2 + 1e-12).sqrt()) * 0.5

        policy_loss = -(surr_min.mean())

        kl_loss_val = 0.0
        total_loss = policy_loss

        if ref_log_probs is not None and self.beta > 0.0:
            # KL approximation: exp(ref_logp - logp) - (ref_logp - logp) - 1.0
            # or D_KL approx = log_prob - ref_logp
            kl_div = log_probs - ref_log_probs
            kl_penalty = (kl_div.exp() - kl_div - 1.0).mean()
            kl_loss = kl_penalty * self.beta
            total_loss = total_loss + kl_loss
            kl_loss_val = float(kl_loss.item())

        metrics = {
            "total_loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "kl_loss": kl_loss_val,
            "mean_ratio": float(ratio.mean().item()),
            "mean_advantage": float(advantages.mean().item()),
        }

        return total_loss, metrics
