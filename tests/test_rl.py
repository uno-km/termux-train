"""
tests/test_rl.py
================
Unit and Integration Tests for On-Device Reinforcement Learning.
Verifies DPO loss, GRPO group advantage normalization, PPO surrogate, and rule-based reward verifiers.
"""

import math
import pytest
from termux_train import Tensor, randn
from termux_train.rl.reward import xml_format_reward, accuracy_reward, length_penalty_reward, extract_answer_content
from termux_train.rl.dpo import DPOLoss
from termux_train.rl.grpo import GRPOLoss
from termux_train.rl.ppo import PPOLoss


def test_reward_verifiers():
    # XML Format
    valid = "<think>Calculating 2+2</think><answer>4</answer>"
    assert xml_format_reward(valid) == 1.0

    partial = "<think>Only thinking</think>"
    assert xml_format_reward(partial) == 0.5

    invalid = "Just plain text response"
    assert xml_format_reward(invalid) == 0.0

    # Extract Answer
    assert extract_answer_content(valid) == "4"
    assert extract_answer_content(invalid) == invalid

    # Accuracy
    assert accuracy_reward(valid, "4") == 1.0
    assert accuracy_reward(valid, "5") == 0.0
    assert accuracy_reward("Result is 4.0", "4.0") == 1.0

    # Length penalty
    short_text = "one two three"
    assert length_penalty_reward(short_text, max_tokens=10) == 0.0
    long_text = "word " * 50
    assert length_penalty_reward(long_text, max_tokens=20) < 0.0


def test_dpo_loss_computation_and_backward():
    dpo = DPOLoss(beta=0.1)

    batch_size = 4
    pi_w = randn((batch_size,), requires_grad=True)
    pi_l = randn((batch_size,), requires_grad=True)
    ref_w = randn((batch_size,))
    ref_l = randn((batch_size,))

    loss, metrics = dpo(pi_w, pi_l, ref_w, ref_l)

    assert loss.ndim == 0
    assert math.isfinite(loss.item())
    assert "loss" in metrics
    assert "reward_margin" in metrics
    assert "accuracy" in metrics

    loss.backward()
    assert pi_w.grad is not None
    assert pi_l.grad is not None


def test_grpo_advantage_normalization_and_backward():
    grpo = GRPOLoss(epsilon=0.2, beta=0.04, group_size=4)

    rewards = [1.0, 0.0, 0.5, 0.5]
    advs = grpo.compute_group_advantages(rewards)
    assert advs.shape == (4,)
    # Mean of advantages should be approximately 0
    assert abs(advs.mean().item()) < 1e-5

    logps = randn((4,), requires_grad=True)
    old_logps = logps.detach()
    ref_logps = randn((4,))

    loss, metrics = grpo(logps, old_logps, rewards, ref_log_probs=ref_logps)

    assert math.isfinite(loss.item())
    assert "total_loss" in metrics
    assert "policy_loss" in metrics
    assert "kl_loss" in metrics

    loss.backward()
    assert logps.grad is not None


def test_ppo_loss_computation():
    ppo = PPOLoss(clip_eps=0.2)

    logps = randn((4,), requires_grad=True)
    old_logps = logps.detach()
    advs = Tensor([1.0, -1.0, 0.5, -0.5])
    values = randn((4,), requires_grad=True)
    returns = Tensor([1.0, 0.0, 0.5, 0.0])

    loss, metrics = ppo(logps, old_logps, advs, values=values, returns=returns)

    assert math.isfinite(loss.item())
    assert "value_loss" in metrics
    loss.backward()
    assert logps.grad is not None
    assert values.grad is not None
