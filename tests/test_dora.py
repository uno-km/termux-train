"""
tests/test_dora.py
==================
Unit and Integration Tests for DoRALinear (Weight-Decomposed Low-Rank Adaptation).
Verifies initialization, gradient flow, numerical stability, and merge/unmerge transactions.
"""

import math
import pytest
from termux_train import Tensor, randn
from termux_train.nn.linear import Linear
from termux_train.nn.dora import DoRALinear
from termux_train.nn.loss import MSELoss
from termux_train.optim.sgd import SGD


def test_dora_initialization():
    in_dim, out_dim, rank = 16, 32, 4
    dora = DoRALinear(in_features=in_dim, out_features=out_dim, rank=rank, alpha=8.0)

    assert dora.in_features == in_dim
    assert dora.out_features == out_dim
    assert dora.rank == rank
    assert dora.alpha == 8.0
    assert dora.scaling == 2.0
    assert not dora.merged

    # Parameters
    assert dora.lora_A.shape == (in_dim, rank)
    assert dora.lora_B.shape == (rank, out_dim)
    assert dora.magnitude.shape == (1, out_dim)
    assert dora.base.weight.requires_grad is False
    assert dora.lora_A.requires_grad is True
    assert dora.lora_B.requires_grad is True
    assert dora.magnitude.requires_grad is True


def test_dora_from_linear():
    base = Linear(8, 12, bias=True)
    dora = DoRALinear.from_linear(base, rank=2, alpha=4.0)

    assert dora.in_features == 8
    assert dora.out_features == 12
    assert dora.rank == 2
    assert dora.base is base
    assert not base.weight.requires_grad


def test_dora_forward_and_backward():
    in_dim, out_dim, rank = 8, 8, 2
    dora = DoRALinear(in_features=in_dim, out_features=out_dim, rank=rank, alpha=2.0)
    optimizer = SGD(dora.adapter_parameters(), lr=0.01)
    loss_fn = MSELoss()

    x = randn((4, in_dim))
    target = randn((4, out_dim))

    # Initial forward
    out = dora(x)
    assert out.shape == (4, out_dim)

    loss = loss_fn(out, target)
    loss.backward()

    # Gradients must be present in LoRA A, B, magnitude
    assert dora.lora_A.grad is not None
    assert dora.lora_B.grad is not None
    assert dora.magnitude.grad is not None
    # Base weight must NOT receive gradients
    assert dora.base.weight.grad is None

    # Step optimizer
    optimizer.step()


def test_dora_merge_unmerge_transaction():
    in_dim, out_dim = 6, 6
    dora = DoRALinear(in_dim, out_dim, rank=2, alpha=2.0)
    x = randn((2, in_dim))

    # Pre-merge output
    out_unmerged = dora(x)

    # Merge
    dora.merge()
    assert dora.merged is True
    out_merged = dora(x)

    # Outputs must match closely
    flat_unm = out_unmerged.tolist()
    flat_m = out_merged.tolist()
    for row_u, row_m in zip(flat_unm, flat_m):
        for u, m in zip(row_u, row_m):
            assert abs(u - m) < 1e-4

    # Second merge must raise RuntimeError
    with pytest.raises(RuntimeError):
        dora.merge()

    # Unmerge
    dora.unmerge()
    assert dora.merged is False
    out_restored = dora(x)
    flat_res = out_restored.tolist()
    for row_u, row_r in zip(flat_unm, flat_res):
        for u, r in zip(row_u, row_r):
            assert abs(u - r) < 1e-4

    # Second unmerge must raise RuntimeError
    with pytest.raises(RuntimeError):
        dora.unmerge()


def test_dora_adapter_state_dict():
    dora = DoRALinear(4, 8, rank=2, alpha=4.0)
    state = dora.adapter_state_dict()

    assert state["type"] == "dora"
    assert state["rank"] == 2
    assert state["alpha"] == 4.0
    assert len(state["lora_A"]) == 8  # 4 * 2
    assert len(state["lora_B"]) == 16  # 2 * 8
    assert len(state["magnitude"]) == 8  # 1 * 8
