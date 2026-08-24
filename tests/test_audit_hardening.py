"""
tests/test_audit_hardening.py
=============================
Test suite validating the resolution of all critical audit vulnerabilities:
  1. Explicit Dtype Promotion rules across all arithmetic operators
  2. In-place modification detection (_version tracking) during autograd backward
  3. IEEE 754 compliance in div and log (no silent magic number clamping)
  4. 2-Phase Transactional Module.load_state_dict with atomic rollback
  5. Atomic Optimizer._commit_step transaction safety
"""

import math
import pytest
from termux_train import Tensor, nn, optim, set_backend, available_backends


# =============================================================================
# 1. Explicit Dtype Promotion Rules
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_explicit_dtype_promotion_arithmetic(backend_name):
    set_backend(backend_name)

    t_f32 = Tensor([1.0, 2.0], dtype="float32")
    t_i64 = Tensor([3, 4], dtype="int64")
    t_bool = Tensor([True, False], dtype="bool")

    # float32 + int64 -> float32
    add_fi = t_f32 + t_i64
    assert add_fi.dtype == "float32"

    # int64 + int64 -> int64
    add_ii = t_i64 + t_i64
    assert add_ii.dtype == "int64"

    # int64 + bool -> int64
    add_ib = t_i64 + t_bool
    assert add_ib.dtype == "int64"

    # bool + bool -> int64 (additive algebraic ring)
    add_bb = t_bool + t_bool
    assert add_bb.dtype == "int64"

    # True division always produces float32
    div_ii = t_i64 / t_i64
    assert div_ii.dtype == "float32"

    # Mathematical functions always produce float32
    assert t_i64.exp().dtype == "float32"
    assert t_i64.sqrt().dtype == "float32"
    assert t_i64.log().dtype == "float32"
    assert t_i64.sigmoid().dtype == "float32"
    assert t_i64.tanh().dtype == "float32"


# =============================================================================
# 2. In-place Mutation Detection (_version Tracking)
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_inplace_mutation_autograd_guard(backend_name):
    set_backend(backend_name)

    x = Tensor([2.0, 3.0], requires_grad=True)
    y = x * x
    loss = y.sum()

    # Modify x in-place after forward pass but before backward
    x.data = [10.0, 20.0]

    # backward must fail fast with RuntimeError
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


# =============================================================================
# 3. IEEE 754 Division & Log (No Silent Magic Numbers)
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_ieee754_division_and_log_behavior(backend_name):
    set_backend(backend_name)

    # 1. Division by zero
    num = Tensor([1.0, -2.0, 0.0])
    denom = Tensor([0.0, 0.0, 0.0])
    res_div = num / denom

    div_vals = res_div.tolist()
    assert div_vals[0] == float('inf') or math.isinf(div_vals[0])
    assert div_vals[1] == float('-inf') or math.isinf(div_vals[1])
    assert math.isnan(div_vals[2])

    # 2. Natural log of zero and negative numbers
    x_log = Tensor([0.0, -5.0])
    res_log = x_log.log()

    log_vals = res_log.tolist()
    assert log_vals[0] == float('-inf') or math.isinf(log_vals[0])
    assert math.isnan(log_vals[1])


# =============================================================================
# 4. 2-Phase Transactional Module.load_state_dict
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_module_load_state_dict_atomic_rollback(backend_name):
    set_backend(backend_name)

    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.Linear(4, 1)
    )

    initial_w0 = model[0].weight.tolist()
    initial_w1 = model[1].weight.tolist()

    # Create an invalid state dict where first layer matches but second layer has shape mismatch
    corrupted_state = {
        "0.weight": [[9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0]],
        "0.bias": [9.0, 9.0, 9.0, 9.0],
        "1.weight": [[9.0, 9.0]], # INVALID SHAPE: expected (4, 1), got (1, 2)
        "1.bias": [9.0]
    }

    # load_state_dict must fail
    with pytest.raises(RuntimeError, match="Shape mismatch"):
        model.load_state_dict(corrupted_state)

    # All parameters must remain 100% untouched (atomic rollback)
    assert model[0].weight.tolist() == initial_w0
    assert model[1].weight.tolist() == initial_w1


# =============================================================================
# 5. Atomic Optimizer Step Transaction Safety
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_optimizer_step_atomic_rollback_on_error(backend_name):
    set_backend(backend_name)

    w = nn.Parameter(Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
    optimizer = optim.SGD([w], lr=0.1)

    initial_w = w.tolist()

    # Step with non-finite gradient should raise FloatingPointError without modifying w
    w.grad = Tensor([[float('nan'), 1.0], [0.0, 0.0]])

    with pytest.raises(FloatingPointError):
        optimizer.step()

    # Parameter must remain identical to initial state
    assert w.tolist() == initial_w
