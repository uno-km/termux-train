"""
tests/test_optim.py
===================
Comprehensive unit, numerical, and contract hardening test suite for termux_train.optim.
Tests:
  - Base validation (empty, duplicate, zero-size, real-number, bool, NaN/Inf)
  - Parameter shape lock & post-construction mutation detection
  - Fail-fast finite checks (NaN, +Inf, -Inf on params, grads, states, updates)
  - Numeric updates for SGD (Momentum, Nesterov, Weight Decay), Adam, AdamW
  - Decoupled weight decay distinction
  - Atomic load_state_dict rollback, schema enforcement, and round-trip
  - Parameter identity, DAG, and backend preservation
  - __repr__ format
"""

import copy
import math
import pytest
from termux_train import Tensor, available_backends, set_backend
from termux_train.nn.parameter import Parameter
from termux_train.optim import Optimizer, SGD, Adam, AdamW

@pytest.fixture(params=["python"] + (["numpy"] if "numpy" in available_backends() else []))
def active_backend(request):
    set_backend(request.param)
    return request.param


# =============================================================================
# 1. Base Validation & Hyperparameter Type Enforcement
# =============================================================================

def test_optimizer_base_validation(active_backend):
    # Empty params
    with pytest.raises(ValueError, match="empty parameter list"):
        _ = SGD([], lr=0.1)

    # Invalid param types
    with pytest.raises(TypeError, match="optimizer params must be Tensor"):
        _ = SGD(["not_a_param"], lr=0.1)

    # Duplicate params
    p = Parameter([1.0, 2.0])
    with pytest.raises(ValueError, match="duplicate parameter"):
        _ = SGD([p, p], lr=0.1)

    # Zero-size parameter rejection
    p_zero = Parameter([])
    with pytest.raises(ValueError, match="zero-size parameter"):
        _ = SGD([p_zero], lr=0.1)

    # Defaults not dict
    with pytest.raises(TypeError, match="defaults must be a dict"):
        _ = Optimizer([p], defaults="invalid")


def test_hyperparameter_strict_real_and_bool_rejection(active_backend):
    p = Parameter([1.0, 2.0])

    # Bool rejection (since bool is a subclass of int in Python)
    with pytest.raises(TypeError, match="real number, not bool"):
        _ = SGD([p], lr=True)
    with pytest.raises(TypeError, match="real number, not bool"):
        _ = SGD([p], lr=0.1, momentum=False)
    with pytest.raises(TypeError, match="real number, not bool"):
        _ = Adam([p], lr=0.1, eps=True)
    with pytest.raises(TypeError, match="real number, not bool"):
        _ = Adam([p], lr=0.1, betas=(True, 0.999))

    # NaN / Inf rejection
    with pytest.raises(ValueError, match="must be finite"):
        _ = SGD([p], lr=float("nan"))
    with pytest.raises(ValueError, match="must be finite"):
        _ = SGD([p], lr=float("inf"))
    with pytest.raises(ValueError, match="must be finite"):
        _ = Adam([p], lr=0.1, eps=float("inf"))
    with pytest.raises(ValueError, match="must be finite"):
        _ = Adam([p], lr=0.1, betas=(0.9, float("nan")))

    # Range rejection
    with pytest.raises(ValueError, match="must be > 0"):
        _ = SGD([p], lr=-0.1)
    with pytest.raises(ValueError, match="must be > 0"):
        _ = SGD([p], lr=0.0)
    with pytest.raises(ValueError, match="must be >= 0"):
        _ = SGD([p], lr=0.1, momentum=-0.5)
    with pytest.raises(ValueError, match="must be in"):
        _ = Adam([p], lr=0.1, betas=(1.0, 0.999))
    with pytest.raises(ValueError):
        _ = Adam([p], lr=0.1, betas=(-0.1, 0.999))


def test_parameter_shape_lock_after_construction(active_backend):
    p = Parameter([1.0, 2.0], requires_grad=True)
    opt = SGD([p], lr=0.1)

    # Mutate shape behind the optimizer's back
    p._data = p.backend.from_data([1.0, 2.0, 3.0])
    p.grad = Tensor([0.1, 0.2, 0.3])

    with pytest.raises(RuntimeError, match="parameter shape changed after optimizer construction"):
        opt.step()


# =============================================================================
# 2. Fail-Fast Non-Finite (NaN / Inf) Detection
# =============================================================================

def test_fail_fast_nan_inf_in_param_and_grad(active_backend):
    p = Parameter([1.0, 2.0], requires_grad=True)
    opt = SGD([p], lr=0.1)

    # 1. NaN gradient
    p.grad = Tensor([float("nan"), 0.0])
    with pytest.raises(FloatingPointError, match="non-finite value"):
        opt.step()

    # 2. +Inf gradient
    p.grad = Tensor([float("inf"), 0.0])
    with pytest.raises(FloatingPointError, match="non-finite value"):
        opt.step()

    # 3. -Inf gradient
    p.grad = Tensor([float("-inf"), 0.0])
    with pytest.raises(FloatingPointError, match="non-finite value"):
        opt.step()

    # 4. NaN parameter
    p._data = p.backend.from_data([float("nan"), 2.0])
    p.grad = Tensor([1.0, 1.0])
    with pytest.raises(FloatingPointError, match="non-finite value"):
        opt.step()


def test_fail_fast_nan_inf_in_adam_states(active_backend):
    p = Parameter([1.0], requires_grad=True)
    opt = Adam([p], lr=0.1)
    p.grad = Tensor([1.0])
    opt.step()

    # Inject NaN into exp_avg
    opt.state[0]["exp_avg"] = p.backend.from_data([float("nan")])
    p.grad = Tensor([1.0])
    with pytest.raises(FloatingPointError, match="non-finite value"):
        opt.step()

    # Inject Inf into exp_avg_sq
    opt.state[0]["exp_avg"] = p.backend.from_data([1.0])
    opt.state[0]["exp_avg_sq"] = p.backend.from_data([float("inf")])
    p.grad = Tensor([1.0])
    with pytest.raises(FloatingPointError, match="non-finite value"):
        opt.step()


# =============================================================================
# 3. Lifecycle, DAG-Safety, and Parameter Identity
# =============================================================================

def test_optimizer_zero_grad_lifecycle(active_backend):
    p1 = Parameter([1.0, 2.0], requires_grad=True)
    p2 = Parameter([3.0, 4.0], requires_grad=True)
    opt = SGD([p1, p2], lr=0.1)

    # Simulate gradients
    p1.grad = Tensor([0.1, 0.2])
    p2.grad = Tensor([0.3, 0.4])

    opt.zero_grad(set_to_none=True)
    assert p1.grad is None
    assert p2.grad is None

    p1.grad = Tensor([0.1, 0.2])
    p2.grad = Tensor([0.3, 0.4])
    opt.zero_grad(set_to_none=False)
    assert p1.grad is not None
    assert p1.grad.tolist() == [0.0, 0.0]
    assert p2.grad is not None
    assert p2.grad.tolist() == [0.0, 0.0]


def test_parameter_identity_and_dag_preservation(active_backend):
    p = Parameter([1.0, 2.0], requires_grad=True)
    opt = SGD([p], lr=0.1)

    before_id = id(p)
    before_prev = p._prev
    before_backward = p._backward
    before_op = p._op
    before_backend = p.backend

    p.grad = Tensor([0.5, 0.5])
    opt.step()

    assert id(p) == before_id
    assert p._prev is before_prev
    assert p._backward is before_backward
    assert p._op == before_op
    assert p.backend is before_backend


def test_optimizer_step_skips_and_errors(active_backend):
    p_none_grad = Parameter([1.0, 2.0], requires_grad=True)
    p_no_grad = Parameter([3.0, 4.0], requires_grad=False)
    p_active = Parameter([5.0, 6.0], requires_grad=True)

    opt = SGD([p_none_grad, p_no_grad, p_active], lr=0.1)
    p_active.grad = Tensor([1.0, 1.0])

    opt.step()
    assert p_none_grad.tolist() == [1.0, 2.0]
    assert p_no_grad.tolist() == [3.0, 4.0]
    assert p_active.tolist() == pytest.approx([4.9, 5.9], abs=1e-5)

    # Shape mismatch error
    p_err = Parameter([1.0, 2.0], requires_grad=True)
    opt_err = SGD([p_err], lr=0.1)
    p_err.grad = Tensor([1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError, match="gradient shape mismatch"):
        opt_err.step()


# =============================================================================
# 4. Numerical Updates for SGD, Adam, AdamW
# =============================================================================

def test_sgd_basic_update(active_backend):
    p = Parameter([1.0, -2.0], requires_grad=True)
    opt = SGD([p], lr=0.1)

    p.grad = Tensor([0.5, -0.25])
    opt.step()

    assert p.tolist() == pytest.approx([0.95, -1.975], abs=1e-6)


def test_sgd_weight_decay(active_backend):
    p = Parameter([1.0, -2.0], requires_grad=True)
    opt = SGD([p], lr=0.1, weight_decay=0.01)

    p.grad = Tensor([0.5, -0.25])
    opt.step()

    assert p.tolist() == pytest.approx([0.949, -1.973], abs=1e-6)


def test_sgd_momentum_2_steps(active_backend):
    p = Parameter([1.0, -2.0], requires_grad=True)
    opt = SGD([p], lr=0.1, momentum=0.9, dampening=0.0)

    # Step 1: v_1 = g_1 = [0.5, -0.25]
    p.grad = Tensor([0.5, -0.25])
    opt.step()
    assert p.tolist() == pytest.approx([0.95, -1.975], abs=1e-6)
    assert p.backend.to_flat_list(opt.state[0]["momentum_buffer"]) == pytest.approx([0.5, -0.25], abs=1e-6)

    # Step 2: g_2 = [0.2, 0.1]
    p.grad = Tensor([0.2, 0.1])
    opt.step()
    assert p.tolist() == pytest.approx([0.885, -1.9625], abs=1e-6)
    assert p.backend.to_flat_list(opt.state[0]["momentum_buffer"]) == pytest.approx([0.65, -0.125], abs=1e-6)


def test_sgd_nesterov(active_backend):
    p = Parameter([1.0, -2.0], requires_grad=True)
    opt = SGD([p], lr=0.1, momentum=0.9, nesterov=True)

    # Step 1
    p.grad = Tensor([0.5, -0.25])
    opt.step()
    assert p.tolist() == pytest.approx([0.905, -1.9525], abs=1e-6)

    # Step 2
    p.grad = Tensor([0.2, 0.1])
    opt.step()
    assert p.tolist() == pytest.approx([0.8265, -1.95125], abs=1e-6)


def test_adam_first_and_second_step(active_backend):
    p = Parameter([1.0], requires_grad=True)
    opt = Adam([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    # Step 1
    p.grad = Tensor([0.5])
    opt.step()
    assert p.tolist() == pytest.approx([0.9], abs=1e-5)
    assert opt.state[0]["step"] == 1

    # Step 2
    p.grad = Tensor([0.5])
    opt.step()
    assert p.tolist() == pytest.approx([0.8], abs=1e-5)
    assert opt.state[0]["step"] == 2


def test_adamw_vs_adam_weight_decay_distinction(active_backend):
    p_adam = Parameter([1.0, -2.0], requires_grad=True)
    p_adamw = Parameter([1.0, -2.0], requires_grad=True)

    opt_adam = Adam([p_adam], lr=0.1, weight_decay=0.05)
    opt_adamw = AdamW([p_adamw], lr=0.1, weight_decay=0.05)

    p_adam.grad = Tensor([0.5, 0.5])
    p_adamw.grad = Tensor([0.5, 0.5])

    opt_adam.step()
    opt_adamw.step()

    assert p_adam.tolist() != p_adamw.tolist()


def test_adamw_zero_gradient_decay(active_backend):
    p = Parameter([10.0, -20.0], requires_grad=True)
    lr = 0.1
    wd = 0.05
    opt = AdamW([p], lr=lr, weight_decay=wd)

    p.grad = Tensor([0.0, 0.0])
    opt.step()

    assert p.tolist() == pytest.approx([9.95, -19.9], abs=1e-6)


# =============================================================================
# 5. Atomic State Dict & Strict Schema Validation
# =============================================================================

@pytest.mark.parametrize("OptimizerClass", [SGD, Adam, AdamW])
def test_optimizer_state_dict_round_trip(OptimizerClass, active_backend):
    p1 = Parameter([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    p2 = Parameter([5.0, 6.0], requires_grad=True)
    opt1 = OptimizerClass([p1, p2], lr=0.05)

    # Perform 2 optimization steps
    for _ in range(2):
        p1.grad = Tensor([[0.1, 0.2], [0.3, 0.4]])
        p2.grad = Tensor([0.5, 0.6])
        opt1.step()

    # Save state
    saved_state = opt1.state_dict()

    # Deep-copy isolation test
    saved_state["defaults"]["lr"] = 999.0
    assert opt1.defaults["lr"] == 0.05

    # Restore into fresh optimizer
    p1_fresh = Parameter([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    p2_fresh = Parameter([5.0, 6.0], requires_grad=True)
    opt2 = OptimizerClass([p1_fresh, p2_fresh], lr=0.05)

    for _ in range(2):
        p1_fresh.grad = Tensor([[0.1, 0.2], [0.3, 0.4]])
        p2_fresh.grad = Tensor([0.5, 0.6])
        opt2.step()

    saved_state["defaults"]["lr"] = 0.05
    opt2.load_state_dict(saved_state)

    # Perform step 3 on both
    p1.grad = Tensor([[0.05, 0.05], [0.05, 0.05]])
    p2.grad = Tensor([0.1, 0.1])
    opt1.step()

    p1_fresh.grad = Tensor([[0.05, 0.05], [0.05, 0.05]])
    p2_fresh.grad = Tensor([0.1, 0.1])
    opt2.step()

    assert p1.flatten().tolist() == pytest.approx(p1_fresh.flatten().tolist(), abs=1e-5)
    assert p2.flatten().tolist() == pytest.approx(p2_fresh.flatten().tolist(), abs=1e-5)


def test_atomic_load_state_dict_rollback_on_failure(active_backend):
    p = Parameter([1.0, 2.0], requires_grad=True)
    opt = Adam([p], lr=0.01)
    p.grad = Tensor([0.1, 0.2])
    opt.step()

    orig_defaults = copy.deepcopy(opt.defaults)
    orig_step = opt.state[0]["step"]

    # Attempt to load corrupted state dict (shape mismatch in exp_avg)
    corrupted_state = {
        "class": "Adam",
        "defaults": {"lr": 0.5, "betas": (0.8, 0.95), "eps": 1e-6, "weight_decay": 0.0},
        "param_count": 1,
        "state": {
            0: {
                "step": 5,
                "exp_avg": [1.0, 2.0, 3.0], # Mismatched shape (3,) != (2,)
                "exp_avg_sq": [0.1, 0.2],
            }
        }
    }

    with pytest.raises(RuntimeError, match="Shape mismatch"):
        opt.load_state_dict(corrupted_state)

    # Verify atomic rollback: internal defaults and state remain completely untouched
    assert opt.defaults["lr"] == orig_defaults["lr"]
    assert opt.state[0]["step"] == orig_step


def test_load_state_dict_schema_rejections(active_backend):
    p = Parameter([1.0, 2.0], requires_grad=True)
    opt = Adam([p], lr=0.01)

    # 1. Invalid class
    with pytest.raises(ValueError, match="class mismatch"):
        opt.load_state_dict({"class": "SGD", "param_count": 1, "defaults": {"lr": 0.01}, "state": {}})

    # 2. Invalid param count
    with pytest.raises(ValueError, match="Parameter count mismatch"):
        opt.load_state_dict({"class": "Adam", "param_count": 2, "defaults": {"lr": 0.01, "eps": 1e-8}, "state": {}})

    # 3. Invalid defaults (e.g. lr <= 0)
    with pytest.raises(ValueError, match="must be > 0"):
        opt.load_state_dict({"class": "Adam", "param_count": 1, "defaults": {"lr": -0.5, "eps": 1e-8}, "state": {}})

    # 4. Invalid state schema (missing step)
    with pytest.raises(ValueError, match="missing 'step'"):
        opt.load_state_dict({
            "class": "Adam",
            "param_count": 1,
            "defaults": {"lr": 0.01, "eps": 1e-8, "betas": (0.9, 0.999), "weight_decay": 0.0},
            "state": {0: {"exp_avg": [0.0, 0.0], "exp_avg_sq": [0.0, 0.0]}}
        })

    # 5. Invalid step (bool or negative)
    with pytest.raises(ValueError, match="non-negative integer"):
        opt.load_state_dict({
            "class": "Adam",
            "param_count": 1,
            "defaults": {"lr": 0.01, "eps": 1e-8, "betas": (0.9, 0.999), "weight_decay": 0.0},
            "state": {0: {"step": -1, "exp_avg": [0.0, 0.0], "exp_avg_sq": [0.0, 0.0]}}
        })


# =============================================================================
# 6. Representation (__repr__)
# =============================================================================

def test_optimizer_repr(active_backend):
    p = Parameter([1.0, 2.0])
    opt_sgd = SGD([p], lr=0.1, momentum=0.9)
    opt_adam = Adam([p], lr=0.001)
    opt_adamw = AdamW([p], lr=0.001, weight_decay=0.01)

    assert "SGD" in repr(opt_sgd)
    assert "lr=0.1" in repr(opt_sgd)
    assert "momentum=0.9" in repr(opt_sgd)
    assert "params=1" in repr(opt_sgd)

    assert "Adam" in repr(opt_adam)
    assert "lr=0.001" in repr(opt_adam)
    assert "params=1" in repr(opt_adam)

    assert "AdamW" in repr(opt_adamw)
    assert "weight_decay=0.01" in repr(opt_adamw)
    assert "params=1" in repr(opt_adamw)


def test_sgd_strict_momentum_schema_rejection(active_backend):
    p = Parameter([1.0, 2.0], requires_grad=True)
    opt = SGD([p], lr=0.01, momentum=0.9)

    # When momentum > 0, state entry without momentum_buffer must be rejected
    with pytest.raises(ValueError, match="missing 'momentum_buffer'"):
        opt.load_state_dict({
            "class": "SGD",
            "param_count": 1,
            "defaults": {"lr": 0.01, "momentum": 0.9, "dampening": 0.0, "weight_decay": 0.0, "nesterov": False},
            "state": {0: {}}
        })


@pytest.mark.parametrize("OptimizerClass", [SGD, Adam, AdamW])
def test_optimizer_full_step_transactional_atomicity_policy_b(OptimizerClass, active_backend):
    p1 = Parameter([1.0, 2.0], requires_grad=True)
    p2 = Parameter([3.0, 4.0], requires_grad=True)
    opt = OptimizerClass([p1, p2], lr=0.1)

    # p1 has valid grad, p2 has NaN grad
    p1.grad = Tensor([0.5, 0.5])
    p2.grad = Tensor([float("nan"), 0.5])

    with pytest.raises(FloatingPointError, match="non-finite value"):
        opt.step()

    # Policy B Guarantee: Full step transaction aborted, p1 must be 100% UNTOUCHED
    assert p1.tolist() == [1.0, 2.0]
    assert p2.tolist() == [3.0, 4.0]
    assert len(opt.state) == 0
