"""
tests/test_optim.py
===================
Comprehensive unit and numerical verification tests for termux_train.optim suite.
Tests Optimizer Base, SGD (Momentum, Nesterov, Weight Decay), Adam, AdamW, and State Dict.
"""

import copy
import pytest
from termux_train import Tensor, available_backends, set_backend
from termux_train.nn.parameter import Parameter
from termux_train.optim import Optimizer, SGD, Adam, AdamW

@pytest.fixture(params=["python"] + (["numpy"] if "numpy" in available_backends() else []))
def active_backend(request):
    set_backend(request.param)
    return request.param


# =============================================================================
# 1. Optimizer Base Tests
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

    # Defaults not dict
    with pytest.raises(TypeError, match="defaults must be a dict"):
        _ = Optimizer([p], defaults="invalid")


def test_optimizer_zero_grad_lifecycle(active_backend):
    p1 = Parameter([1.0, 2.0], requires_grad=True)
    p2 = Parameter([3.0, 4.0], requires_grad=True)
    opt = SGD([p1, p2], lr=0.1)

    # Simulate gradients
    p1.grad = Tensor([0.1, 0.2])
    p2.grad = Tensor([0.3, 0.4])

    # Default set_to_none=True
    opt.zero_grad(set_to_none=True)
    assert p1.grad is None
    assert p2.grad is None

    # set_to_none=False
    p1.grad = Tensor([0.1, 0.2])
    p2.grad = Tensor([0.3, 0.4])
    opt.zero_grad(set_to_none=False)
    assert p1.grad is not None
    assert p1.grad.tolist() == [0.0, 0.0]
    assert p2.grad is not None
    assert p2.grad.tolist() == [0.0, 0.0]


def test_optimizer_step_skips_and_errors(active_backend):
    p_none_grad = Parameter([1.0, 2.0], requires_grad=True)
    p_no_grad = Parameter([3.0, 4.0], requires_grad=False)
    p_active = Parameter([5.0, 6.0], requires_grad=True)

    opt = SGD([p_none_grad, p_no_grad, p_active], lr=0.1)
    p_active.grad = Tensor([1.0, 1.0])

    opt.step()
    # p_none_grad and p_no_grad untouched
    assert p_none_grad.tolist() == [1.0, 2.0]
    assert p_no_grad.tolist() == [3.0, 4.0]
    # p_active updated: 5.0 - 0.1*1.0 = 4.9, 6.0 - 0.1*1.0 = 5.9
    assert p_active.tolist() == pytest.approx([4.9, 5.9], abs=1e-5)

    # Shape mismatch error
    p_err = Parameter([1.0, 2.0], requires_grad=True)
    opt_err = SGD([p_err], lr=0.1)
    p_err.grad = Tensor([1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError, match="does not match parameter shape"):
        opt_err.step()


# =============================================================================
# 2. SGD Numerical Tests
# =============================================================================

def test_sgd_basic_update(active_backend):
    p = Parameter([1.0, -2.0], requires_grad=True)
    opt = SGD([p], lr=0.1)

    p.grad = Tensor([0.5, -0.25])
    opt.step()

    # Expected: [1.0 - 0.1*0.5, -2.0 - 0.1*(-0.25)] = [0.95, -1.975]
    assert p.tolist() == pytest.approx([0.95, -1.975], abs=1e-6)


def test_sgd_weight_decay(active_backend):
    p = Parameter([1.0, -2.0], requires_grad=True)
    opt = SGD([p], lr=0.1, weight_decay=0.01)

    p.grad = Tensor([0.5, -0.25])
    opt.step()

    # Effective grad = g + wd * p = [0.5 + 0.01*1.0, -0.25 + 0.01*(-2.0)] = [0.51, -0.27]
    # Expected p = [1.0 - 0.1*0.51, -2.0 - 0.1*(-0.27)] = [0.949, -1.973]
    assert p.tolist() == pytest.approx([0.949, -1.973], abs=1e-6)


def test_sgd_momentum_2_steps(active_backend):
    p = Parameter([1.0, -2.0], requires_grad=True)
    opt = SGD([p], lr=0.1, momentum=0.9, dampening=0.0)

    # Step 1: v_1 = g_1 = [0.5, -0.25]
    # p_1 = [1.0, -2.0] - 0.1 * [0.5, -0.25] = [0.95, -1.975]
    p.grad = Tensor([0.5, -0.25])
    opt.step()
    assert p.tolist() == pytest.approx([0.95, -1.975], abs=1e-6)
    assert p.backend.to_flat_list(opt.state[0]["momentum_buffer"]) == pytest.approx([0.5, -0.25], abs=1e-6)

    # Step 2: g_2 = [0.2, 0.1]
    # v_2 = 0.9 * [0.5, -0.25] + [0.2, 0.1] = [0.65, -0.125]
    # p_2 = [0.95, -1.975] - 0.1 * [0.65, -0.125] = [0.885, -1.9625]
    p.grad = Tensor([0.2, 0.1])
    opt.step()
    assert p.tolist() == pytest.approx([0.885, -1.9625], abs=1e-6)
    assert p.backend.to_flat_list(opt.state[0]["momentum_buffer"]) == pytest.approx([0.65, -0.125], abs=1e-6)


def test_sgd_nesterov(active_backend):
    p = Parameter([1.0, -2.0], requires_grad=True)
    opt = SGD([p], lr=0.1, momentum=0.9, nesterov=True)

    # Step 1: v_1 = g_1 = [0.5, -0.25]
    # update_1 = g_1 + 0.9 * v_1 = [0.95, -0.475]
    # p_1 = [1.0, -2.0] - 0.1 * [0.95, -0.475] = [0.905, -1.9525]
    p.grad = Tensor([0.5, -0.25])
    opt.step()
    assert p.tolist() == pytest.approx([0.905, -1.9525], abs=1e-6)

    # Step 2: g_2 = [0.2, 0.1]
    # v_2 = 0.9 * v_1 + g_2 = [0.65, -0.125]
    # update_2 = g_2 + 0.9 * v_2 = [0.2, 0.1] + [0.585, -0.1125] = [0.785, -0.0125]
    # p_2 = [0.905, -1.9525] - 0.1 * [0.785, -0.0125] = [0.8265, -1.95125]
    p.grad = Tensor([0.2, 0.1])
    opt.step()
    assert p.tolist() == pytest.approx([0.8265, -1.95125], abs=1e-6)


# =============================================================================
# 3. Adam Numerical Tests
# =============================================================================

def test_adam_first_and_second_step(active_backend):
    p = Parameter([1.0], requires_grad=True)
    opt = Adam([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    # Step 1: g_1 = [0.5]
    # m_1 = 0.1 * 0.5 = 0.05 -> m_hat_1 = 0.05 / (1 - 0.9) = 0.5
    # v_1 = 0.001 * 0.25 = 0.00025 -> v_hat_1 = 0.00025 / (1 - 0.999) = 0.25
    # denom = sqrt(0.25) + 1e-8 = 0.5
    # step_size = 0.1 * (0.5 / 0.5) = 0.1
    # p_1 = 1.0 - 0.1 = 0.9
    p.grad = Tensor([0.5])
    opt.step()
    assert p.tolist() == pytest.approx([0.9], abs=1e-5)
    assert opt.state[0]["step"] == 1

    # Step 2: g_2 = [0.5]
    # m_2 = 0.9 * 0.05 + 0.1 * 0.5 = 0.095 -> m_hat_2 = 0.095 / (1 - 0.81) = 0.5
    # v_2 = 0.999 * 0.00025 + 0.001 * 0.25 = 0.00049975 -> v_hat_2 = 0.00049975 / (1 - 0.999^2) = 0.250000
    # step_size = 0.1 * (0.5 / 0.5) = 0.1
    # p_2 = 0.9 - 0.1 = 0.8
    p.grad = Tensor([0.5])
    opt.step()
    assert p.tolist() == pytest.approx([0.8], abs=1e-5)
    assert opt.state[0]["step"] == 2


def test_adam_step_increment_and_skip(active_backend):
    p1 = Parameter([1.0], requires_grad=True)
    p2 = Parameter([2.0], requires_grad=True)
    opt = Adam([p1, p2], lr=0.1)

    p1.grad = Tensor([0.5])
    p2.grad = None

    opt.step()
    assert opt.state[0]["step"] == 1
    assert 1 not in opt.state  # p2 was skipped, no state created


# =============================================================================
# 4. AdamW & Decoupled Weight Decay Tests
# =============================================================================

def test_adamw_vs_adam_weight_decay_distinction(active_backend):
    # Parameter and gradient setup
    p_adam = Parameter([1.0, -2.0], requires_grad=True)
    p_adamw = Parameter([1.0, -2.0], requires_grad=True)

    opt_adam = Adam([p_adam], lr=0.1, weight_decay=0.05)
    opt_adamw = AdamW([p_adamw], lr=0.1, weight_decay=0.05)

    p_adam.grad = Tensor([0.5, 0.5])
    p_adamw.grad = Tensor([0.5, 0.5])

    opt_adam.step()
    opt_adamw.step()

    # In Adam, weight decay alters effective gradient (and second moment)
    # In AdamW, weight decay directly scales the parameter before Adam step
    # Thus, their resulting parameters must NOT be equal
    assert p_adam.tolist() != p_adamw.tolist()


def test_adamw_zero_gradient_decay(active_backend):
    p = Parameter([10.0, -20.0], requires_grad=True)
    lr = 0.1
    wd = 0.05
    opt = AdamW([p], lr=lr, weight_decay=wd)

    p.grad = Tensor([0.0, 0.0])
    opt.step()

    # When grad=0, m=0, v=0, update=0. Parameter only experiences decoupled decay:
    # p_1 = p_0 * (1 - lr * wd) = p_0 * (1 - 0.005) = p_0 * 0.995
    # [10.0 * 0.995, -20.0 * 0.995] = [9.95, -19.9]
    assert p.tolist() == pytest.approx([9.95, -19.9], abs=1e-6)


# =============================================================================
# 5. State Dict Serialization & Round-trip
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

    # Run opt2 2 steps as well so initial weights match opt1 after 2 steps
    for _ in range(2):
        p1_fresh.grad = Tensor([[0.1, 0.2], [0.3, 0.4]])
        p2_fresh.grad = Tensor([0.5, 0.6])
        opt2.step()

    # Load state into opt2
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
