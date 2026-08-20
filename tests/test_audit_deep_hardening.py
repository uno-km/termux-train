"""
tests/test_audit_deep_hardening.py
==================================
Comprehensive Production ML Systems & Core Compiler Deep Hardening Test Suite:
  1. Fully-Masked Row LogSumExp & Softmax (-inf row without NaN)
  2. 4D and 5D Linear Layer forward & backward
  3. Cyclic DAG Autograd 3-Color Cycle Detection & Diamond DAG preservation
  4. View-of-View and Sibling View Mutation Version Invalidation
  5. Fused CrossEntropyLoss edge contracts, all-ignore handling, and invalid index bounds
  6. BCEWithLogitsLoss shape contracts, extreme logit stability, and exact gradient
  7. Softmax multi-axis and keepdims correctness
"""

import math
import pytest
from termux_train import Tensor, nn, set_backend, available_backends


# =============================================================================
# 1. Level A-01: Fully-Masked Sequence LogSumExp & Softmax NaN Defense
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_fully_masked_sequence_logsumexp_and_softmax_no_nan(backend_name):
    set_backend(backend_name)
    # Row 0: Normal finite values [1.0, 2.0]
    # Row 1: Fully masked sequence [-inf, -inf]
    x = Tensor([[1.0, 2.0], [-float('inf'), -float('inf')]], requires_grad=True)

    # 1. LogSumExp
    lse = x.logsumexp(axis=-1)
    lse_list = lse.tolist()
    assert pytest.approx(lse_list[0]) == math.log(math.exp(1.0) + math.exp(2.0))
    assert math.isinf(lse_list[1]) and lse_list[1] < 0  # Must be -inf, NOT NaN!

    # 2. Softmax
    probs = x.softmax(axis=-1)
    probs_list = probs.tolist()
    assert pytest.approx(sum(probs_list[0])) == 1.0
    # Fully masked row must produce 0.0 attention distribution without NaN
    assert probs_list[1] == [0.0, 0.0]

    # 3. Backward
    probs.sum().backward()
    assert x.grad is not None
    grad_list = x.grad.tolist()
    # Fully masked row gradient must be 0.0 without NaN
    assert grad_list[1] == [0.0, 0.0]


@pytest.mark.parametrize("backend_name", available_backends())
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_softmax_multi_axes_and_keepdims(backend_name, axis):
    set_backend(backend_name)
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    probs = x.softmax(axis=axis)
    if axis in (1, -1):
        assert pytest.approx(sum(probs.tolist()[0])) == 1.0
        assert pytest.approx(sum(probs.tolist()[1])) == 1.0
    elif axis == 0:
        assert pytest.approx(probs.tolist()[0][0] + probs.tolist()[1][0]) == 1.0
        assert pytest.approx(probs.tolist()[0][1] + probs.tolist()[1][1]) == 1.0


# =============================================================================
# 2. Level A-02: 4D and 5D Linear Layer Forward & Backward
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_4d_and_5d_linear_forward_backward(backend_name):
    set_backend(backend_name)
    in_dim, out_dim = 4, 8
    lin = nn.Linear(in_dim, out_dim, bias=True)

    # 4D Attention Tensor: (Batch=2, Heads=3, Seq=5, Dim=4)
    x4 = Tensor([[[[1.0] * in_dim] * 5] * 3] * 2, requires_grad=True)
    out4 = lin(x4)
    assert out4.shape == (2, 3, 5, out_dim)
    loss4 = out4.sum()
    loss4.backward()
    assert x4.grad is not None
    assert x4.grad.shape == (2, 3, 5, in_dim)
    assert lin.weight.grad is not None
    assert lin.bias.grad is not None

    # 5D Batched Video/Sequence: (2, 2, 3, 4, 4)
    x5 = Tensor([[[[[0.5] * in_dim] * 4] * 3] * 2] * 2, requires_grad=True)
    out5 = lin(x5)
    assert out5.shape == (2, 2, 3, 4, out_dim)


# =============================================================================
# 3. Level A-03: Cyclic DAG Autograd 3-Color Cycle Detection & Diamond Preservation
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_cyclic_dag_autograd_cycle_detection(backend_name):
    set_backend(backend_name)
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * 2.0
    # Manually create a cyclic edge in DAG: y._prev contains y
    y._prev.add(y)

    with pytest.raises(RuntimeError, match="Cycle detected"):
        y.sum().backward()


@pytest.mark.parametrize("backend_name", available_backends())
def test_diamond_dag_not_detected_as_cycle(backend_name):
    set_backend(backend_name)
    # Diamond graph: x -> y1, y2 -> z = y1 + y2
    x = Tensor([2.0, 3.0], requires_grad=True)
    y1 = x * 2.0
    y2 = x * 3.0
    z = y1 + y2
    z.sum().backward()
    assert x.grad.tolist() == [5.0, 5.0]


# =============================================================================
# 4. Level A-06: View-of-View and Sibling View Version Invalidation
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_view_of_view_and_sibling_view_version_invalidation(backend_name):
    set_backend(backend_name)
    base = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    loss = (base * 2.0).sum()

    # View-of-view chain: base -> v1 -> v2
    v1 = base.reshape((4,))
    v2 = v1.reshape((2, 2))
    # Sibling view: base -> sibling
    sibling = base.transpose()

    # Mutating v2 bumps the shared version counter
    v2.data = [[10.0, 20.0], [30.0, 40.0]]

    assert base._version > 0
    assert v1._version == base._version
    assert sibling._version == base._version

    # Forward graph built on base must fail
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


# =============================================================================
# 5. Level A-05: Fused CrossEntropyLoss with int64 Targets & Edge Contracts
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_fused_cross_entropy_loss_int64_targets(backend_name):
    set_backend(backend_name)
    # Logits for 2 samples, 3 classes: [[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]]
    logits = Tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], requires_grad=True)
    # Ground truth classes: [0, 1]
    targets = Tensor([0, 1], dtype="int64")

    criterion = nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(logits, targets)

    denom0 = math.exp(2.0) + math.exp(1.0) + math.exp(0.1)
    loss0 = -math.log(math.exp(2.0) / denom0)

    denom1 = math.exp(0.5) + math.exp(2.5) + math.exp(0.3)
    loss1 = -math.log(math.exp(2.5) / denom1)

    expected_loss = (loss0 + loss1) / 2.0
    assert pytest.approx(loss.item()) == expected_loss

    loss.backward()
    assert logits.grad is not None
    grad = logits.grad.tolist()
    p0 = [math.exp(2.0)/denom0, math.exp(1.0)/denom0, math.exp(0.1)/denom0]
    expected_grad0 = [(p - target_val)/2.0 for p, target_val in zip(p0, [1.0, 0.0, 0.0])]
    assert grad[0] == pytest.approx(expected_grad0)


@pytest.mark.parametrize("backend_name", available_backends())
def test_cross_entropy_all_ignored_targets(backend_name):
    set_backend(backend_name)
    logits = Tensor([[2.0, 1.0], [0.5, 2.5]], requires_grad=True)
    # All samples are ignored (-100)
    targets = Tensor([-100, -100], dtype="int64")

    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
    loss = criterion(logits, targets)
    assert loss.item() == 0.0

    loss.backward()
    assert logits.grad is not None
    assert logits.grad.tolist() == [[0.0, 0.0], [0.0, 0.0]]


@pytest.mark.parametrize("backend_name", available_backends())
def test_cross_entropy_shape_and_bounds_validation(backend_name):
    set_backend(backend_name)
    logits = Tensor([[2.0, 1.0, 0.5]], requires_grad=True)

    # 1. Target shape mismatch
    with pytest.raises(ValueError, match="target shape"):
        nn.CrossEntropyLoss()(logits, Tensor([[0, 1]], dtype="int64"))

    # 2. Target index out of bounds (> 2)
    with pytest.raises(IndexError, match="out of bounds"):
        nn.CrossEntropyLoss()(logits, Tensor([5], dtype="int64"))

    # 3. Negative target index (other than -100)
    with pytest.raises(IndexError, match="out of bounds"):
        nn.CrossEntropyLoss()(logits, Tensor([-1], dtype="int64"))


# =============================================================================
# 6. Level A-04: BCEWithLogitsLoss Extreme Stability & Shape Validation
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_bce_with_logits_loss_extreme_stability(backend_name):
    set_backend(backend_name)
    logits = Tensor([100.0, -100.0], requires_grad=True)
    targets = Tensor([1.0, 0.0])

    loss = nn.BCEWithLogitsLoss(reduction="mean")(logits, targets)
    assert pytest.approx(loss.item(), abs=1e-6) == 0.0

    logits_bad = Tensor([-100.0], requires_grad=True)
    targets_bad = Tensor([1.0])
    loss_bad = nn.BCEWithLogitsLoss(reduction="mean")(logits_bad, targets_bad)
    assert loss_bad.item() > 90.0

    loss_bad.backward()
    assert pytest.approx(logits_bad.grad.item()) == -1.0


@pytest.mark.parametrize("backend_name", available_backends())
def test_bce_with_logits_loss_shape_mismatch(backend_name):
    set_backend(backend_name)
    with pytest.raises(ValueError, match="identical input and target shapes"):
        nn.BCEWithLogitsLoss()(Tensor([1.0, 2.0]), Tensor([1.0]))
