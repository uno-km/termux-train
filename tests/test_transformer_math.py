"""
tests/test_transformer_math.py
==============================
Test suite for Gate 7.2: Transformer Math Primitives & Generalized N-D Batched Matmul.

Tests:
  1. 4D Batched Matmul forward & backward (B, H, S, d_k) @ (B, H, d_k, S) -> (B, H, S, S)
  2. Broadcasted batch dimensions & unbroadcasting backward gradient accumulation
  3. 1D vector promotion matmul combinations
  4. Core math primitives with autograd (exp, sqrt, max subgradient, swapaxes)
  5. Stable LogSumExp & Softmax numerical behavior under large logits
  6. Deep autograd DAG iterative traversal (>1000 nodes without RecursionError)
"""

import math
import pytest
from termux_train import Tensor, zeros, ones, randn, get_backend, set_backend, available_backends
from termux_train.utils.gradcheck import gradcheck


# =============================================================================
# Section 1: Generalized N-D Batched Matmul & Broadcasting
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_4d_batched_matmul_attention_shapes(backend_name):
    set_backend(backend_name)

    # Q: (B=2, H=3, S=4, D=5), K: (B=2, H=3, S=4, D=5)
    B, H, S, D = 2, 3, 4, 5
    Q = randn((B, H, S, D), requires_grad=True)
    K = randn((B, H, S, D), requires_grad=True)

    # K^T: (B, H, D, S)
    K_T = K.transpose(0, 1, 3, 2)
    assert K_T.shape == (B, H, D, S)

    # Scores = Q @ K^T -> (B, H, S, S)
    Scores = Q @ K_T
    assert Scores.shape == (B, H, S, S)

    # Backward pass
    loss = Scores.sum()
    loss.backward()

    assert Q.grad is not None
    assert K.grad is not None
    assert Q.grad.shape == (B, H, S, D)
    assert K.grad.shape == (B, H, S, D)


@pytest.mark.parametrize("backend_name", available_backends())
def test_broadcasted_batch_matmul_and_unbroadcasting(backend_name):
    set_backend(backend_name)

    # A: (1, 3, 4, 5), B: (2, 1, 5, 6) -> Output: (2, 3, 4, 6)
    A = randn((1, 3, 4, 5), requires_grad=True)
    B = randn((2, 1, 5, 6), requires_grad=True)

    C = A @ B
    assert C.shape == (2, 3, 4, 6)

    loss = C.sum()
    loss.backward()

    assert A.grad is not None
    assert B.grad is not None
    assert A.grad.shape == (1, 3, 4, 5)
    assert B.grad.shape == (2, 1, 5, 6)


@pytest.mark.parametrize("backend_name", available_backends())
def test_1d_promoted_matmul_combinations(backend_name):
    set_backend(backend_name)

    # 1. 1D @ 1D -> scalar
    v1 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    v2 = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    dot = v1 @ v2
    assert dot.shape == ()
    assert dot.item() == 32.0
    dot.backward()
    assert v1.grad.tolist() == [4.0, 5.0, 6.0]
    assert v2.grad.tolist() == [1.0, 2.0, 3.0]

    # 2. 1D @ 2D: (3,) @ (3, 4) -> (4,)
    v = randn((3,), requires_grad=True)
    M = randn((3, 4), requires_grad=True)
    out1 = v @ M
    assert out1.shape == (4,)
    out1.sum().backward()
    assert v.grad.shape == (3,)
    assert M.grad.shape == (3, 4)

    # 3. 2D @ 1D: (4, 3) @ (3,) -> (4,)
    M2 = randn((4, 3), requires_grad=True)
    v3 = randn((3,), requires_grad=True)
    out2 = M2 @ v3
    assert out2.shape == (4,)
    out2.sum().backward()
    assert M2.grad.shape == (4, 3)
    assert v3.grad.shape == (3,)


# =============================================================================
# Section 2: Core Math Primitives & Autograd
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_exp_and_sqrt_autograd(backend_name):
    set_backend(backend_name)

    # exp
    x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
    y = x.exp()
    y.sum().backward()
    assert x.grad is not None
    assert pytest.approx(x.grad.tolist()) == [1.0, math.exp(1.0), math.exp(2.0)]

    # sqrt
    z = Tensor([4.0, 9.0, 16.0], requires_grad=True)
    w = z.sqrt()
    assert pytest.approx(w.tolist()) == [2.0, 3.0, 4.0]
    w.sum().backward()
    # d/dx (sqrt(x)) = 0.5 / sqrt(x)
    assert pytest.approx(z.grad.tolist()) == [0.25, 1.0 / 6.0, 0.125]


@pytest.mark.parametrize("backend_name", available_backends())
def test_max_reduction_subgradient(backend_name):
    set_backend(backend_name)

    # x: [[1, 5, 2], [7, 3, 4]]
    x = Tensor([[1.0, 5.0, 2.0], [7.0, 3.0, 4.0]], requires_grad=True)
    m = x.max(axis=-1)
    assert m.shape == (2,)
    assert m.tolist() == [5.0, 7.0]

    m.sum().backward()
    assert x.grad is not None
    # Subgradient must route 1.0 to positions [0, 1] and [1, 0]
    assert x.grad.tolist() == [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]


@pytest.mark.parametrize("backend_name", available_backends())
def test_swapaxes_and_transpose(backend_name):
    set_backend(backend_name)

    t = randn((2, 3, 4, 5), requires_grad=True)
    swapped = t.swapaxes(1, 2)
    assert swapped.shape == (2, 4, 3, 5)

    swapped.sum().backward()
    assert t.grad.shape == (2, 3, 4, 5)


@pytest.mark.parametrize("backend_name", available_backends())
def test_stable_logsumexp_and_softmax(backend_name):
    set_backend(backend_name)

    # Test extreme values (normally overflows in naive exp without max subtraction)
    x = Tensor([[1000.0, 1001.0, 1002.0]], requires_grad=True)
    
    # LogSumExp
    lse = x.logsumexp(axis=-1)
    # log(exp(1000) + exp(1001) + exp(1002)) = 1002 + log(exp(-2) + exp(-1) + 1)
    expected_lse = 1002.0 + math.log(math.exp(-2.0) + math.exp(-1.0) + 1.0)
    assert pytest.approx(lse.item(), rel=1e-5) == expected_lse

    # Softmax
    probs = x.softmax(axis=-1)
    assert probs.shape == (1, 3)
    # Sum of probabilities must equal 1.0
    assert pytest.approx(probs.sum().item(), abs=1e-4) == 1.0
    assert all(p > 0.0 and math.isfinite(p) for p in probs.tolist()[0])

    # Softmax autograd backward
    probs.sum().backward()
    assert x.grad is not None
    assert all(math.isfinite(g) for g in x.grad.tolist()[0])


# =============================================================================
# Section 3: Deep Autograd DAG Iterative Traversal
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_deep_dag_iterative_backward_recursion_safety(backend_name):
    set_backend(backend_name)

    # Chain of 1500 additions (would exceed Python recursion limit of 1000 in recursive DFS)
    x = Tensor(1.0, requires_grad=True)
    curr = x
    depth = 1500
    for _ in range(depth):
        curr = curr + 1.0

    assert curr.item() == 1501.0

    # backward must complete without RecursionError
    curr.backward()
    assert x.grad is not None
    assert x.grad.item() == 1.0
