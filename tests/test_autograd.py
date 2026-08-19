"""
tests/test_autograd.py
======================
Unit tests for Reverse-Mode Autograd DAG graph, gradient accumulation, and chain rule.
"""

import pytest
from termux_train import Tensor, set_backend, available_backends

@pytest.fixture(params=["python"] + (["numpy"] if "numpy" in available_backends() else []))
def active_backend(request):
    set_backend(request.param)
    return request.param

def test_autograd_scalar_ops(active_backend):
    # y = x^2 + 3x + 5 -> dy/dx = 2x + 3
    x = Tensor(2.0, requires_grad=True)
    y = x * x + 3.0 * x + 5.0
    y.backward()

    assert x.grad is not None
    assert x.grad.item() == pytest.approx(7.0)

def test_autograd_gradient_accumulation(active_backend):
    # y = x + x + x -> dy/dx = 3.0
    x = Tensor(4.0, requires_grad=True)
    y = x + x + x
    y.backward()

    assert x.grad.item() == pytest.approx(3.0)

def test_autograd_division_and_pow(active_backend):
    # y = (x^3) / 2.0 -> dy/dx = 3/2 * x^2 = 1.5 * 4 = 6.0
    x = Tensor(2.0, requires_grad=True)
    y = (x ** 3) / 2.0
    y.backward()

    assert x.grad.item() == pytest.approx(6.0)

def test_autograd_vector_elementwise(active_backend):
    # y = (x * 2.0).sum()
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x * 2.0).sum()
    y.backward()

    assert x.grad.tolist() == [2.0, 2.0, 2.0]

def test_autograd_matmul_2d(active_backend):
    # Y = A @ B, loss = Y.sum()
    # A: (2, 3), B: (3, 2)
    # dA = dY @ B.T, dB = A.T @ dY where dY is ones(2, 2)
    A = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    B = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True)

    Y = A @ B
    loss = Y.sum()
    loss.backward()

    # B.T: [[1, 0, 1], [0, 1, 1]]
    # dA = ones(2, 2) @ [[1, 0, 1], [0, 1, 1]] = [[1, 1, 2], [1, 1, 2]]
    assert A.grad.shape == (2, 3)
    assert A.grad.tolist() == [[1.0, 1.0, 2.0], [1.0, 1.0, 2.0]]

    # A.T: [[1, 4], [2, 5], [3, 6]]
    # dB = A.T @ ones(2, 2) = [[5, 5], [7, 7], [9, 9]]
    assert B.grad.shape == (3, 2)
    assert B.grad.tolist() == [[5.0, 5.0], [7.0, 7.0], [9.0, 9.0]]

def test_autograd_relu_and_mean(active_backend):
    x = Tensor([-2.0, 3.0, 0.0, 5.0], requires_grad=True)
    h = x.relu() # [0.0, 3.0, 0.0, 5.0]
    loss = h.mean() # (0 + 3 + 0 + 5) / 4 = 2.0
    loss.backward()

    # d(loss)/dx_i = (1/4) if x_i > 0 else 0.0
    assert x.grad.tolist() == [0.0, 0.25, 0.0, 0.25]

def test_scalar_backward_without_gradient(active_backend):
    x = Tensor(2.0, requires_grad=True)
    y = x * x
    y.backward()
    assert x.grad.item() == 4.0

def test_non_scalar_backward_requires_gradient(active_backend):
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * x
    with pytest.raises(RuntimeError):
        y.backward()

def test_non_scalar_backward_with_explicit_gradient(active_backend):
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * x
    y.backward(Tensor([1.0, 1.0]))
    assert x.grad.tolist() == [2.0, 4.0]

def test_non_scalar_backward_allow_implicit_grad(active_backend):
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * x
    y.backward(allow_implicit_grad=True)
    assert x.grad.tolist() == [2.0, 4.0]

def test_transpose_nd_inverse_permutation_backward(active_backend):
    # 2D .T
    x2d = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y2d = x2d.T
    loss2d = y2d.sum()
    loss2d.backward()
    assert x2d.grad.tolist() == [[1.0, 1.0], [1.0, 1.0]]

    # 3D transpose with general permutation (2, 0, 1)
    data3d = [
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    ]
    x3d = Tensor(data3d, requires_grad=True) # shape (2, 3, 2)
    y3d = x3d.transpose((2, 0, 1)) # shape (2, 2, 3)
    loss3d = y3d.sum()
    loss3d.backward()
    
    # Gradient should propagate 1.0 to all original elements of shape (2, 3, 2)
    assert x3d.grad.shape == (2, 3, 2)
    for row in x3d.grad.tolist():
        for col in row:
            for val in col:
                assert val == 1.0

def test_axis_aware_sum_mean_backward(active_backend):
    # 1. axis=None mean
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y_mean = x.mean()
    y_mean.backward()
    assert x.grad.tolist() == [[0.25, 0.25], [0.25, 0.25]]

    # 2. axis=0 mean
    x0 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y0 = x0.mean(axis=0)
    y0.backward(Tensor([1.0, 1.0]))
    assert x0.grad.tolist() == [[0.5, 0.5], [0.5, 0.5]]

    # 3. axis=1 mean
    x1 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y1 = x1.mean(axis=1)
    y1.backward(Tensor([1.0, 1.0]))
    assert x1.grad.tolist() == [[0.5, 0.5], [0.5, 0.5]]

    # 4. axis=0 sum
    xs = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    ys = xs.sum(axis=0)
    ys.backward(Tensor([1.0, 1.0]))
    assert xs.grad.tolist() == [[1.0, 1.0], [1.0, 1.0]]

    # 5. keepdims=True mean
    xk = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    yk = xk.mean(axis=0, keepdims=True)
    yk.backward(Tensor([[1.0, 1.0]]))
    assert xk.grad.tolist() == [[0.5, 0.5], [0.5, 0.5]]

def test_matmul_1d_1d_backward(active_backend):
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)

    y = a @ b
    y.backward()

    assert y.shape == ()
    assert y.item() == pytest.approx(11.0)
    assert a.grad.tolist() == [3.0, 4.0]
    assert b.grad.tolist() == [1.0, 2.0]

def test_matmul_2d_1d_backward(active_backend):
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([5.0, 6.0], requires_grad=True)

    y = a @ b
    y.backward(Tensor([1.0, 1.0]))

    assert y.tolist() == [17.0, 39.0]
    assert a.grad.tolist() == [[5.0, 6.0], [5.0, 6.0]]
    assert b.grad.tolist() == [4.0, 6.0]

def test_matmul_2d_2d_backward(active_backend):
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    w = Tensor([[3.0], [4.0]], requires_grad=True)

    y = x @ w
    loss = y.mean()
    loss.backward()

    assert x.grad.tolist() == [[3.0, 4.0]]
    assert w.grad.tolist() == [[1.0], [2.0]]

def test_matmul_3d_2d_backward(active_backend):
    # a: (2, 2, 2), w: (2, 1) -> y: (2, 2, 1)
    a = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True)
    w = Tensor([[1.0], [2.0]], requires_grad=True)

    y = a @ w
    loss = y.sum()
    loss.backward()

    assert y.shape == (2, 2, 1)
    assert a.grad.shape == (2, 2, 2)
    assert a.grad.tolist() == [[[1.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [1.0, 2.0]]]
    # dW = sum over all batches: sum([[1, 2], [3, 4], [5, 6], [7, 8]]) = [16, 20]
    assert w.grad.tolist() == [[16.0], [20.0]]

def test_matmul_3d_3d_backward(active_backend):
    # a: (2, 1, 2), b: (2, 2, 1) -> y: (2, 1, 1)
    a = Tensor([[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=True)
    b = Tensor([[[1.0], [2.0]], [[3.0], [4.0]]], requires_grad=True)

    y = a @ b
    loss = y.sum()
    loss.backward()

    assert y.shape == (2, 1, 1)
    assert a.grad.tolist() == [[[1.0, 2.0]], [[3.0, 4.0]]]
    assert b.grad.tolist() == [[[1.0], [2.0]], [[3.0], [4.0]]]

def test_accumulate_grad_data_respects_requires_grad(active_backend):
    frozen = Tensor([1.0, 2.0], requires_grad=False)
    frozen._accumulate_grad_data([3.0, 4.0])
    assert frozen.grad is None

def test_accumulate_grad_data_accumulates_twice(active_backend):
    x = Tensor([1.0, 2.0], requires_grad=True)

    x._accumulate_grad_data([3.0, 4.0])
    assert x.grad.tolist() == [3.0, 4.0]

    x._accumulate_grad_data([5.0, 6.0])
    assert x.grad.tolist() == [8.0, 10.0]
    assert x.grad.requires_grad is False
    assert x.grad.backend.name == x.backend.name

def test_matmul_gradient_accumulation_across_backward_calls(active_backend):
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    w = Tensor([[3.0], [4.0]], requires_grad=True)

    loss1 = (x @ w).sum()
    loss1.backward()

    first_x_grad = x.grad.tolist()
    first_w_grad = w.grad.tolist()

    loss2 = (x @ w).sum()
    loss2.backward()

    assert x.grad.tolist() == [
        [2.0 * first_x_grad[0][0], 2.0 * first_x_grad[0][1]]
    ]
    assert w.grad.tolist() == [
        [2.0 * first_w_grad[0][0]],
        [2.0 * first_w_grad[1][0]],
    ]

def test_matmul_zero_grad_then_backward_restarts_gradient(active_backend):
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    w = Tensor([[3.0], [4.0]], requires_grad=True)

    (x @ w).sum().backward()

    x.zero_grad()
    w.zero_grad()

    assert x.grad is None
    assert w.grad is None

    (x @ w).sum().backward()

    assert x.grad.tolist() == [[3.0, 4.0]]
    assert w.grad.tolist() == [[1.0], [2.0]]

def test_matmul_only_left_requires_grad(active_backend):
    a = Tensor([[1.0, 2.0]], requires_grad=True)
    b = Tensor([[3.0], [4.0]], requires_grad=False)

    (a @ b).sum().backward()

    assert a.grad.tolist() == [[3.0, 4.0]]
    assert b.grad is None

def test_matmul_only_right_requires_grad(active_backend):
    a = Tensor([[1.0, 2.0]], requires_grad=False)
    b = Tensor([[3.0], [4.0]], requires_grad=True)

    (a @ b).sum().backward()

    assert a.grad is None
    assert b.grad.tolist() == [[1.0], [2.0]]

def test_matmul_3d_2d_only_input_requires_grad(active_backend):
    a = Tensor([[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=True) # (2, 1, 2)
    w = Tensor([[1.0], [2.0]], requires_grad=False)              # (2, 1)

    (a @ w).sum().backward()

    assert a.grad is not None
    assert a.grad.shape == (2, 1, 2)
    assert w.grad is None

def test_matmul_3d_2d_only_weight_requires_grad(active_backend):
    a = Tensor([[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=False) # (2, 1, 2)
    w = Tensor([[1.0], [2.0]], requires_grad=True)               # (2, 1)

    (a @ w).sum().backward()

    assert a.grad is None
    assert w.grad is not None
    assert w.grad.tolist() == [[4.0], [6.0]]

def test_matmul_1d_2d_backward(active_backend):
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], requires_grad=True)

    y = a @ b
    assert y.shape == (3,)
    assert y.tolist() == [15.0, 18.0, 21.0]

    loss = y.sum()
    loss.backward()

    # dA = dY @ B^T = [1, 1, 1] @ [[3, 6], [4, 7], [5, 8]] = [12, 21]
    assert a.grad.tolist() == [12.0, 21.0]
    # dB = outer(A, dY) = [[1, 1, 1], [2, 2, 2]]
    assert b.grad.tolist() == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]

def test_matmul_1d_3d_backward(active_backend):
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([[[3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0]]], requires_grad=True)

    y = a @ b
    assert y.shape == (2, 2)
    assert y.tolist() == [[13.0, 16.0], [25.0, 28.0]]

    loss = y.sum()
    loss.backward()

    # dA = sum_b dY[b] @ B[b]^T = [7, 11] + [15, 19] = [22, 30]
    assert a.grad.tolist() == [22.0, 30.0]
    # dB[b] = outer(A, dY[b])
    assert b.grad.tolist() == [[[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]]

def test_matmul_2d_3d_backward(active_backend):
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[[1.0], [2.0]], [[3.0], [4.0]]], requires_grad=True)

    y = a @ b
    assert y.shape == (2, 2, 1)
    assert y.tolist() == [[[5.0], [11.0]], [[11.0], [25.0]]]

    loss = y.sum()
    loss.backward()

    # dA = sum_b dY[b] @ B[b]^T = [[1, 2], [1, 2]] + [[3, 4], [3, 4]] = [[4, 6], [4, 6]]
    assert a.grad.tolist() == [[4.0, 6.0], [4.0, 6.0]]
    # dB[b] = A^T @ dY[b] = [[4], [6]]
    assert b.grad.tolist() == [[[4.0], [6.0]], [[4.0], [6.0]]]

def test_matmul_3d_1d_backward(active_backend):
    a = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True)
    b = Tensor([2.0, 3.0], requires_grad=True)

    y = a @ b
    assert y.shape == (2, 2)
    assert y.tolist() == [[8.0, 18.0], [28.0, 38.0]]

    loss = y.sum()
    loss.backward()

    # dA[b] = outer(dY[b], b)
    assert a.grad.tolist() == [[[2.0, 3.0], [2.0, 3.0]], [[2.0, 3.0], [2.0, 3.0]]]
    # dB = sum_b A[b]^T @ dY[b] = [4, 6] + [12, 14] = [16, 20]
    assert b.grad.tolist() == [16.0, 20.0]

def test_matmul_all_shape_mismatches_and_unsupported(active_backend):
    # 1. 1D@1D inner mismatch
    with pytest.raises(ValueError):
        _ = Tensor([1.0, 2.0]) @ Tensor([1.0, 2.0, 3.0])

    # 2. 1D@2D inner mismatch
    with pytest.raises(ValueError):
        _ = Tensor([1.0, 2.0]) @ Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # 3. 1D@3D inner mismatch
    with pytest.raises(ValueError):
        _ = Tensor([1.0, 2.0]) @ Tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

    # 4. 2D@1D inner mismatch
    with pytest.raises(ValueError):
        _ = Tensor([[1.0, 2.0]]) @ Tensor([1.0, 2.0, 3.0])

    # 5. 2D@2D inner mismatch
    with pytest.raises(ValueError):
        _ = Tensor([[1.0, 2.0]]) @ Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # 6. 2D@3D inner mismatch
    with pytest.raises(ValueError):
        _ = Tensor([[1.0, 2.0]]) @ Tensor([[[1.0], [2.0], [3.0]]])

    # 7. 3D@1D inner mismatch
    with pytest.raises(ValueError):
        _ = Tensor([[[1.0, 2.0]]]) @ Tensor([1.0, 2.0, 3.0])

    # 8. 3D@2D inner mismatch
    with pytest.raises(ValueError):
        _ = Tensor([[[1.0, 2.0]]]) @ Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # 9. 3D@3D inner mismatch
    with pytest.raises(ValueError):
        _ = Tensor([[[1.0, 2.0]]]) @ Tensor([[[1.0], [2.0], [3.0]]])

    # 10. 3D@3D batch mismatch
    with pytest.raises(ValueError):
        _ = Tensor([[[1.0, 2.0]]]) @ Tensor([[[1.0], [2.0]], [[3.0], [4.0]]])

    # 11. 4D+ unsupported
    with pytest.raises(NotImplementedError):
        _ = Tensor([[[[1.0, 2.0]]]]) @ Tensor([[[[3.0], [4.0]]]])

    # 12. 0D input unsupported
    with pytest.raises(NotImplementedError):
        _ = Tensor(2.0) @ Tensor(3.0)
