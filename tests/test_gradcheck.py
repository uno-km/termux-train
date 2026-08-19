"""
tests/test_gradcheck.py
=======================
Verify Autograd analytical gradients against numerical finite differences.
"""

import pytest
from termux_train import Tensor, set_backend, available_backends
from termux_train.utils import gradcheck

@pytest.fixture(params=["python"] + (["numpy"] if "numpy" in available_backends() else []))
def active_backend(request):
    set_backend(request.param)
    return request.param

def test_gradcheck_simple_math(active_backend):
    def f(a, b):
        return (a * b + (a / (b + 1.0)) + (a ** 2)).sum()

    a = Tensor([[1.5, -2.0], [0.8, 3.2]], requires_grad=True)
    b = Tensor([[2.0, 1.0], [0.5, -0.5]], requires_grad=True)

    assert gradcheck(f, (a, b)) is True

def test_gradcheck_matmul_1d_1d(active_backend):
    def f(a, b):
        return a @ b

    a = Tensor([1.5, -2.0, 0.8], requires_grad=True)
    b = Tensor([2.0, 1.0, -0.5], requires_grad=True)
    assert gradcheck(f, (a, b)) is True

def test_gradcheck_matmul_1d_2d(active_backend):
    def f(a, b):
        return (a @ b).sum()

    a = Tensor([1.5, -2.0], requires_grad=True)
    b = Tensor([[0.5, -1.2, 0.3], [2.1, 0.4, -0.9]], requires_grad=True) # (2, 3)
    assert gradcheck(f, (a, b)) is True

def test_gradcheck_matmul_1d_3d(active_backend):
    def f(a, b):
        return (a @ b).sum()

    a = Tensor([1.5, -2.0], requires_grad=True)
    b = Tensor([[[0.5, -1.2], [2.1, 0.4]], [[-0.4, 1.1], [0.7, -0.2]]], requires_grad=True) # (2, 2, 2)
    assert gradcheck(f, (a, b)) is True

def test_gradcheck_matmul_2d_1d(active_backend):
    def f(a, b):
        return (a @ b).sum()

    a = Tensor([[1.5, -2.0], [0.8, 3.2]], requires_grad=True)
    b = Tensor([2.0, 1.0], requires_grad=True)
    assert gradcheck(f, (a, b)) is True

def test_gradcheck_matmul_2d_2d(active_backend):
    def f(w1, w2):
        return (w1 @ w2).sum()

    w1 = Tensor([[0.5, -1.2, 0.3], [2.1, 0.4, -0.9]], requires_grad=True) # (2, 3)
    w2 = Tensor([[-0.4, 1.1], [0.7, -0.2], [1.5, 0.8]], requires_grad=True) # (3, 2)
    assert gradcheck(f, (w1, w2)) is True

def test_gradcheck_matmul_2d_3d(active_backend):
    def f(a, b):
        return (a @ b).sum()

    a = Tensor([[1.5, -2.0], [0.8, 3.2]], requires_grad=True) # (2, 2)
    b = Tensor([[[0.5], [2.1]], [[-0.4], [0.7]]], requires_grad=True) # (2, 2, 1)
    assert gradcheck(f, (a, b)) is True

def test_gradcheck_matmul_3d_1d(active_backend):
    def f(a, b):
        return (a @ b).sum()

    a = Tensor([[[0.5, -1.2], [2.1, 0.4]], [[-0.4, 1.1], [0.7, -0.2]]], requires_grad=True) # (2, 2, 2)
    b = Tensor([1.5, -0.8], requires_grad=True) # (2,)
    assert gradcheck(f, (a, b)) is True

def test_gradcheck_matmul_3d_2d(active_backend):
    def f(a, w):
        return (a @ w).sum()

    a = Tensor([[[0.5, -1.2], [2.1, 0.4]], [[-0.4, 1.1], [0.7, -0.2]]], requires_grad=True) # (2, 2, 2)
    w = Tensor([[1.5], [-0.8]], requires_grad=True) # (2, 1)
    assert gradcheck(f, (a, w)) is True

def test_gradcheck_matmul_3d_3d(active_backend):
    def f(a, b):
        return (a @ b).sum()

    a = Tensor([[[0.5, -1.2]], [[-0.4, 1.1]]], requires_grad=True) # (2, 1, 2)
    b = Tensor([[[1.5], [-0.8]], [[0.7], [2.1]]], requires_grad=True) # (2, 2, 1)
    assert gradcheck(f, (a, b)) is True

def test_gradcheck_activations(active_backend):
    def f(x):
        return (x.sigmoid() + x.tanh()).mean()

    x = Tensor([[-1.0, 0.5], [2.0, -1.5]], requires_grad=True)
    assert gradcheck(f, (x,)) is True

def test_gradcheck_rejects_non_scalar_output(active_backend):
    x = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(RuntimeError):
        gradcheck(lambda t: t * t, (x,))

def test_gradcheck_accepts_scalar_output(active_backend):
    x = Tensor([1.0, 2.0], requires_grad=True)
    assert gradcheck(lambda t: (t * t).sum(), (x,)) is True
