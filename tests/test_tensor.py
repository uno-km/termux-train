"""
tests/test_tensor.py
====================
Unit tests for Tensor data model, shape inference, operators, and representations.
"""

import pytest
from termux_train import Tensor, zeros, ones, zeros_like, ones_like, randn, set_backend, get_backend, available_backends

@pytest.fixture(params=["python"] + (["numpy"] if "numpy" in available_backends() else []))
def active_backend(request):
    set_backend(request.param)
    return request.param

def test_tensor_creation_and_shape(active_backend):
    t0 = Tensor(42.0)
    assert t0.shape == ()
    assert t0.ndim == 0
    assert t0.item() == 42.0

    t1 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    assert t1.shape == (3,)
    assert t1.ndim == 1
    assert t1.requires_grad is True

    t2 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert t2.shape == (2, 2)
    assert t2.ndim == 2

def test_tensor_ragged_error(active_backend):
    with pytest.raises(ValueError):
        Tensor([[1.0, 2.0], [3.0]])

def test_tensor_reshape_and_flatten(active_backend):
    t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert t.shape == (2, 3)

    t_flat = t.flatten()
    assert t_flat.shape == (6,)
    assert t_flat.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    t_reshaped = t.reshape(3, 2)
    assert t_reshaped.shape == (3, 2)

def test_tensor_transpose(active_backend):
    t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t_t = t.T
    assert t_t.shape == (3, 2)
    assert t_t.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

def test_tensor_factories(active_backend):
    z = zeros((2, 3))
    assert z.shape == (2, 3)
    assert z.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    o = ones((3, 2))
    assert o.shape == (3, 2)
    assert o.tolist() == [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]

    zl = zeros_like(o)
    assert zl.shape == (3, 2)
    assert zl.tolist() == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

def test_tensor_representation(active_backend):
    t = Tensor([[1.0, 2.0]], requires_grad=True)
    rep = repr(t)
    assert "Tensor(" in rep
    assert "shape=(1, 2)" in rep
    assert "requires_grad=True" in rep

def test_tensor_data_cross_backend_safety():
    # 1. Python backend tensor
    set_backend("python")
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])
    a.data = b
    assert a.tolist() == [3.0, 4.0]
    assert a.backend.name == "python"

    # 2. Cross backend assign (NumPy -> Python)
    if "numpy" in available_backends():
        set_backend("numpy")
        np_t = Tensor([5.0, 6.0])
        a.data = np_t
        assert a.tolist() == [5.0, 6.0]
        assert a.backend.name == "python"

    # 3. Raw list assign
    a.data = [7.0, 8.0]
    assert a.tolist() == [7.0, 8.0]
    assert a.backend.name == "python"

def test_tensor_constructor_uses_source_backend_when_backend_none():
    set_backend("python")
    a = Tensor([1.0, 2.0])
    b = Tensor(a)
    assert b.backend.name == "python"
    assert b.tolist() == [1.0, 2.0]

def test_tensor_constructor_backend_override():
    if "numpy" in available_backends():
        set_backend("python")
        a = Tensor([1.0, 2.0])
        set_backend("numpy")
        np_b = get_backend()

        b = Tensor(a, backend=np_b)
        assert b.backend.name == "numpy"
        assert b.tolist() == [1.0, 2.0]

def test_cross_backend_binary_auto_convert_non_grad():
    if "numpy" in available_backends():
        set_backend("python")
        a = Tensor([1.0, 2.0])

        set_backend("numpy")
        b = Tensor([3.0, 4.0])

        c = a + b
        assert c.backend.name == "python"
        assert c.tolist() == [4.0, 6.0]

        d = b + a
        assert d.backend.name == "numpy"
        assert d.tolist() == [4.0, 6.0]

def test_cross_backend_requires_grad_raises():
    if "numpy" in available_backends():
        set_backend("python")
        a = Tensor([1.0, 2.0], requires_grad=True)

        set_backend("numpy")
        b = Tensor([3.0, 4.0], requires_grad=True)

        with pytest.raises(RuntimeError):
            _ = a + b
