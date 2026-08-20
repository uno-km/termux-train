"""
tests/test_dtype.py
===================
Test suite for Gate 7.1: Tensor Multi-Dtype Foundation.

Tests:
  1. Dtype creation and automatic inference (float32, int64, bool)
  2. Non-differentiable integer/boolean tensor invariants (requires_grad=False enforcement)
  3. Dtype preservation in .item(), .tolist(), and factory functions
  4. Backend dtype parity and cross-backend conversion
"""

import pytest
from termux_train import Tensor, tensor, zeros, ones, zeros_like, ones_like, randn, get_backend, set_backend, available_backends


@pytest.mark.parametrize("backend_name", available_backends())
def test_dtype_creation_and_inference(backend_name):
    set_backend(backend_name)

    # 1. Inferred float32
    t_float = Tensor([1.5, 2.5, 3.5])
    assert t_float.dtype == "float32"
    assert isinstance(t_float.item() if t_float.ndim == 0 else t_float.tolist()[0], float)

    # 2. Inferred int64
    t_int = Tensor([1, 2, 3])
    assert t_int.dtype == "int64"
    assert t_int.requires_grad is False
    assert isinstance(t_int.tolist()[0], int)

    # 3. Inferred bool
    t_bool = Tensor([True, False, True])
    assert t_bool.dtype == "bool"
    assert t_bool.requires_grad is False
    assert isinstance(t_bool.tolist()[0], bool)

    # 4. Explicit dtype override
    t_forced_float = Tensor([1, 2, 3], dtype="float32")
    assert t_forced_float.dtype == "float32"

    t_forced_int = Tensor([1.9, 2.1, 3.0], dtype="int64")
    assert t_forced_int.dtype == "int64"
    assert t_forced_int.tolist() == [1, 2, 3]


@pytest.mark.parametrize("backend_name", available_backends())
def test_non_differentiable_dtype_invariants(backend_name):
    set_backend(backend_name)

    # Setting requires_grad=True on integer or boolean tensor must raise ValueError
    with pytest.raises(ValueError, match="Only Tensors with floating point dtype"):
        Tensor([1, 2, 3], dtype="int64", requires_grad=True)

    with pytest.raises(ValueError, match="Only Tensors with floating point dtype"):
        Tensor([True, False], dtype="bool", requires_grad=True)

    with pytest.raises(ValueError, match="Unsupported dtype"):
        Tensor([1.0], dtype="float16")


@pytest.mark.parametrize("backend_name", available_backends())
def test_dtype_factory_functions_and_item(backend_name):
    set_backend(backend_name)

    z_int = zeros((2, 3), dtype="int64")
    assert z_int.dtype == "int64"
    assert z_int.tolist() == [[0, 0, 0], [0, 0, 0]]

    o_bool = ones((2, 2), dtype="bool")
    assert o_bool.dtype == "bool"
    assert o_bool.tolist() == [[True, True], [True, True]]

    z_like = zeros_like(o_bool)
    assert z_like.dtype == "bool"
    assert z_like.tolist() == [[False, False], [False, False]]

    # Scalar item() type fidelity
    s_int = Tensor(42, dtype="int64")
    assert s_int.item() == 42
    assert isinstance(s_int.item(), int)

    s_bool = Tensor(True, dtype="bool")
    assert s_bool.item() is True
    assert isinstance(s_bool.item(), bool)


def test_cross_backend_dtype_preservation():
    if "numpy" not in available_backends():
        pytest.skip("NumPy backend not available")

    # Create int64 tensor on Python backend and move to NumPy backend
    set_backend("python")
    t_py = Tensor([[1, 2], [3, 4]], dtype="int64")

    t_np = t_py.to("numpy")
    assert t_np.dtype == "int64"
    assert t_np.backend.name == "numpy"
    assert t_np.tolist() == [[1, 2], [3, 4]]
