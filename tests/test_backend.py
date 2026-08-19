"""
tests/test_backend.py
=====================
Verify parity and accuracy between PythonBackend and NumPyBackend.
"""

import pytest
from termux_train.backend import PythonBackend, available_backends
from termux_train.backend.base import BaseBackend

backends_to_test = [PythonBackend()]
if "numpy" in available_backends():
    from termux_train.backend.numpy_backend import NumPyBackend
    backends_to_test.append(NumPyBackend())

@pytest.mark.parametrize("backend", backends_to_test)
def test_backend_shape_and_conversions(backend: BaseBackend):
    d_scalar = backend.from_data(3.14)
    assert backend.get_shape(d_scalar) == ()
    assert backend.to_flat_list(d_scalar) == [pytest.approx(3.14)]

    d_vec = backend.from_data([1.0, 2.0, 3.0])
    assert backend.get_shape(d_vec) == (3,)
    assert backend.to_flat_list(d_vec) == [1.0, 2.0, 3.0]

    d_mat = backend.from_data([[1.0, 2.0], [3.0, 4.0]])
    assert backend.get_shape(d_mat) == (2, 2)
    assert backend.to_flat_list(d_mat) == [1.0, 2.0, 3.0, 4.0]

@pytest.mark.parametrize("backend", backends_to_test)
def test_backend_elementwise(backend: BaseBackend):
    a = backend.from_data([1.0, 2.0, 3.0])
    b = backend.from_data([4.0, 5.0, 6.0])

    add_res = backend.add(a, b)
    assert backend.to_flat_list(add_res) == [5.0, 7.0, 9.0]

    mul_res = backend.mul(a, b)
    assert backend.to_flat_list(mul_res) == [4.0, 10.0, 18.0]

@pytest.mark.parametrize("backend", backends_to_test)
def test_backend_matmul(backend: BaseBackend):
    a = backend.from_data([[1.0, 2.0], [3.0, 4.0]])
    b = backend.from_data([[2.0, 0.0], [1.0, 2.0]])
    # [[1*2 + 2*1, 1*0 + 2*2], [3*2 + 4*1, 3*0 + 4*2]] = [[4, 4], [10, 8]]
    c = backend.matmul(a, b)
    assert backend.get_shape(c) == (2, 2)
    assert backend.to_flat_list(c) == [4.0, 4.0, 10.0, 8.0]

@pytest.mark.parametrize("backend", backends_to_test)
def test_backend_broadcasting_unbroadcast(backend: BaseBackend):
    grad = backend.from_data([[1.0, 2.0], [3.0, 4.0]]) # shape (2, 2)
    unb = backend.unbroadcast(grad, (2, 1))
    assert backend.get_shape(unb) == (2, 1)
    assert backend.to_flat_list(unb) == [3.0, 7.0]

    unb_scalar = backend.unbroadcast(grad, ())
    assert backend.to_flat_list(unb_scalar) == [10.0]

@pytest.mark.parametrize("backend", backends_to_test)
def test_backend_matmul_shapes(backend: BaseBackend):
    # 1. 1D @ 1D -> scalar
    v1 = backend.from_data([1.0, 2.0, 3.0])
    v2 = backend.from_data([4.0, 5.0, 6.0])
    dot = backend.matmul(v1, v2)
    assert backend.to_flat_list(dot) == [pytest.approx(32.0)]

    # 2. 1D @ 2D -> (N,)
    a_1d = backend.from_data([1.0, 2.0])
    b_2d = backend.from_data([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    out_1d_2d = backend.matmul(a_1d, b_2d)
    assert backend.get_shape(out_1d_2d) == (3,)
    assert backend.to_flat_list(out_1d_2d) == [15.0, 18.0, 21.0]

    # 3. 1D @ 3D -> (B, N)
    a_1d = backend.from_data([1.0, 2.0])
    b_3d = backend.from_data([[[3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0]]])
    out_1d_3d = backend.matmul(a_1d, b_3d)
    assert backend.get_shape(out_1d_3d) == (2, 2)
    assert backend.to_flat_list(out_1d_3d) == [13.0, 16.0, 25.0, 28.0]

    # 4. 2D @ 1D -> (M,)
    m = backend.from_data([[1.0, 2.0], [3.0, 4.0]])
    v = backend.from_data([5.0, 6.0])
    mv = backend.matmul(m, v)
    assert backend.get_shape(mv) == (2,)
    assert backend.to_flat_list(mv) == [17.0, 39.0]

    # 5. 2D @ 2D -> (M, N)
    a_2d = backend.from_data([[1.0, 2.0], [3.0, 4.0]])
    b_2d = backend.from_data([[2.0, 0.0], [1.0, 2.0]])
    out_2d_2d = backend.matmul(a_2d, b_2d)
    assert backend.get_shape(out_2d_2d) == (2, 2)
    assert backend.to_flat_list(out_2d_2d) == [4.0, 4.0, 10.0, 8.0]

    # 6. 2D @ 3D -> (B, M, N)
    a_2d = backend.from_data([[1.0, 2.0], [3.0, 4.0]])
    b_3d = backend.from_data([[[1.0], [2.0]], [[3.0], [4.0]]])
    out_2d_3d = backend.matmul(a_2d, b_3d)
    assert backend.get_shape(out_2d_3d) == (2, 2, 1)
    assert backend.to_flat_list(out_2d_3d) == [5.0, 11.0, 11.0, 25.0]

    # 7. 3D @ 1D -> (B, M)
    a_3d = backend.from_data([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b_1d = backend.from_data([2.0, 3.0])
    out_3d_1d = backend.matmul(a_3d, b_1d)
    assert backend.get_shape(out_3d_1d) == (2, 2)
    assert backend.to_flat_list(out_3d_1d) == [8.0, 18.0, 28.0, 38.0]

    # 8. 3D @ 2D -> (B, M, N)
    t3d = backend.from_data([[[1.0, 2.0]], [[3.0, 4.0]]]) # (2, 1, 2)
    w2d = backend.from_data([[1.0, 0.0], [0.0, 1.0]])     # (2, 2)
    t3d_out = backend.matmul(t3d, w2d)
    assert backend.get_shape(t3d_out) == (2, 1, 2)
    assert backend.to_flat_list(t3d_out) == [1.0, 2.0, 3.0, 4.0]

    # 9. 3D @ 3D -> (B, M, N)
    a3d = backend.from_data([[[1.0, 2.0]], [[3.0, 4.0]]]) # (2, 1, 2)
    b3d = backend.from_data([[[1.0], [2.0]], [[3.0], [4.0]]]) # (2, 2, 1)
    c3d = backend.matmul(a3d, b3d)
    assert backend.get_shape(c3d) == (2, 1, 1)
    assert backend.to_flat_list(c3d) == [5.0, 25.0]

@pytest.mark.parametrize("backend", backends_to_test)
def test_backend_take(backend: BaseBackend):
    d = backend.from_data([[[1.0, 2.0]], [[3.0, 4.0]]])
    b0 = backend.take(d, 0, axis=0)
    assert backend.get_shape(b0) == (1, 2)
    assert backend.to_flat_list(b0) == [1.0, 2.0]
    b1 = backend.take(d, 1, axis=0)
    assert backend.get_shape(b1) == (1, 2)
    assert backend.to_flat_list(b1) == [3.0, 4.0]

@pytest.mark.skipif(
    "numpy" not in available_backends(),
    reason="NumPy backend is unavailable",
)
def test_matmul_all_rank_combinations_backend_parity():
    from termux_train.backend.python_backend import PythonBackend
    from termux_train.backend.numpy_backend import NumPyBackend
    python_backend = PythonBackend()
    numpy_backend = NumPyBackend()

    cases = [
        ([1.0, 2.0], [3.0, 4.0]),                                    # 1D @ 1D
        ([1.0, 2.0], [[3.0, 4.0], [5.0, 6.0]]),                      # 1D @ 2D
        ([1.0, 2.0], [[[3.0], [4.0]], [[5.0], [6.0]]]),             # 1D @ 3D
        ([[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0]),                      # 2D @ 1D
        ([[1.0, 2.0], [3.0, 4.0]], [[5.0], [6.0]]),                 # 2D @ 2D
        ([[1.0, 2.0], [3.0, 4.0]], [[[5.0], [6.0]], [[7.0], [8.0]]]), # 2D @ 3D
        ([[[1.0, 2.0]], [[3.0, 4.0]]], [5.0, 6.0]),                 # 3D @ 1D
        ([[[1.0, 2.0]], [[3.0, 4.0]]], [[5.0], [6.0]]),             # 3D @ 2D
        ([[[1.0, 2.0]], [[3.0, 4.0]]], [[[5.0], [6.0]], [[7.0], [8.0]]]), # 3D @ 3D
    ]

    for left, right in cases:
        py_result = python_backend.matmul(
            python_backend.from_data(left),
            python_backend.from_data(right),
        )
        np_result = numpy_backend.matmul(
            numpy_backend.from_data(left),
            numpy_backend.from_data(right),
        )
        py_flat = python_backend.to_flat_list(py_result)
        np_flat = numpy_backend.to_flat_list(np_result)
        assert python_backend.get_shape(py_result) == numpy_backend.get_shape(np_result)
        assert py_flat == pytest.approx(np_flat, abs=1e-6, rel=1e-6)
