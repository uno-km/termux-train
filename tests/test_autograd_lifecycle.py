"""
tests/test_autograd_lifecycle.py
================================
Unit tests for Big-Tech Autograd Lifecycle & Memory Architecture:
  1. no_grad context manager and decorator
  2. In-flight DAG dissection on backward (retain_graph=False default)
  3. retain_graph=True re-entrant backward support
  4. Instant memory freeing and second-backward guard
"""

import pytest
from termux_train import Tensor, no_grad, set_backend, available_backends


@pytest.mark.parametrize("backend_name", available_backends())
def test_no_grad_context_manager(backend_name):
    set_backend(backend_name)

    # 1. Normal gradient tracking
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * 2.0
    assert y.requires_grad is True
    assert len(y._prev) > 0

    # 2. Inside no_grad context
    with no_grad():
        a = Tensor([1.0, 2.0], requires_grad=True)
        assert a.requires_grad is False
        b = a * 2.0
        assert b.requires_grad is False
        assert len(b._prev) == 0

        # Existing requires_grad tensor operating inside no_grad
        c = x * 3.0
        assert c.requires_grad is False
        assert len(c._prev) == 0

    # 3. Restored after context exit
    assert Tensor.is_grad_enabled() is True
    d = x * 4.0
    assert d.requires_grad is True


@pytest.mark.parametrize("backend_name", available_backends())
def test_no_grad_decorator(backend_name):
    set_backend(backend_name)

    @no_grad()
    def evaluate(model_weight, x_val):
        return model_weight * x_val

    w = Tensor([2.0, 3.0], requires_grad=True)
    x = Tensor([4.0, 5.0])
    out = evaluate(w, x)
    assert out.requires_grad is False
    assert len(out._prev) == 0
    assert Tensor.is_grad_enabled() is True


@pytest.mark.parametrize("backend_name", available_backends())
def test_inflight_dag_dissection_and_second_backward_guard(backend_name):
    set_backend(backend_name)

    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * x
    loss = y.sum()

    # First backward frees the graph
    loss.backward()
    assert x.grad.tolist() == [2.0, 4.0]
    assert len(loss._prev) == 0

    # Second backward without retain_graph must raise RuntimeError
    with pytest.raises(RuntimeError, match="Trying to backward through the graph a second time"):
        loss.backward()


@pytest.mark.parametrize("backend_name", available_backends())
def test_retain_graph_multi_backward(backend_name):
    set_backend(backend_name)

    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * 3.0
    loss = y.sum()

    # First backward with retain_graph=True
    loss.backward(retain_graph=True)
    assert x.grad.tolist() == [3.0, 3.0]

    # Second backward accumulates gradients
    loss.backward(retain_graph=False)
    assert x.grad.tolist() == [6.0, 6.0]

    # Third backward must fail because retain_graph was False in second run
    with pytest.raises(RuntimeError, match="Trying to backward through the graph a second time"):
        loss.backward()
