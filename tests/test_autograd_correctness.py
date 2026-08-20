"""
tests/test_autograd_correctness.py
==================================
Comprehensive test suite validating P0 & P1 Autograd Correctness & Integrity:
  1. 1D Vector Dot-Product Upstream Gradient Respect
  2. Optimizer step & state_dict monotonic version invalidation of stale graphs
  3. Intermediate tensor graph-freed propagation & second-backward guards
  4. no_grad zero closure allocation and detachment
  5. max() tie subgradient mass conservation
  6. sqrt() negative NaN domain behavior & exact derivative
  7. int64 64-bit two's complement overflow parity
  8. Shared module parameter deduplication & traversal
  9. Poisoned graph protection on backward failure
"""

import math
import pytest
from termux_train import Tensor, nn, optim, no_grad, set_backend, available_backends


# =============================================================================
# 1. P0-1: 1D @ 1D Vector Dot Product Upstream Gradient
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_dot_product_respects_explicit_scalar_upstream_gradient(backend_name):
    set_backend(backend_name)
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)
    y = a @ b

    # Upstream gradient 2.0
    y.backward(Tensor(2.0, backend=y.backend))

    # dL/da = (dL/dy) * b = 2.0 * [3.0, 4.0] = [6.0, 8.0]
    # dL/db = (dL/dy) * a = 2.0 * [1.0, 2.0] = [2.0, 4.0]
    assert a.grad.tolist() == pytest.approx([6.0, 8.0])
    assert b.grad.tolist() == pytest.approx([2.0, 4.0])


# =============================================================================
# 2. P0-2: Optimizer Step & Load State Dict Monotonic Version Invalidation
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_optimizer_step_invalidates_existing_graph(backend_name):
    set_backend(backend_name)
    w = nn.Parameter(Tensor([2.0]))
    opt = optim.SGD([w], lr=0.1)

    loss = (w * w).sum()
    w.grad = Tensor([1.0], backend=w.backend)

    # Optimizer step mutates w._data and increments w._version
    opt.step()

    # Forward graph was built with old version of w -> must fail with RuntimeError
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


@pytest.mark.parametrize("backend_name", available_backends())
def test_load_state_dict_invalidates_existing_graph(backend_name):
    set_backend(backend_name)
    model = nn.Linear(1, 1, bias=False)
    x = Tensor([[2.0]])
    loss = model(x).sum()

    state = model.state_dict()
    state["weight"][0][0] += 1.0
    model.load_state_dict(state)

    # load_state_dict mutates model weights -> must fail with RuntimeError
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


# =============================================================================
# 3. P0-4: Intermediate Tensor Graph-Freed Propagation
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_intermediate_tensor_rejects_backward_after_parent_graph_freed(backend_name):
    set_backend(backend_name)
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * x
    loss = y.sum()

    # First backward frees the graph across all nodes
    loss.backward()
    assert x.grad.tolist() == [2.0, 4.0]

    # Intermediate node y must also know graph is freed
    with pytest.raises(RuntimeError, match="freed"):
        y.backward(Tensor([1.0, 1.0], backend=y.backend))


# =============================================================================
# 4. P1-5: no_grad Zero Closure & Function Hook Retention
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_no_grad_output_does_not_retain_grad_fn(backend_name):
    set_backend(backend_name)
    x = Tensor([1.0, 2.0], requires_grad=True)

    with no_grad():
        y = x * 2.0
        assert y.requires_grad is False
        assert len(y._prev) == 0
        assert y._backward is None
        assert y._grad_fn_state in ("leaf", "freed")


# =============================================================================
# 5. P1-1: max() Tie Subgradient Mass Conservation
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_max_tied_values_gradient_conserves_upstream_mass(backend_name):
    set_backend(backend_name)

    # 1D Tie: [2.0, 2.0]
    x = Tensor([2.0, 2.0], requires_grad=True)
    x.max().backward()
    assert sum(x.grad.tolist()) == pytest.approx(1.0)
    assert x.grad.tolist() == pytest.approx([0.5, 0.5])

    # 2D Tied Reduction along axis=-1: [[3.0, 3.0, 1.0]]
    x2 = Tensor([[3.0, 3.0, 1.0]], requires_grad=True)
    x2.max(axis=-1).sum().backward()
    assert sum(x2.grad.tolist()[0]) == pytest.approx(1.0)
    assert x2.grad.tolist()[0] == pytest.approx([0.5, 0.5, 0.0])


# =============================================================================
# 6. P1-2: sqrt() Negative Domain & Exact Gradient
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_sqrt_negative_is_nan(backend_name):
    set_backend(backend_name)
    out = Tensor([-1.0]).sqrt()
    assert math.isnan(out.tolist()[0])

    z = Tensor([4.0], requires_grad=True)
    w = z.sqrt()
    w.sum().backward()
    # d/dx(sqrt(4)) = 0.5 / sqrt(4) = 0.25
    assert z.grad.tolist() == pytest.approx([0.25])


# =============================================================================
# 7. P1-4: int64 Two's Complement Overflow Parity
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_int64_overflow_parity(backend_name):
    set_backend(backend_name)
    # (2**62) * 4 overflows signed 64-bit int
    # 2**64 wraps to 0 in 64-bit two's complement
    a = Tensor([1 << 62], dtype="int64")
    b = Tensor([4], dtype="int64")
    c = a * b
    assert c.dtype == "int64"
    # Result must be 0 in 64-bit two's complement modulo 2**64
    assert c.tolist()[0] == 0


# =============================================================================
# 8. P1-8: Shared Module Parameter Deduplication
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_shared_module_parameters_dedup(backend_name):
    set_backend(backend_name)
    shared = nn.Linear(4, 4)
    model = nn.Sequential(shared, shared)

    params = model.parameters()
    # shared Linear has 2 parameters (weight, bias) -> total must be exactly 2
    assert len(params) == 2

    named = model.named_parameters()
    assert len(named) == 2


# =============================================================================
# 9. P0-3: Backward Exception Graph Poisoning
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_backward_exception_poisons_graph(backend_name):
    set_backend(backend_name)
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * x
    loss = y.sum()

    # Mutate x in-place before backward to trigger failure midway
    x.data = [5.0, 6.0]

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()

    # Graph is now poisoned
    with pytest.raises(RuntimeError, match="invalid/poisoned"):
        loss.backward()


# =============================================================================
# 10. Leaf Usability & Backward Invariance
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_leaf_remains_usable_after_graph_backward(backend_name):
    set_backend(backend_name)
    x = Tensor(2.0, requires_grad=True)
    y = x * x
    y.backward()
    assert x.grad.item() == 4.0
    x.zero_grad()

    # Leaf x itself can be backwarded directly or used in new graph
    x.backward()
    assert x.grad.item() == 1.0
    x.zero_grad()

    z = x * 3.0
    z.backward()
    assert x.grad.item() == 3.0


# =============================================================================
# 11. ContextVar Multithread & Async Context Isolation
# =============================================================================

def test_contextvar_multithread_isolation():
    import threading
    results = {}

    def worker(name, use_no_grad):
        if use_no_grad:
            with no_grad():
                t = Tensor(1.0, requires_grad=True)
                results[name] = t.requires_grad
        else:
            t = Tensor(1.0, requires_grad=True)
            results[name] = t.requires_grad

    t1 = threading.Thread(target=worker, args=("thread_no_grad", True))
    t2 = threading.Thread(target=worker, args=("thread_grad", False))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["thread_no_grad"] is False
    assert results["thread_grad"] is True
    assert Tensor.is_grad_enabled() is True


# =============================================================================
# 12. Int64 Full Boundary Two's Complement Overflows
# =============================================================================

@pytest.mark.parametrize("backend_name", available_backends())
def test_int64_boundary_overflow_cases(backend_name):
    set_backend(backend_name)
    INT64_MAX = (1 << 63) - 1
    INT64_MIN = -(1 << 63)

    # INT64_MAX + 1 -> INT64_MIN
    t_max = Tensor([INT64_MAX], dtype="int64")
    t_one = Tensor([1], dtype="int64")
    res_add = t_max + t_one
    assert res_add.tolist()[0] == INT64_MIN

    # INT64_MIN - 1 -> INT64_MAX
    t_min = Tensor([INT64_MIN], dtype="int64")
    res_sub = t_min - t_one
    assert res_sub.tolist()[0] == INT64_MAX
