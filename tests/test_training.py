"""
tests/test_training.py
======================
Integration and convergence test suite for neural network training in termux-train.
Validates end-to-end forward-backward-optimizer step convergence on XOR non-linear benchmark.
"""

import math
import random
import pytest
from termux_train import Tensor, nn, optim, available_backends, set_backend

@pytest.fixture(params=["python"] + (["numpy"] if "numpy" in available_backends() else []))
def active_backend(request):
    set_backend(request.param)
    return request.param


def test_xor_convergence_adam(active_backend):
    random.seed(42)
    x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    target = Tensor([[0.0], [1.0], [1.0], [0.0]])

    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.Tanh(),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    )
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    initial_loss = None
    final_loss = None

    for epoch in range(1, 1000):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, target)

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        if loss.item() < 0.02:
            final_loss = loss.item()
            break

    if final_loss is None:
        final_loss = loss.item()

    # Accuracy check
    preds = model(x)
    pred_vals = [row[0] for row in preds.tolist()]
    target_vals = [row[0] for row in target.tolist()]
    accuracy = sum(1 for pv, tv in zip(pred_vals, target_vals) if (1 if pv >= 0.5 else 0) == int(tv)) / 4.0

    assert final_loss < initial_loss
    assert final_loss < 0.03
    assert accuracy == 1.0

    # Ensure all parameters and grads are finite
    for p in model.parameters():
        for v in p.flatten().tolist():
            assert math.isfinite(v)


def test_xor_convergence_sgd_momentum(active_backend):
    random.seed(42)
    x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    target = Tensor([[0.0], [1.0], [1.0], [0.0]])

    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.Tanh(),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    )
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
    criterion = nn.MSELoss()

    initial_loss = None
    final_loss = None

    for epoch in range(1, 1500):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, target)

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        if loss.item() < 0.02:
            final_loss = loss.item()
            break

    if final_loss is None:
        final_loss = loss.item()

    preds = model(x)
    pred_vals = [row[0] for row in preds.tolist()]
    target_vals = [row[0] for row in target.tolist()]
    accuracy = sum(1 for pv, tv in zip(pred_vals, target_vals) if (1 if pv >= 0.5 else 0) == int(tv)) / 4.0

    assert final_loss < initial_loss
    assert final_loss < 0.03
    assert accuracy == 1.0


def test_xor_convergence_adamw(active_backend):
    random.seed(42)
    x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    target = Tensor([[0.0], [1.0], [1.0], [0.0]])

    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.Tanh(),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    )
    optimizer = optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-4)
    criterion = nn.MSELoss()

    initial_loss = None
    final_loss = None

    for epoch in range(1, 1000):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, target)

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        if loss.item() < 0.02:
            final_loss = loss.item()
            break

    if final_loss is None:
        final_loss = loss.item()

    preds = model(x)
    pred_vals = [row[0] for row in preds.tolist()]
    target_vals = [row[0] for row in target.tolist()]
    accuracy = sum(1 for pv, tv in zip(pred_vals, target_vals) if (1 if pv >= 0.5 else 0) == int(tv)) / 4.0

    assert final_loss < initial_loss
    assert final_loss < 0.03
    assert accuracy == 1.0


def test_linear_no_bias_parameter_count(active_backend):
    layer_with_bias = nn.Linear(2, 3, bias=True)
    layer_no_bias = nn.Linear(2, 3, bias=False)

    assert len(list(layer_with_bias.parameters())) == 2
    assert len(list(layer_no_bias.parameters())) == 1
    assert layer_no_bias.bias is None


def test_optimizer_backend_preservation_and_grad_lifecycle(active_backend):
    p = nn.Parameter([1.0, 2.0], requires_grad=True)
    original_backend_name = p.backend.name
    opt = optim.SGD([p], lr=0.1)

    opt.zero_grad()
    assert p.grad is None

    # Forward + Loss + Backward
    loss = (p * p).sum()
    loss.backward()

    assert p.grad is not None
    assert p.grad.backend.name == original_backend_name

    # Step should NOT clear grad
    opt.step()
    assert p.grad is not None
    assert p.backend.name == original_backend_name
    assert p.grad.backend.name == original_backend_name
