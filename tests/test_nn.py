"""
tests/test_nn.py
================
Unit tests for Neural Network Mini Framework (Module, Parameter, Linear, Sequential, Activations, Loss).
"""

import pytest
from termux_train import Tensor, nn, set_backend, available_backends

@pytest.fixture(params=["python"] + (["numpy"] if "numpy" in available_backends() else []))
def active_backend(request):
    set_backend(request.param)
    return request.param

def test_parameter_creation(active_backend):
    p = nn.Parameter([[1.0, 2.0], [3.0, 4.0]])
    assert isinstance(p, Tensor)
    assert p.requires_grad is True
    assert p.shape == (2, 2)

def test_module_parameter_introspection(active_backend):
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 2)
            
        def forward(self, x):
            return self.fc2(self.fc1(x).relu())

    model = SimpleMLP()
    params = model.parameters()
    # fc1.weight, fc1.bias, fc2.weight, fc2.bias -> 4 parameters
    assert len(params) == 4
    assert model.fc1.weight.shape == (4, 8)
    assert model.fc1.bias.shape == (1, 8)
    assert model.fc2.weight.shape == (8, 2)
    assert model.fc2.bias.shape == (1, 2)

def test_linear_forward_and_backward(active_backend):
    layer = nn.Linear(2, 3)
    x = Tensor([[1.0, 2.0]], requires_grad=False)
    out = layer(x)
    assert out.shape == (1, 3)

    loss = out.sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == (2, 3)
    assert layer.bias.grad is not None
    assert layer.bias.grad.shape == (1, 3)

def test_sequential_container(active_backend):
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )
    assert len(model) == 3
    assert len(model.parameters()) == 4

    x = Tensor([[0.5, -0.5]], requires_grad=False)
    out = model(x)
    assert out.shape == (1, 1)

def test_mse_loss_backward(active_backend):
    pred = Tensor([[2.0], [4.0]], requires_grad=True)
    target = Tensor([[1.0], [1.0]], requires_grad=False)
    # diff = [[1.0], [3.0]] -> sq = [[1.0], [9.0]] -> mean = 5.0
    loss = nn.mse_loss(pred, target)
    assert loss.item() == pytest.approx(5.0)

    loss.backward()
    # d(loss)/d(pred) = (2 / N) * (pred - target) = (2 / 2) * diff = [[1.0], [3.0]]
    assert pred.grad.tolist() == [[1.0], [3.0]]

def test_state_dict_save_and_load(active_backend):
    m1 = nn.Sequential(nn.Linear(2, 2))
    m2 = nn.Sequential(nn.Linear(2, 2))

    state = m1.state_dict()
    m2.load_state_dict(state)

    x = Tensor([[1.0, 2.0]])
    assert m1(x).tolist() == m2(x).tolist()

def test_sprint3_canonical_pipeline(active_backend):
    """Exact Sprint 3 verification pipeline specified in Master Blueprint."""
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )

    x = Tensor([[0.0, 1.0]], requires_grad=False)
    pred = model(x)
    assert pred.shape == (1, 1)
    assert len(model.parameters()) == 4

def test_zero_grad_policies(active_backend):
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )

    # 1. Parameter.grad initial state is None
    for p in model.parameters():
        assert p.grad is None

    # 2. Forward and backward populates grads
    x = Tensor([[1.0, 2.0]])
    loss = model(x).sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None

    # 3. model.zero_grad() (default set_to_none=True) -> releases memory, grad is None
    model.zero_grad()
    for p in model.parameters():
        assert p.grad is None

    # 4. Backward again -> grad populated
    loss = model(x).sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None

    # 5. model.zero_grad(set_to_none=False) -> grad is zero Tensor
    model.zero_grad(set_to_none=False)
    for p in model.parameters():
        assert p.grad is not None
        for val in p.grad.backend.to_flat_list(p.grad._data):
            assert val == 0.0

def test_state_dict_deep_copy_isolation(active_backend):
    model = nn.Sequential(nn.Linear(2, 2))
    state = model.state_dict()

    orig_val = state["0.weight"][0][0]
    # Mutate underlying model parameter
    model[0].weight._data = model[0].weight.backend.from_data([[999.0, 888.0], [777.0, 666.0]])

    # Saved state_dict must remain completely unaffected
    assert state["0.weight"][0][0] == orig_val
    assert state["0.weight"][0][0] != 999.0

def test_clamp_and_log_ops(active_backend):
    t = Tensor([-5.0, 0.5, 10.0], requires_grad=True)
    c = t.clamp(min_val=0.0, max_val=1.0)
    assert c.tolist() == [0.0, 0.5, 1.0]

    loss = c.sum()
    loss.backward()
    # Gradient flows only through elements strictly within bounds [0.0, 1.0]
    assert t.grad.tolist() == [0.0, 1.0, 0.0]

    t2 = Tensor([1.0, 2.718281828], requires_grad=True)
    l = t2.log()
    assert l.tolist()[0] == pytest.approx(0.0)
    assert l.tolist()[1] == pytest.approx(1.0, abs=1e-3)

def test_bce_loss_stability(active_backend):
    # Test numerical stability with extreme boundary values (0.0 and 1.0)
    pred = Tensor([[0.0], [1.0]], requires_grad=True)
    target = Tensor([[0.0], [1.0]], requires_grad=False)

    loss = nn.bce_loss(pred, target)
    # Should not be NaN or Inf due to eps clamping
    val = loss.item()
    assert not (val != val or val == float('inf') or val == float('-inf'))
    assert val == pytest.approx(0.0, abs=1e-3)

    loss.backward()
    assert pred.grad is not None

def test_load_state_dict_errors(active_backend):
    model = nn.Sequential(nn.Linear(2, 2))

    # Unexpected key
    with pytest.raises(KeyError):
        model.load_state_dict({"0.weight": [[1.0, 2.0], [3.0, 4.0]], "0.bias": [[0.0, 0.0]], "extra": [1.0]})

    # Missing key
    with pytest.raises(KeyError):
        model.load_state_dict({"0.weight": [[1.0, 2.0], [3.0, 4.0]]})

    # Shape mismatch
    with pytest.raises(RuntimeError):
        model.load_state_dict({"0.weight": [[1.0, 2.0, 3.0]], "0.bias": [[0.0, 0.0]]})

def test_linear_accepts_1d_input(active_backend):
    layer = nn.Linear(2, 3)
    x = Tensor([1.0, 2.0], requires_grad=True)

    out = layer(x)
    assert out.shape == (3,)

    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == (2,)
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == (2, 3)
    assert layer.bias.grad is not None
    assert layer.bias.grad.shape == (1, 3)

def test_linear_accepts_3d_sequence_input(active_backend):
    layer = nn.Linear(2, 3)

    x = Tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        requires_grad=False,
    )

    out = layer(x)

    assert out.shape == (2, 2, 3)

    loss = out.sum()
    loss.backward()

    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == (2, 3)
    assert layer.bias.grad is not None
    assert layer.bias.grad.shape == (1, 3)

def test_linear_shape_errors_and_bias_false(active_backend):
    # 1. Bias False
    l_no_bias = nn.Linear(2, 3, bias=False)
    assert l_no_bias.bias is None
    x1 = Tensor([1.0, 2.0])
    out1 = l_no_bias(x1)
    assert out1.shape == (3,)
    out1.sum().backward()
    assert l_no_bias.weight.grad is not None

    # 2. Feature dimension mismatch
    l = nn.Linear(2, 3)
    with pytest.raises(ValueError):
        _ = l(Tensor([1.0, 2.0, 3.0]))

    # 3. 0D input error
    with pytest.raises(ValueError):
        _ = l(Tensor(1.0))

    # 4. 4D input is supported in N-D Linear (e.g. for Multi-Head Attention)
    out4 = l(Tensor([[[[1.0, 2.0]]]]))
    assert out4.shape == (1, 1, 1, 3)
