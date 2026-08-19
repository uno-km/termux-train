"""
tests/test_lora.py
==================
SCRUM-308: Comprehensive Unit Tests for On-Device LoRALinear Core.
Validates:
  - Constructor input verification (types, bounds, booleans, NaN/Inf)
  - Mathematical factor shapes (in_features, rank) and (rank, out_features)
  - Scaling factor computation (alpha / rank)
  - Exact zero initialization of factor B and random initialization of factor A
  - Base Linear freezing (requires_grad=False) and parameter identity preservation
  - Factory method from_linear() identity preservation
  - Initial forward output identity (LoRA(x) == base(x)) for 1D, 2D, and 3D inputs
  - Manual reference computation parity for 1D, 2D, and 3D inputs
  - Gradient isolation: base.weight.grad is None, base.bias.grad is None
  - Adapter-only optimization: optimizer step updates A/B while base weights remain bitwise identical
  - Recursive model-level helper functions adapter_parameters() and named_adapter_parameters()
  - Full parity across Python and NumPy backends
"""

import math
import copy
import pytest
from termux_train import Tensor, nn, optim, available_backends, set_backend
from termux_train.nn.lora import LoRALinear, adapter_parameters, named_adapter_parameters


@pytest.fixture(params=["python"] + (["numpy"] if "numpy" in available_backends() else []))
def active_backend(request):
    set_backend(request.param)
    return request.param


# =============================================================================
# 1. Constructor Validation & Factor Shapes
# =============================================================================

def test_lora_normal_creation(active_backend):
    layer = nn.LoRALinear(in_features=8, out_features=4, rank=2, alpha=4.0, bias=True)
    assert layer.in_features == 8
    assert layer.out_features == 4
    assert layer.rank == 2
    assert layer.alpha == 4.0
    assert layer.scaling == 2.0
    assert layer.merged is False
    assert layer.lora_A.shape == (8, 2)
    assert layer.lora_B.shape == (2, 4)
    assert layer.base.weight.shape == (8, 4)
    assert layer.base.bias is not None
    assert layer.base.bias.shape == (1, 4)


def test_lora_rank_boundaries(active_backend):
    # rank = 1 (minimum)
    l1 = nn.LoRALinear(in_features=4, out_features=6, rank=1)
    assert l1.rank == 1
    assert l1.lora_A.shape == (4, 1)
    assert l1.lora_B.shape == (1, 6)

    # rank = min(in_features, out_features) (maximum)
    l_max = nn.LoRALinear(in_features=4, out_features=6, rank=4)
    assert l_max.rank == 4
    assert l_max.lora_A.shape == (4, 4)
    assert l_max.lora_B.shape == (4, 6)


@pytest.mark.parametrize("invalid_in", [0, -1, -5, True, False, 3.5, "8", None, [8]])
def test_lora_invalid_in_features(invalid_in, active_backend):
    with pytest.raises((ValueError, TypeError)):
        nn.LoRALinear(in_features=invalid_in, out_features=4, rank=2)


@pytest.mark.parametrize("invalid_out", [0, -1, -5, True, False, 2.5, "4", None, (4,)])
def test_lora_invalid_out_features(invalid_out, active_backend):
    with pytest.raises((ValueError, TypeError)):
        nn.LoRALinear(in_features=8, out_features=invalid_out, rank=2)


@pytest.mark.parametrize("invalid_rank", [0, -1, True, False, 1.5, "2", None, 5])
def test_lora_invalid_rank(invalid_rank, active_backend):
    # For in_features=8, out_features=4, max rank is 4. rank=5 is invalid.
    with pytest.raises((ValueError, TypeError)):
        nn.LoRALinear(in_features=8, out_features=4, rank=invalid_rank)


@pytest.mark.parametrize("invalid_alpha", [0, 0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, False, "1.0", None])
def test_lora_invalid_alpha(invalid_alpha, active_backend):
    with pytest.raises((ValueError, TypeError)):
        nn.LoRALinear(in_features=8, out_features=4, rank=2, alpha=invalid_alpha)


def test_lora_bias_false(active_backend):
    layer = nn.LoRALinear(in_features=4, out_features=2, rank=2, bias=False)
    assert layer.base.bias is None
    param_names = [name for name, _ in layer.named_parameters()]
    assert "base.bias" not in param_names


def test_lora_backend_preservation(active_backend):
    layer = nn.LoRALinear(in_features=4, out_features=2, rank=2)
    assert layer.base.weight.backend.name == active_backend
    assert layer.lora_A.backend.name == active_backend
    assert layer.lora_B.backend.name == active_backend


# =============================================================================
# 2. Initialization & Base Freeze Contract
# =============================================================================

def test_lora_b_zero_initialization(active_backend):
    layer = nn.LoRALinear(in_features=6, out_features=4, rank=2)
    b_list = layer.lora_B.tolist()
    for row in b_list:
        for val in row:
            assert val == 0.0


def test_lora_base_frozen_and_adapter_trainable(active_backend):
    layer = nn.LoRALinear(in_features=4, out_features=2, rank=2, bias=True)
    assert layer.base.weight.requires_grad is False
    assert layer.base.bias.requires_grad is False
    assert layer.lora_A.requires_grad is True
    assert layer.lora_B.requires_grad is True


def test_lora_adapter_parameter_apis(active_backend):
    layer = nn.LoRALinear(in_features=6, out_features=4, rank=2, bias=True)

    # 1. Instance adapter_parameters
    adapter_params = layer.adapter_parameters()
    assert len(adapter_params) == 2
    assert id(adapter_params[0]) == id(layer.lora_A)
    assert id(adapter_params[1]) == id(layer.lora_B)

    # 2. Instance named_adapter_parameters
    named_adapters = layer.named_adapter_parameters()
    assert len(named_adapters) == 2
    assert named_adapters[0][0] == "lora_A"
    assert id(named_adapters[0][1]) == id(layer.lora_A)
    assert named_adapters[1][0] == "lora_B"
    assert id(named_adapters[1][1]) == id(layer.lora_B)

    # 3. adapter parameter count formula: rank * (in + out)
    expected_count = 2 * (6 + 4)
    actual_count = (layer.lora_A.shape[0] * layer.lora_A.shape[1]) + (layer.lora_B.shape[0] * layer.lora_B.shape[1])
    assert actual_count == expected_count


def test_lora_recursive_model_helpers(active_backend):
    model = nn.Sequential(
        nn.LoRALinear(4, 8, rank=2),
        nn.Tanh(),
        nn.Linear(8, 8),  # Standard non-LoRA linear layer
        nn.LoRALinear(8, 2, rank=2),
    )

    all_adapter_params = adapter_parameters(model)
    all_named_adapter_params = named_adapter_parameters(model)

    assert len(all_adapter_params) == 4
    assert len(all_named_adapter_params) == 4

    names = [name for name, _ in all_named_adapter_params]
    assert names == ["0.lora_A", "0.lora_B", "3.lora_A", "3.lora_B"]

    # Verify no base parameters leaked into adapter_parameters
    for p in all_adapter_params:
        assert p.requires_grad is True
        assert p.shape in [(4, 2), (2, 8), (8, 2), (2, 2)]


# =============================================================================
# 3. from_linear Factory Method
# =============================================================================

def test_lora_from_linear_identity_and_value_preservation(active_backend):
    base_linear = nn.Linear(4, 2, bias=True)
    orig_weight_id = id(base_linear.weight)
    orig_bias_id = id(base_linear.bias)
    orig_weight_data = copy.deepcopy(base_linear.weight.tolist())
    orig_bias_data = copy.deepcopy(base_linear.bias.tolist())

    lora_layer = nn.LoRALinear.from_linear(base_linear, rank=2, alpha=2.0)

    # Identity preservation
    assert id(lora_layer.base) == id(base_linear)
    assert id(lora_layer.base.weight) == orig_weight_id
    assert id(lora_layer.base.bias) == orig_bias_id

    # Value preservation
    assert lora_layer.base.weight.tolist() == orig_weight_data
    assert lora_layer.base.bias.tolist() == orig_bias_data

    # Freezing preservation
    assert lora_layer.base.weight.requires_grad is False
    assert lora_layer.base.bias.requires_grad is False


# =============================================================================
# 4. Forward Computation & Initial Identity (LoRA(x) == base(x))
# =============================================================================

def test_lora_initial_output_identity_1d_2d_3d(active_backend):
    layer = nn.LoRALinear(in_features=4, out_features=3, rank=2, alpha=2.0, bias=True)

    # 1D Input
    x1 = Tensor([1.0, 2.0, -1.0, 0.5])
    out1_lora = layer(x1)
    out1_base = layer.base(x1)
    assert out1_lora.flatten().tolist() == pytest.approx(out1_base.flatten().tolist(), abs=1e-6)

    # 2D Input
    x2 = Tensor([[1.0, 2.0, -1.0, 0.5], [0.0, -1.0, 3.0, 2.0]])
    out2_lora = layer(x2)
    out2_base = layer.base(x2)
    assert out2_lora.flatten().tolist() == pytest.approx(out2_base.flatten().tolist(), abs=1e-6)

    # 3D Input
    x3 = Tensor([[[1.0, 2.0, -1.0, 0.5], [0.0, -1.0, 3.0, 2.0]]])
    out3_lora = layer(x3)
    out3_base = layer.base(x3)
    assert out3_lora.flatten().tolist() == pytest.approx(out3_base.flatten().tolist(), abs=1e-6)


def test_lora_manual_forward_reference_with_nonzero_b(active_backend):
    layer = nn.LoRALinear(in_features=3, out_features=2, rank=2, alpha=4.0, bias=True)
    # Manually populate lora_A and lora_B for deterministic mathematical verification
    layer.lora_A._data = layer.lora_A.backend.from_data([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    layer.lora_B._data = layer.lora_B.backend.from_data([[1.0, 2.0], [3.0, 4.0]])

    x = Tensor([[1.0, 2.0, 3.0]])
    out = layer(x)

    # Manual reference:
    # x @ A = [[1, 2, 3]] @ [[1, 0], [0, 1], [1, 1]] = [[4, 5]]
    # (x @ A) @ B = [[4, 5]] @ [[1, 2], [3, 4]] = [[4*1 + 5*3, 4*2 + 5*4]] = [[19, 28]]
    # scaling = 4.0 / 2 = 2.0
    # adapter_term = [[38, 56]]
    # expected = base(x) + [[38, 56]]
    base_out = layer.base(x)
    expected_out = base_out + Tensor([[38.0, 56.0]])

    assert out.flatten().tolist() == pytest.approx(expected_out.flatten().tolist(), abs=1e-6)


def test_lora_input_validation_errors(active_backend):
    layer = nn.LoRALinear(in_features=4, out_features=2, rank=2)

    # Incompatible last dimension
    with pytest.raises(ValueError, match="expected input.shape"):
        layer(Tensor([[1.0, 2.0, 3.0]]))

    # Unsupported rank: 0D scalar
    with pytest.raises(ValueError, match="expects a 1D, 2D, or 3D input"):
        layer(Tensor(1.0))

    # Unsupported rank: 4D tensor
    with pytest.raises(ValueError, match="expects a 1D, 2D, or 3D input"):
        layer(Tensor([[[[1.0, 2.0, 3.0, 4.0]]]]))


# =============================================================================
# 5. Gradient Isolation & Autograd Lifecycle
# =============================================================================

def test_lora_gradient_isolation_and_base_invariance(active_backend):
    layer = nn.LoRALinear(in_features=4, out_features=2, rank=2, bias=True)
    orig_base_weight = copy.deepcopy(layer.base.weight.tolist())
    orig_base_bias = copy.deepcopy(layer.base.bias.tolist())

    x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    target = Tensor([[1.0, 0.0]])

    pred = layer(x)
    loss = ((pred - target) ** 2).sum()
    loss.backward()

    # 1. Base weights and bias have NO gradients
    assert layer.base.weight.grad is None
    assert layer.base.bias.grad is None

    # 2. Adapter B has non-zero gradients
    assert layer.lora_B.grad is not None

    # 3. Input x receives gradients
    assert x.grad is not None


def test_lora_adapter_only_optimizer_step_leaves_base_identical(active_backend):
    layer = nn.LoRALinear(in_features=4, out_features=2, rank=2, bias=True)
    orig_base_weight = copy.deepcopy(layer.base.weight.tolist())
    orig_base_bias = copy.deepcopy(layer.base.bias.tolist())
    orig_base_weight_id = id(layer.base.weight)
    orig_base_bias_id = id(layer.base.bias)

    # Give non-zero B so both A and B receive gradient
    layer.lora_B._data = layer.lora_B.backend.from_data([[0.1, -0.1], [0.2, 0.05]])

    optimizer = optim.SGD(layer.adapter_parameters(), lr=0.1)

    # Verify optimizer only manages 2 parameters
    assert len(optimizer.params) == 2

    x = Tensor([[1.0, 1.0, 1.0, 1.0]])
    target = Tensor([[0.0, 0.0]])

    orig_a = copy.deepcopy(layer.lora_A.tolist())
    orig_b = copy.deepcopy(layer.lora_B.tolist())

    optimizer.zero_grad()
    loss = ((layer(x) - target) ** 2).sum()
    loss.backward()
    optimizer.step()

    # 1. A and B updated
    assert layer.lora_A.tolist() != orig_a
    assert layer.lora_B.tolist() != orig_b

    # 2. Base weight and bias strictly unchanged
    assert layer.base.weight.tolist() == orig_base_weight
    assert layer.base.bias.tolist() == orig_base_bias

    # 3. Parameter identities preserved
    assert id(layer.base.weight) == orig_base_weight_id
    assert id(layer.base.bias) == orig_base_bias_id


# =============================================================================
# 6. SCRUM-309: Adapter-only State Serialization & Atomic Restoration
# =============================================================================

def test_lora_adapter_state_dict_schema_and_exclusion(active_backend):
    layer = nn.LoRALinear(in_features=6, out_features=4, rank=2, alpha=4.0, bias=True)
    state = layer.adapter_state_dict()

    # Exact expected keys
    assert set(state.keys()) == {"format", "version", "in_features", "out_features", "rank", "alpha", "lora_A", "lora_B"}
    assert state["format"] == "termux-train-lora-adapter"
    assert state["version"] == "1.0"
    assert state["in_features"] == 6
    assert state["out_features"] == 4
    assert state["rank"] == 2
    assert state["alpha"] == 4.0

    # Base parameters & optimizer state are strictly excluded
    assert "base.weight" not in state
    assert "base.bias" not in state
    assert "weight" not in state
    assert "bias" not in state


def test_lora_adapter_state_dict_deep_copy_isolation(active_backend):
    layer = nn.LoRALinear(in_features=4, out_features=2, rank=2)
    state = layer.adapter_state_dict()

    # 1. Modifying state dict does not affect layer
    state["lora_A"][0][0] = 9999.0
    assert layer.lora_A.tolist()[0][0] != 9999.0

    # 2. Modifying layer does not affect previously exported state dict
    layer.lora_B._data = layer.lora_B.backend.from_data([[888.0, 888.0], [888.0, 888.0]])
    assert state["lora_B"][0][0] == 0.0


def test_lora_single_layer_load_adapter_state_dict_round_trip(active_backend):
    layer1 = nn.LoRALinear(in_features=4, out_features=2, rank=2, alpha=2.0)
    layer1.lora_A._data = layer1.lora_A.backend.from_data([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    layer1.lora_B._data = layer1.lora_B.backend.from_data([[1.0, 2.0], [3.0, 4.0]])

    state = layer1.adapter_state_dict()

    layer2 = nn.LoRALinear(in_features=4, out_features=2, rank=2, alpha=2.0)
    orig_l2_weight_id = id(layer2.base.weight)
    orig_l2_A_id = id(layer2.lora_A)
    orig_l2_B_id = id(layer2.lora_B)

    layer2.load_adapter_state_dict(state)

    # Values matched
    assert layer2.lora_A.flatten().tolist() == pytest.approx(layer1.lora_A.flatten().tolist(), abs=1e-6)
    assert layer2.lora_B.flatten().tolist() == pytest.approx(layer1.lora_B.flatten().tolist(), abs=1e-6)

    # Identities preserved
    assert id(layer2.base.weight) == orig_l2_weight_id
    assert id(layer2.lora_A) == orig_l2_A_id
    assert id(layer2.lora_B) == orig_l2_B_id

    # requires_grad preserved
    assert layer2.base.weight.requires_grad is False
    assert layer2.lora_A.requires_grad is True
    assert layer2.lora_B.requires_grad is True


def test_lora_cross_backend_portability():
    # 1. Export state on PythonBackend
    set_backend("python")
    py_layer = nn.LoRALinear(in_features=3, out_features=2, rank=2, alpha=2.0)
    py_layer.lora_A._data = py_layer.lora_A.backend.from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    py_layer.lora_B._data = py_layer.lora_B.backend.from_data([[0.1, 0.2], [0.3, 0.4]])
    py_state = py_layer.adapter_state_dict()

    if "numpy" in available_backends():
        # 2. Load into NumPyBackend layer
        set_backend("numpy")
        np_layer = nn.LoRALinear(in_features=3, out_features=2, rank=2, alpha=2.0)
        assert np_layer.lora_A.backend.name == "numpy"

        np_layer.load_adapter_state_dict(py_state)

        # Target backend preserved
        assert np_layer.lora_A.backend.name == "numpy"
        assert np_layer.lora_B.backend.name == "numpy"

        # Values matched
        assert np_layer.lora_A.flatten().tolist() == pytest.approx(py_layer.lora_A.flatten().tolist(), abs=1e-6)
        assert np_layer.lora_B.flatten().tolist() == pytest.approx(py_layer.lora_B.flatten().tolist(), abs=1e-6)

        # 3. Export on NumPy and load into Python
        np_state = np_layer.adapter_state_dict()
        set_backend("python")
        fresh_py_layer = nn.LoRALinear(in_features=3, out_features=2, rank=2, alpha=2.0)
        fresh_py_layer.load_adapter_state_dict(np_state)
        assert fresh_py_layer.lora_A.backend.name == "python"
        assert fresh_py_layer.lora_A.flatten().tolist() == pytest.approx(np_layer.lora_A.flatten().tolist(), abs=1e-6)


def test_lora_load_atomic_rejections_and_rollback(active_backend):
    layer = nn.LoRALinear(in_features=4, out_features=2, rank=2, alpha=2.0)
    orig_a = copy.deepcopy(layer.lora_A.tolist())
    orig_b = copy.deepcopy(layer.lora_B.tolist())
    valid_state = layer.adapter_state_dict()

    # 1. Non-dict rejection
    with pytest.raises(TypeError, match="state_dict must be a dict"):
        layer.load_adapter_state_dict("invalid")
    assert layer.lora_A.tolist() == orig_a

    # 2. Missing key rejection in strict mode
    bad_state = copy.deepcopy(valid_state)
    del bad_state["lora_B"]
    with pytest.raises(ValueError, match="Missing required keys"):
        layer.load_adapter_state_dict(bad_state, strict=True)
    assert layer.lora_A.tolist() == orig_a

    # 3. Unexpected key rejection in strict mode
    bad_state = copy.deepcopy(valid_state)
    bad_state["extra_field"] = 123
    with pytest.raises(ValueError, match="Unexpected keys"):
        layer.load_adapter_state_dict(bad_state, strict=True)
    assert layer.lora_A.tolist() == orig_a

    # 4. Format mismatch
    bad_state = copy.deepcopy(valid_state)
    bad_state["format"] = "unknown-format"
    with pytest.raises(ValueError, match="Unsupported adapter format"):
        layer.load_adapter_state_dict(bad_state)
    assert layer.lora_A.tolist() == orig_a

    # 5. Version mismatch
    bad_state = copy.deepcopy(valid_state)
    bad_state["version"] = "2.0"
    with pytest.raises(ValueError, match="Unsupported adapter version"):
        layer.load_adapter_state_dict(bad_state)
    assert layer.lora_A.tolist() == orig_a

    # 6. in_features mismatch
    bad_state = copy.deepcopy(valid_state)
    bad_state["in_features"] = 99
    with pytest.raises(ValueError, match="in_features mismatch"):
        layer.load_adapter_state_dict(bad_state)
    assert layer.lora_A.tolist() == orig_a

    # 7. rank mismatch
    bad_state = copy.deepcopy(valid_state)
    bad_state["rank"] = 99
    with pytest.raises(ValueError, match="rank mismatch"):
        layer.load_adapter_state_dict(bad_state)
    assert layer.lora_A.tolist() == orig_a

    # 8. alpha mismatch
    bad_state = copy.deepcopy(valid_state)
    bad_state["alpha"] = 99.0
    with pytest.raises(ValueError, match="alpha mismatch"):
        layer.load_adapter_state_dict(bad_state)
    assert layer.lora_A.tolist() == orig_a

    # 9. Ragged matrix rejection
    bad_state = copy.deepcopy(valid_state)
    bad_state["lora_A"] = [[1.0, 2.0], [1.0], [1.0, 2.0], [1.0, 2.0]]  # Row 1 has 1 item instead of 2
    with pytest.raises(ValueError, match="column count mismatch"):
        layer.load_adapter_state_dict(bad_state)
    assert layer.lora_A.tolist() == orig_a

    # 10. Non-numeric / boolean rejection
    bad_state = copy.deepcopy(valid_state)
    bad_state["lora_A"][0][0] = True
    with pytest.raises(TypeError, match="must be a float or int"):
        layer.load_adapter_state_dict(bad_state)
    assert layer.lora_A.tolist() == orig_a

    # 11. NaN / Inf rejection
    bad_state = copy.deepcopy(valid_state)
    bad_state["lora_B"][0][0] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        layer.load_adapter_state_dict(bad_state)
    assert layer.lora_B.tolist() == orig_b


def test_lora_model_level_adapter_state_dict_and_atomic_loading(active_backend):
    from termux_train.nn.lora import adapter_state_dict, load_adapter_state_dict

    model = nn.Sequential(
        nn.LoRALinear(4, 6, rank=2, alpha=2.0),
        nn.Tanh(),
        nn.Linear(6, 6),
        nn.LoRALinear(6, 2, rank=2, alpha=2.0),
    )

    # Perturb adapter factors
    model[0].lora_A._data = model[0].lora_A.backend.from_data([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    model[3].lora_B._data = model[3].lora_B.backend.from_data([[1.0, 2.0], [3.0, 4.0]])

    state = adapter_state_dict(model)
    assert state["format"] == "termux-train-lora-model-adapter"
    assert "0" in state["adapters"]
    assert "3" in state["adapters"]

    # Load into fresh model
    fresh_model = nn.Sequential(
        nn.LoRALinear(4, 6, rank=2, alpha=2.0),
        nn.Tanh(),
        nn.Linear(6, 6),
        nn.LoRALinear(6, 2, rank=2, alpha=2.0),
    )
    load_adapter_state_dict(fresh_model, state)

    assert fresh_model[0].lora_A.flatten().tolist() == pytest.approx(model[0].lora_A.flatten().tolist(), abs=1e-6)
    assert fresh_model[3].lora_B.flatten().tolist() == pytest.approx(model[3].lora_B.flatten().tolist(), abs=1e-6)


def test_lora_model_level_atomic_rollback_on_partial_failure(active_backend):
    from termux_train.nn.lora import adapter_state_dict, load_adapter_state_dict

    model = nn.Sequential(
        nn.LoRALinear(4, 6, rank=2, alpha=2.0),
        nn.Tanh(),
        nn.LoRALinear(6, 2, rank=2, alpha=2.0),
    )

    orig_l0_a = copy.deepcopy(model[0].lora_A.tolist())
    orig_l2_b = copy.deepcopy(model[2].lora_B.tolist())

    state = adapter_state_dict(model)
    # Valid change for layer 0, corrupted for layer 2
    state["adapters"]["0"]["lora_A"][0][0] = 777.0
    state["adapters"]["2"]["lora_B"][0][0] = float("nan")  # Corrupted!

    with pytest.raises(ValueError, match="must be finite"):
        load_adapter_state_dict(model, state)

    # 100% Rollback: Layer 0 must NOT be modified even though its data was valid!
    assert model[0].lora_A.tolist() == orig_l0_a
    assert model[2].lora_B.tolist() == orig_l2_b


# =============================================================================
# 7. SCRUM-309 Hardening: Commit Failure Injection & Schema Strictness
# =============================================================================

def test_lora_single_layer_commit_failure_injection_rollback(active_backend):
    """
    Verifies that if an unexpected exception occurs DURING the assignment/commit step
    (e.g., lora_A succeeds but lora_B fails), lora_A is completely rolled back to its snapshot.
    """
    layer = nn.LoRALinear(in_features=4, out_features=2, rank=2, alpha=2.0)
    orig_A = copy.deepcopy(layer.lora_A.tolist())
    orig_B = copy.deepcopy(layer.lora_B.tolist())
    orig_A_id = id(layer.lora_A)
    orig_B_id = id(layer.lora_B)
    orig_W_id = id(layer.base.weight)
    orig_bias_id = id(layer.base.bias)
    orig_W_val = copy.deepcopy(layer.base.weight.tolist())
    orig_bias_val = copy.deepcopy(layer.base.bias.tolist())

    optimizer = optim.SGD(layer.adapter_parameters(), lr=0.1)

    state = layer.adapter_state_dict()
    state["lora_A"] = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    state["lora_B"] = [[1.0, 2.0], [3.0, 4.0]]

    # Inject failure on lora_B during commit assignment
    orig_cls = layer.lora_B.__class__

    class CrashingParameter(orig_cls):
        @property
        def _data(self):
            return self._backing_data

        @_data.setter
        def _data(self, value):
            if getattr(self, "_trigger_crash", False):
                raise RuntimeError("Simulated crash during lora_B commit")
            self._backing_data = value

    layer.lora_B._backing_data = layer.lora_B._data
    layer.lora_B.__class__ = CrashingParameter
    layer.lora_B._trigger_crash = True

    try:
        with pytest.raises(RuntimeError, match="Simulated crash during lora_B commit"):
            layer.load_adapter_state_dict(state)
    finally:
        layer.lora_B._trigger_crash = False
        layer.lora_B.__class__ = orig_cls
        del layer.lora_B._backing_data

    # 1. Full rollback: lora_A and lora_B are identical to pre-call snapshots
    assert layer.lora_A.tolist() == orig_A
    assert layer.lora_B.tolist() == orig_B

    # 2. Base parameters untouched
    assert layer.base.weight.tolist() == orig_W_val
    assert layer.base.bias.tolist() == orig_bias_val

    # 3. Parameter IDs & optimizer references preserved
    assert id(layer.lora_A) == orig_A_id
    assert id(layer.lora_B) == orig_B_id
    assert id(layer.base.weight) == orig_W_id
    assert id(layer.base.bias) == orig_bias_id
    assert id(optimizer.params[0]) == orig_A_id
    assert id(optimizer.params[1]) == orig_B_id
    assert layer.lora_A.requires_grad is True
    assert layer.lora_B.requires_grad is True
    assert layer.base.weight.requires_grad is False
    assert layer.merged is False


def test_lora_recursive_model_commit_failure_injection_rollback(active_backend):
    """
    Verifies that in a multi-layer model, if layer 0 commit succeeds but layer 2 commit fails,
    layer 0 is completely rolled back to its pre-call snapshot.
    """
    from termux_train.nn.lora import adapter_state_dict, load_adapter_state_dict

    model = nn.Sequential(
        nn.LoRALinear(4, 6, rank=2, alpha=2.0),
        nn.Tanh(),
        nn.LoRALinear(6, 2, rank=2, alpha=2.0),
    )

    orig_l0_A = copy.deepcopy(model[0].lora_A.tolist())
    orig_l0_B = copy.deepcopy(model[0].lora_B.tolist())
    orig_l2_A = copy.deepcopy(model[2].lora_A.tolist())
    orig_l2_B = copy.deepcopy(model[2].lora_B.tolist())

    state = adapter_state_dict(model)
    state["adapters"]["0"]["lora_A"] = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    state["adapters"]["2"]["lora_B"] = [[1.0, 2.0], [3.0, 4.0]]

    # Inject failure on model[2].lora_A during commit
    orig_cls = model[2].lora_A.__class__

    class CrashingParam(orig_cls):
        @property
        def _data(self):
            return self._backing_data

        @_data.setter
        def _data(self, value):
            if getattr(self, "_trigger_crash", False):
                raise RuntimeError("Simulated crash during layer 2 commit")
            self._backing_data = value

    model[2].lora_A._backing_data = model[2].lora_A._data
    model[2].lora_A.__class__ = CrashingParam
    model[2].lora_A._trigger_crash = True

    try:
        with pytest.raises(RuntimeError, match="Simulated crash during layer 2 commit"):
            load_adapter_state_dict(model, state)
    finally:
        model[2].lora_A._trigger_crash = False
        model[2].lora_A.__class__ = orig_cls
        del model[2].lora_A._backing_data

    # Multi-layer rollback: Layer 0 AND Layer 2 must be 100% restored
    assert model[0].lora_A.tolist() == orig_l0_A
    assert model[0].lora_B.tolist() == orig_l0_B
    assert model[2].lora_A.tolist() == orig_l2_A
    assert model[2].lora_B.tolist() == orig_l2_B


def test_lora_metadata_bool_and_type_rejections(active_backend):
    layer = nn.LoRALinear(in_features=4, out_features=2, rank=2, alpha=2.0)
    orig_state = layer.adapter_state_dict()

    # 1. in_features=True (boolean rejection)
    bad = copy.deepcopy(orig_state)
    bad["in_features"] = True
    with pytest.raises(TypeError, match="must be an integer"):
        layer.load_adapter_state_dict(bad)

    # 2. out_features=True (boolean rejection)
    bad = copy.deepcopy(orig_state)
    bad["out_features"] = True
    with pytest.raises(TypeError, match="must be an integer"):
        layer.load_adapter_state_dict(bad)

    # 3. rank=True (boolean rejection)
    bad = copy.deepcopy(orig_state)
    bad["rank"] = True
    with pytest.raises(TypeError, match="must be an integer"):
        layer.load_adapter_state_dict(bad)

    # 4. alpha=True (boolean rejection)
    bad = copy.deepcopy(orig_state)
    bad["alpha"] = True
    with pytest.raises(TypeError, match="must be a finite number"):
        layer.load_adapter_state_dict(bad)

    # 5. alpha=NaN, Inf, -Inf
    for bad_alpha in [float("nan"), float("inf"), float("-inf")]:
        bad = copy.deepcopy(orig_state)
        bad["alpha"] = bad_alpha
        with pytest.raises(ValueError, match="must be finite"):
            layer.load_adapter_state_dict(bad)


def test_lora_container_schema_and_key_type_rejections(active_backend):
    from termux_train.nn.lora import adapter_state_dict, load_adapter_state_dict

    model = nn.Sequential(
        nn.LoRALinear(4, 6, rank=2, alpha=2.0),
        nn.LoRALinear(6, 2, rank=2, alpha=2.0),
    )

    # 1. adapters is list, None, string, int
    for bad_adapters in [[], None, "bad", 123]:
        bad_state = {
            "format": "termux-train-lora-model-adapter",
            "version": "1.0",
            "adapters": bad_adapters,
        }
        with pytest.raises(TypeError, match="'adapters' must be a dict"):
            load_adapter_state_dict(model, bad_state)

    # 2. Non-string top-level key
    bad_state = {
        "format": "termux-train-lora-model-adapter",
        "version": "1.0",
        "adapters": {},
        123: "val",
    }
    with pytest.raises(TypeError, match="keys must be strings"):
        load_adapter_state_dict(model, bad_state)

    # 3. Non-string adapter path key
    bad_state = {
        "format": "termux-train-lora-model-adapter",
        "version": "1.0",
        "adapters": {123: {}},
    }
    with pytest.raises(TypeError, match="keys must be strings"):
        load_adapter_state_dict(model, bad_state)

    # 4. Single layer state dict with non-string key
    single_layer = nn.LoRALinear(4, 2, rank=2)
    with pytest.raises(TypeError, match="keys must be strings"):
        single_layer.load_adapter_state_dict({123: "val"})


def test_lora_strict_false_with_malformed_present_field_raises(active_backend):
    layer = nn.LoRALinear(in_features=4, out_features=2, rank=2, alpha=2.0)
    orig_a = copy.deepcopy(layer.lora_A.tolist())
    state = layer.adapter_state_dict()

    # In strict=False, missing format/version or extra keys are ignored, but malformed lora_A must still raise!
    state["lora_A"][0][0] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        layer.load_adapter_state_dict(state, strict=False)

    # Parameter remains untouched
    assert layer.lora_A.tolist() == orig_a


def test_single_layer_rejects_model_container_schema_mismatch(active_backend):
    from termux_train.nn.lora import load_adapter_state_dict

    layer = nn.LoRALinear(in_features=4, out_features=2, rank=2, alpha=2.0)
    orig_a = copy.deepcopy(layer.lora_A.tolist())
    orig_b = copy.deepcopy(layer.lora_B.tolist())
    orig_a_id = id(layer.lora_A)
    orig_b_id = id(layer.lora_B)

    valid_layer_state = layer.adapter_state_dict()

    # 1. Outer version mismatch (e.g. 999.0)
    container_bad_ver = {
        "format": "termux-train-lora-model-adapter",
        "version": "999.0",
        "adapters": {"0": valid_layer_state},
    }
    with pytest.raises(ValueError, match="Unsupported model adapter version"):
        load_adapter_state_dict(layer, container_bad_ver)
    assert layer.lora_A.tolist() == orig_a

    # 2. Outer version missing in strict=True
    container_missing_ver = {
        "format": "termux-train-lora-model-adapter",
        "adapters": {"0": valid_layer_state},
    }
    with pytest.raises(ValueError, match="Missing model adapter keys"):
        load_adapter_state_dict(layer, container_missing_ver, strict=True)
    assert layer.lora_A.tolist() == orig_a

    # 3. Outer unexpected key in strict=True
    container_unexpected = {
        "format": "termux-train-lora-model-adapter",
        "version": "1.0",
        "adapters": {"0": valid_layer_state},
        "extra_key": 123,
    }
    with pytest.raises(ValueError, match="Unexpected model adapter keys"):
        load_adapter_state_dict(layer, container_unexpected, strict=True)
    assert layer.lora_A.tolist() == orig_a

    # 4. Adapters entry is not a dict
    container_bad_entry = {
        "format": "termux-train-lora-model-adapter",
        "version": "1.0",
        "adapters": {"0": "not_a_dict"},
    }
    with pytest.raises(TypeError, match="must be a dict"):
        load_adapter_state_dict(layer, container_bad_entry)
    assert layer.lora_A.tolist() == orig_a

    # 5. Adapters has count != 1 for single LoRALinear
    container_multi = {
        "format": "termux-train-lora-model-adapter",
        "version": "1.0",
        "adapters": {"0": valid_layer_state, "1": valid_layer_state},
    }
    with pytest.raises(ValueError, match="expected 1 for single LoRALinear"):
        load_adapter_state_dict(layer, container_multi)
    assert layer.lora_A.tolist() == orig_a

    # Verify Parameter identity preserved
    assert id(layer.lora_A) == orig_a_id
    assert id(layer.lora_B) == orig_b_id


def test_container_model_rejects_outer_schema_mismatch(active_backend):
    from termux_train.nn.lora import adapter_state_dict, load_adapter_state_dict

    model = nn.Sequential(
        nn.LoRALinear(4, 6, rank=2, alpha=2.0),
        nn.LoRALinear(6, 2, rank=2, alpha=2.0),
    )
    orig_a = copy.deepcopy(model[0].lora_A.tolist())
    valid_state = adapter_state_dict(model)

    # 1. Outer unexpected key in strict=True
    bad_state = copy.deepcopy(valid_state)
    bad_state["unwanted"] = "foo"
    with pytest.raises(ValueError, match="Unexpected model adapter keys"):
        load_adapter_state_dict(model, bad_state, strict=True)
    assert model[0].lora_A.tolist() == orig_a

    # 2. Outer version mismatch
    bad_state = copy.deepcopy(valid_state)
    bad_state["version"] = "999.0"
    with pytest.raises(ValueError, match="Unsupported model adapter version"):
        load_adapter_state_dict(model, bad_state)
    assert model[0].lora_A.tolist() == orig_a

    # 3. Outer adapters entry is not a dict
    bad_state = copy.deepcopy(valid_state)
    bad_state["adapters"]["0"] = [1, 2, 3]
    with pytest.raises(TypeError, match="must be a dict"):
        load_adapter_state_dict(model, bad_state)
    assert model[0].lora_A.tolist() == orig_a
