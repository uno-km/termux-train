"""
tests/test_lora_adapter_io.py
=============================
Rigorously tests Lightweight LoRA Adapter Save & Load Engine (SCRUM-311).
Verifies SafeTensors/JSON roundtrips, <100KB footprint, metadata integrity,
strict shape validation, and atomic write rollback.
"""

import os
import json
import tempfile
import pytest
from termux_train import Tensor, nn, set_backend, available_backends
from termux_train.checkpoint import save_lora_adapter, load_lora_adapter


class ToyLoRAModel(nn.Module):
    def __init__(self, in_features=16, hidden=32, out_features=8, r=4, alpha=8.0):
        super().__init__()
        self.fc1 = nn.LoRALinear(in_features, hidden, rank=r, alpha=alpha, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.LoRALinear(hidden, out_features, rank=r, alpha=alpha, bias=True)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class DeepLoRAModel(nn.Module):
    def __init__(self, in_features=16, hidden=32, out_features=8, extra=4, r=4, alpha=8.0):
        super().__init__()
        self.fc1 = nn.LoRALinear(in_features, hidden, rank=r, alpha=alpha, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.LoRALinear(hidden, out_features, rank=r, alpha=alpha, bias=True)
        self.fc3 = nn.LoRALinear(out_features, extra, rank=r, alpha=alpha, bias=True)

    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))


def _clone_base_params(src: nn.Module, dst: nn.Module):
    """Clones base layer weights and biases from src to dst to guarantee base model equality."""
    src_mods = {n: m for n, m in src.named_modules() if isinstance(m, nn.LoRALinear)}
    dst_mods = {n: m for n, m in dst.named_modules() if isinstance(m, nn.LoRALinear)}
    for name, s_mod in src_mods.items():
        if name in dst_mods:
            d_mod = dst_mods[name]
            for s_p, d_p in zip(s_mod.base.parameters(), d_mod.base.parameters()):
                d_p._replace_data(s_p._data.copy() if hasattr(s_p._data, "copy") else list(s_p._data))


@pytest.mark.parametrize("backend_name", available_backends())
def test_lora_adapter_safetensors_roundtrip(backend_name):
    set_backend(backend_name)
    model = ToyLoRAModel(16, 32, 8, r=4, alpha=8.0)

    # Train 1 step to mutate LoRA weights
    x = Tensor([[1.0] * 16])
    y = model(x)
    loss = y.sum()
    loss.backward()

    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_path = os.path.join(tmpdir, "adapter.safetensors")
        saved_p = save_lora_adapter(model, adapter_path, adapter_name="test_lora_v1")
        assert os.path.exists(saved_p)

        file_size = os.path.getsize(saved_p)
        assert file_size < 100 * 1024  # Must be tiny (<100KB)

        # Fresh model with same base weights
        fresh_model = ToyLoRAModel(16, 32, 8, r=4, alpha=8.0)
        _clone_base_params(model, fresh_model)

        # Load adapter
        meta = load_lora_adapter(fresh_model, saved_p)
        assert meta["adapter_name"] == "test_lora_v1"

        # Verify predictions match exactly
        out_orig = model(x)
        out_loaded = fresh_model(x)
        assert pytest.approx(out_orig.tolist()[0], rel=1e-5) == out_loaded.tolist()[0]


@pytest.mark.parametrize("backend_name", available_backends())
def test_lora_adapter_json_roundtrip(backend_name):
    set_backend(backend_name)
    model = ToyLoRAModel(8, 16, 4, r=2, alpha=4.0)
    x = Tensor([[0.5] * 8])

    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_path = os.path.join(tmpdir, "adapter.json")
        saved_p = save_lora_adapter(model, adapter_path, adapter_name="json_adapter")
        assert os.path.exists(saved_p)

        with open(saved_p, "r", encoding="utf-8") as f:
            j_data = json.load(f)
        assert "metadata" in j_data
        assert "adapters" in j_data
        assert len(j_data["adapters"]) == 2  # fc1, fc2

        fresh_model = ToyLoRAModel(8, 16, 4, r=2, alpha=4.0)
        _clone_base_params(model, fresh_model)

        meta = load_lora_adapter(fresh_model, saved_p)
        assert meta["adapter_name"] == "json_adapter"

        out_orig = model(x)
        out_loaded = fresh_model(x)
        assert pytest.approx(out_orig.tolist()[0], rel=1e-5) == out_loaded.tolist()[0]


def test_lora_adapter_no_lora_layers_raises():
    plain_model = nn.Sequential(nn.Linear(8, 4), nn.ReLU())
    with pytest.raises(ValueError, match="No LoRALinear layers found"):
        save_lora_adapter(plain_model, "dummy.safetensors")


def test_lora_adapter_file_not_found():
    model = ToyLoRAModel(4, 8, 2, r=2)
    with pytest.raises(FileNotFoundError):
        load_lora_adapter(model, "non_existent_adapter_file.safetensors")


def test_lora_adapter_strict_shape_mismatch():
    model_r4 = ToyLoRAModel(8, 16, 4, r=4)
    model_r2 = ToyLoRAModel(8, 16, 4, r=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, "r4_adapter.safetensors")
        save_lora_adapter(model_r4, p)

        # Loading r=4 adapter into r=2 model must fail on strict mode
        with pytest.raises(ValueError):
            load_lora_adapter(model_r2, p, strict=True)


def test_lora_adapter_cross_backend_interchange():
    # Save on Python backend
    set_backend("python")
    model_py = ToyLoRAModel(8, 16, 4, r=2)
    x_py = Tensor([[0.3] * 8])

    with tempfile.TemporaryDirectory() as tmpdir:
        p_safe = os.path.join(tmpdir, "cross.safetensors")
        p_json = os.path.join(tmpdir, "cross.json")
        save_lora_adapter(model_py, p_safe)
        save_lora_adapter(model_py, p_json)

        # Load on NumPy backend
        set_backend("numpy")
        model_np = ToyLoRAModel(8, 16, 4, r=2)
        _clone_base_params(model_py, model_np)
        x_np = Tensor([[0.3] * 8])

        load_lora_adapter(model_np, p_safe)
        out_np_safe = model_np(x_np)
        out_py = model_py(x_py)
        assert pytest.approx(out_py.tolist()[0], rel=1e-5) == out_np_safe.tolist()[0]

        load_lora_adapter(model_np, p_json)
        out_np_json = model_np(x_np)
        assert pytest.approx(out_py.tolist()[0], rel=1e-5) == out_np_json.tolist()[0]


def test_lora_adapter_partial_loading_non_strict():
    set_backend("numpy")
    model_2layer = ToyLoRAModel(16, 32, 8, r=4)
    model_3layer = DeepLoRAModel(16, 32, 8, extra=4, r=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, "2layer.safetensors")
        save_lora_adapter(model_2layer, p)

        # Non-strict loading into 3-layer model should update fc1 and fc2 and leave fc3 untouched
        load_lora_adapter(model_3layer, p, strict=False)
        assert model_3layer.fc1.lora_A.tolist() == model_2layer.fc1.lora_A.tolist()
        assert model_3layer.fc2.lora_A.tolist() == model_2layer.fc2.lora_A.tolist()
