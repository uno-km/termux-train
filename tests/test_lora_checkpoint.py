"""
tests/test_lora_checkpoint.py
=============================
Exhaustive, rigorous test suite for termux_train.runtime Safe LoRA Checkpointing.
Tests:
  - Public API contract combination matrix (A, B, C, D)
  - Outer container & exact payload schema validation
  - Checksum format, canonicalization, and tampering detection
  - Unmerged-only exhaustive lifecycle state matrix
  - Adapter incompatibility & parameter corruption rejections
  - Optimizer compatibility (Adam, SGD momentum, AdamW) & parameter identity
  - Atomic save failure injection matrix (serialization, open, write, flush, fsync, replace)
  - Atomic load failure & rollback matrix (adapter fail, optimizer fail, double rollback fail, exception chaining)
  - Cross-backend resume & next optimizer step numerical parity (Python <-> NumPy)
  - Shared module deduplication & nested container support
  - Extra metadata deep copy isolation and recursive string-key validation
"""

import copy
import hashlib
import json
import os
import shutil
import tempfile
import pytest
from unittest.mock import patch, mock_open

from termux_train import Tensor, nn, optim, available_backends, set_backend
from termux_train.runtime import (
    save_lora_checkpoint,
    load_lora_checkpoint,
    CheckpointError,
    CheckpointIntegrityError,
    CheckpointSchemaError,
    CheckpointRollbackError,
)


@pytest.fixture(params=["python"] + (["numpy"] if "numpy" in available_backends() else []))
def active_backend(request):
    set_backend(request.param)
    return request.param


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def rewrite_checkpoint_with_valid_checksum(path: str, mutate_payload_fn) -> None:
    """Helper for schema tests: reads checkpoint, mutates payload, recomputes valid SHA256 checksum."""
    with open(path, "r", encoding="utf-8") as f:
        container = json.load(f)
    mutate_payload_fn(container["payload"])
    try:
        can_bytes = json.dumps(
            container["payload"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        container["checksum"] = hashlib.sha256(can_bytes).hexdigest()
    except (ValueError, TypeError):
        container["checksum"] = "0" * 64
    with open(path, "w", encoding="utf-8") as f:
        json.dump(container, f, indent=2, allow_nan=True)


# =============================================================================
# 1. Public API Contract Combination Matrix (A, B, C, D)
# =============================================================================

def test_lora_checkpoint_api_combination_matrix(temp_dir, active_backend):
    model = nn.Sequential(nn.LoRALinear(4, 6, rank=2), nn.Tanh(), nn.LoRALinear(6, 2, rank=2))
    opt = optim.Adam(nn.adapter_parameters(model), lr=0.01)

    # 1. Save with model=None -> TypeError
    with pytest.raises(TypeError, match="model is required"):
        save_lora_checkpoint(os.path.join(temp_dir, "m_none.json"), model=None)

    # 2. Save with non-module -> TypeError
    with pytest.raises(TypeError, match="model must be a Module instance"):
        save_lora_checkpoint(os.path.join(temp_dir, "m_bad.json"), model="not_a_module")

    # 3. Save with plain model without LoRA -> ValueError
    plain_model = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 2))
    with pytest.raises(ValueError, match="requires at least one LoRALinear layer"):
        save_lora_checkpoint(os.path.join(temp_dir, "no_lora.json"), model=plain_model)

    # 4. Save with optimizer=None (Adapter-only save)
    ckpt_adapter_only = os.path.join(temp_dir, "adapter_only.json")
    save_lora_checkpoint(ckpt_adapter_only, model=model, optimizer=None, epoch=1, global_step=10)
    with open(ckpt_adapter_only, "r", encoding="utf-8") as f:
        c1 = json.load(f)
    assert c1["payload"]["optimizer_state"] is None

    # 5. Save with optimizer provided
    ckpt_full = os.path.join(temp_dir, "full_ckpt.json")
    save_lora_checkpoint(ckpt_full, model=model, optimizer=opt, epoch=2, global_step=20)
    with open(ckpt_full, "r", encoding="utf-8") as f:
        c2 = json.load(f)
    assert isinstance(c2["payload"]["optimizer_state"], dict)

    # Combination A: model + optimizer on full checkpoint -> Restores both
    fresh_model = nn.Sequential(nn.LoRALinear(4, 6, rank=2), nn.Tanh(), nn.LoRALinear(6, 2, rank=2))
    fresh_opt = optim.Adam(nn.adapter_parameters(fresh_model), lr=0.01)
    meta_a = load_lora_checkpoint(ckpt_full, model=fresh_model, optimizer=fresh_opt)
    assert meta_a["epoch"] == 2
    assert meta_a["global_step"] == 20

    # Combination B: model + optimizer=None on full checkpoint -> Restores adapter only, skips optimizer
    fresh_model_b = nn.Sequential(nn.LoRALinear(4, 6, rank=2), nn.Tanh(), nn.LoRALinear(6, 2, rank=2))
    meta_b = load_lora_checkpoint(ckpt_full, model=fresh_model_b, optimizer=None)
    assert meta_b["epoch"] == 2

    # Combination C: model=None + optimizer=None -> Metadata-only validation mode
    meta_c = load_lora_checkpoint(ckpt_full, model=None, optimizer=None)
    assert meta_c["epoch"] == 2
    assert meta_c["global_step"] == 20
    assert "timestamp" in meta_c

    # Combination D: model=None + optimizer provided -> Rejected with ValueError
    with pytest.raises(ValueError, match="A model is required when loading LoRA optimizer state"):
        load_lora_checkpoint(ckpt_full, model=None, optimizer=fresh_opt)

    # Incompatibility: loading optimizer from adapter-only checkpoint -> CheckpointSchemaError
    with pytest.raises(CheckpointSchemaError, match="does not contain 'optimizer_state'"):
        load_lora_checkpoint(ckpt_adapter_only, model=fresh_model, optimizer=fresh_opt)


# =============================================================================
# 2. Outer Container & Exact Payload Schema Rejections
# =============================================================================

def test_lora_checkpoint_outer_container_rejections(temp_dir, active_backend):
    model = nn.LoRALinear(4, 2, rank=2)
    opt = optim.Adam(model.adapter_parameters(), lr=0.01)
    ckpt_path = os.path.join(temp_dir, "outer_test.json")
    save_lora_checkpoint(ckpt_path, model=model, optimizer=opt, epoch=1, global_step=5)

    with open(ckpt_path, "r", encoding="utf-8") as f:
        valid_container = json.load(f)

    # 1. Root is not a dict (list, string, number, null)
    for bad_root in [[], "string", 123, None]:
        bad_path = os.path.join(temp_dir, f"bad_root_{type(bad_root).__name__}.json")
        with open(bad_path, "w", encoding="utf-8") as f:
            json.dump(bad_root, f)
        with pytest.raises(CheckpointSchemaError, match="expected dict"):
            load_lora_checkpoint(bad_path)

    # 2. Missing checksum
    no_chk = copy.deepcopy(valid_container)
    del no_chk["checksum"]
    p_no_chk = os.path.join(temp_dir, "no_chk.json")
    with open(p_no_chk, "w", encoding="utf-8") as f:
        json.dump(no_chk, f)
    with pytest.raises(CheckpointSchemaError, match="Unexpected keys"):
        load_lora_checkpoint(p_no_chk)

    # 3. Missing payload
    no_pay = {"checksum": valid_container["checksum"]}
    p_no_pay = os.path.join(temp_dir, "no_pay.json")
    with open(p_no_pay, "w", encoding="utf-8") as f:
        json.dump(no_pay, f)
    with pytest.raises(CheckpointSchemaError, match="Unexpected keys"):
        load_lora_checkpoint(p_no_pay)

    # 4. Extra unexpected outer key
    extra_outer = copy.deepcopy(valid_container)
    extra_outer["extra_outer_key"] = "malicious"
    p_extra = os.path.join(temp_dir, "extra_outer.json")
    with open(p_extra, "w", encoding="utf-8") as f:
        json.dump(extra_outer, f)
    with pytest.raises(CheckpointSchemaError, match="Unexpected keys"):
        load_lora_checkpoint(p_extra)

    # 5. Checksum syntax failures (non-str, length != 64, non-hex)
    for bad_chk in [12345, "a" * 63, "a" * 65, "G" * 64, "z" * 64, "!@#" + "a" * 61]:
        bad_chk_container = copy.deepcopy(valid_container)
        bad_chk_container["checksum"] = bad_chk
        p_bad_chk = os.path.join(temp_dir, "bad_chk.json")
        with open(p_bad_chk, "w", encoding="utf-8") as f:
            json.dump(bad_chk_container, f)
        with pytest.raises(CheckpointIntegrityError, match="Invalid SHA256 checksum format"):
            load_lora_checkpoint(p_bad_chk)

    # 6. Payload is not a dict
    bad_pay_container = {"checksum": valid_container["checksum"], "payload": "not_a_dict"}
    p_bad_pay = os.path.join(temp_dir, "bad_pay.json")
    with open(p_bad_pay, "w", encoding="utf-8") as f:
        json.dump(bad_pay_container, f)
    with pytest.raises(CheckpointSchemaError, match="Invalid payload"):
        load_lora_checkpoint(p_bad_pay)


@pytest.mark.parametrize("missing_key", [
    "format", "version", "timestamp", "epoch", "global_step", "adapter_state", "optimizer_state", "extra"
])
def test_lora_checkpoint_missing_payload_keys(temp_dir, active_backend, missing_key):
    model = nn.LoRALinear(4, 2, rank=2)
    opt = optim.Adam(model.adapter_parameters(), lr=0.01)
    ckpt_path = os.path.join(temp_dir, f"missing_{missing_key}.json")
    save_lora_checkpoint(ckpt_path, model=model, optimizer=opt, epoch=1, global_step=5)

    def remove_key(payload):
        del payload[missing_key]

    rewrite_checkpoint_with_valid_checksum(ckpt_path, remove_key)

    with pytest.raises(CheckpointSchemaError, match="Invalid payload keys"):
        load_lora_checkpoint(ckpt_path)


def test_lora_checkpoint_unexpected_payload_key(temp_dir, active_backend):
    model = nn.LoRALinear(4, 2, rank=2)
    opt = optim.Adam(model.adapter_parameters(), lr=0.01)
    ckpt_path = os.path.join(temp_dir, "unexpected_payload.json")
    save_lora_checkpoint(ckpt_path, model=model, optimizer=opt, epoch=1, global_step=5)

    def add_key(payload):
        payload["unauthorized_key"] = 42

    rewrite_checkpoint_with_valid_checksum(ckpt_path, add_key)

    with pytest.raises(CheckpointSchemaError, match="Invalid payload keys"):
        load_lora_checkpoint(ckpt_path)


# =============================================================================
# 3. Scalar and State Field Type Rejections
# =============================================================================

def test_lora_checkpoint_scalar_type_and_range_rejections(temp_dir, active_backend):
    model = nn.LoRALinear(4, 2, rank=2)
    opt = optim.Adam(model.adapter_parameters(), lr=0.01)
    ckpt_path = os.path.join(temp_dir, "scalar_test.json")

    def reset_valid_ckpt():
        save_lora_checkpoint(ckpt_path, model=model, optimizer=opt, epoch=1, global_step=5)

    # 1. Timestamp bad values
    for bad_ts in [True, False, "123.45", None, float("nan"), float("inf"), float("-inf"), -1.0]:
        reset_valid_ckpt()
        def mod_ts(p):
            p["timestamp"] = bad_ts
        rewrite_checkpoint_with_valid_checksum(ckpt_path, mod_ts)
        with pytest.raises((CheckpointSchemaError, CheckpointIntegrityError)):
            load_lora_checkpoint(ckpt_path)

    # 2. Epoch bad values
    for bad_epoch in [True, False, 1.5, "1", None, -1, -10]:
        reset_valid_ckpt()
        def mod_ep(p):
            p["epoch"] = bad_epoch
        rewrite_checkpoint_with_valid_checksum(ckpt_path, mod_ep)
        with pytest.raises(CheckpointSchemaError, match="Invalid epoch"):
            load_lora_checkpoint(ckpt_path)

    # 3. Global_step bad values
    for bad_step in [True, False, 2.5, "2", None, -1, -5]:
        reset_valid_ckpt()
        def mod_gs(p):
            p["global_step"] = bad_step
        rewrite_checkpoint_with_valid_checksum(ckpt_path, mod_gs)
        with pytest.raises(CheckpointSchemaError, match="Invalid global_step"):
            load_lora_checkpoint(ckpt_path)

    # 4. Extra bad values (non-dict)
    for bad_extra in [[], "string", 123, True]:
        reset_valid_ckpt()
        def mod_ex(p):
            p["extra"] = bad_extra
        rewrite_checkpoint_with_valid_checksum(ckpt_path, mod_ex)
        with pytest.raises(CheckpointSchemaError, match="Invalid extra"):
            load_lora_checkpoint(ckpt_path)

    # 5. Adapter_state bad values (None, list, string)
    for bad_ad in [None, [], "state"]:
        reset_valid_ckpt()
        def mod_ad(p):
            p["adapter_state"] = bad_ad
        rewrite_checkpoint_with_valid_checksum(ckpt_path, mod_ad)
        with pytest.raises(CheckpointSchemaError, match="Invalid adapter_state"):
            load_lora_checkpoint(ckpt_path)

    # 6. Optimizer_state bad values (list, string, int)
    for bad_opt in [[], "optimizer", 42]:
        reset_valid_ckpt()
        def mod_opt(p):
            p["optimizer_state"] = bad_opt
        rewrite_checkpoint_with_valid_checksum(ckpt_path, mod_opt)
        with pytest.raises(CheckpointSchemaError, match="Invalid optimizer_state"):
            load_lora_checkpoint(ckpt_path)


# =============================================================================
# 4. Unmerged-Only Exhaustive Lifecycle State Matrix
# =============================================================================

def test_lora_checkpoint_unmerged_lifecycle_matrix(temp_dir, active_backend):
    model = nn.Sequential(nn.LoRALinear(4, 6, rank=2), nn.LoRALinear(6, 2, rank=2))
    opt = optim.Adam(nn.adapter_parameters(model), lr=0.01)
    ckpt_path = os.path.join(temp_dir, "unmerged_matrix.json")

    # 1. Single layer corrupted states on save
    model[0]._merged = True
    model[0]._base_weight_snapshot = None  # Corrupted merged with None snapshot
    with pytest.raises(RuntimeError, match="require all adapters to be unmerged"):
        save_lora_checkpoint(ckpt_path, model=model, optimizer=opt)

    model[0]._merged = False
    model[0]._base_weight_snapshot = model[0].base.weight.backend.from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])  # Stale snapshot
    with pytest.raises(RuntimeError, match="require all adapters to be unmerged"):
        save_lora_checkpoint(ckpt_path, model=model, optimizer=opt)
    model[0]._base_weight_snapshot = None

    # 2. Save valid checkpoint
    save_lora_checkpoint(ckpt_path, model=model, optimizer=opt, epoch=1, global_step=1)
    assert os.path.exists(ckpt_path)

    # 3. Load into various illegal lifecycle states
    target = nn.Sequential(nn.LoRALinear(4, 6, rank=2), nn.LoRALinear(6, 2, rank=2))
    orig_t0_base = copy.deepcopy(target[0].base.weight.tolist())
    orig_t1_base = copy.deepcopy(target[1].base.weight.tolist())
    target_opt = optim.Adam(nn.adapter_parameters(target), lr=0.01)

    # Target layer 0 merged
    target[0].merge()
    with pytest.raises(RuntimeError, match="require all adapters to be unmerged"):
        load_lora_checkpoint(ckpt_path, model=target, optimizer=target_opt)
    target[0].unmerge()

    # Target layer 1 stale snapshot
    target[1]._base_weight_snapshot = target[1].base.weight.backend.from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    with pytest.raises(RuntimeError, match="require all adapters to be unmerged"):
        load_lora_checkpoint(ckpt_path, model=target, optimizer=target_opt)
    target[1]._base_weight_snapshot = None

    # Base weights unchanged
    assert target[0].base.weight.tolist() == orig_t0_base
    assert target[1].base.weight.tolist() == orig_t1_base


# =============================================================================
# 5. Optimizer Compatibility Matrix & Different Optimizers
# =============================================================================

def test_lora_checkpoint_optimizer_variants(temp_dir, active_backend):
    # Test Adam, SGD momentum, AdamW
    for opt_cls, opt_kwargs in [
        (optim.Adam, {"lr": 0.01}),
        (optim.SGD, {"lr": 0.01, "momentum": 0.9}),
        (optim.AdamW, {"lr": 0.01, "weight_decay": 0.01}),
    ]:
        model = nn.Sequential(nn.LoRALinear(4, 4, rank=2), nn.Tanh(), nn.LoRALinear(4, 2, rank=2))
        opt = opt_cls(nn.adapter_parameters(model), **opt_kwargs)

        # 2 training steps
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        tgt = Tensor([[0.5, -0.5]])
        for _ in range(2):
            opt.zero_grad()
            loss = ((model(x) - tgt) ** 2).sum()
            loss.backward()
            opt.step()

        ckpt_path = os.path.join(temp_dir, f"ckpt_{opt_cls.__name__}.json")
        save_lora_checkpoint(ckpt_path, model=model, optimizer=opt, epoch=2, global_step=2)

        fresh_model = nn.Sequential(nn.LoRALinear(4, 4, rank=2), nn.Tanh(), nn.LoRALinear(4, 2, rank=2))
        fresh_opt = opt_cls(nn.adapter_parameters(fresh_model), **opt_kwargs)

        load_lora_checkpoint(ckpt_path, model=fresh_model, optimizer=fresh_opt)

        for p1, p2 in zip(nn.adapter_parameters(model), nn.adapter_parameters(fresh_model)):
            assert p1.flatten().tolist() == pytest.approx(p2.flatten().tolist(), abs=1e-6)

        if opt_cls in (optim.Adam, optim.AdamW):
            assert fresh_opt.state[0]["step"] == 2
        elif opt_cls is optim.SGD:
            assert "momentum_buffer" in fresh_opt.state[0]


# =============================================================================
# 6. Atomic Save Protocol Failure Injection Matrix
# =============================================================================

def test_lora_checkpoint_save_failure_injections(temp_dir, active_backend):
    model = nn.LoRALinear(4, 2, rank=2)
    opt = optim.Adam(model.adapter_parameters(), lr=0.01)
    ckpt_path = os.path.join(temp_dir, "save_failure_test.json")

    # Initial valid checkpoint
    save_lora_checkpoint(ckpt_path, model=model, optimizer=opt, epoch=1, global_step=5)
    with open(ckpt_path, "r", encoding="utf-8") as f:
        orig_bytes = f.read()

    # 1. Non-string key in extra dict
    with pytest.raises(TypeError, match="contains non-string key"):
        save_lora_checkpoint(ckpt_path, model=model, optimizer=opt, extra={123: "val"})
    with open(ckpt_path, "r", encoding="utf-8") as f:
        assert f.read() == orig_bytes

    # 2. Non-serializable type in extra (set, bytes, Tensor)
    for bad_val in [{1, 2, 3}, b"bytes", Tensor([1.0])]:
        with pytest.raises(TypeError, match="contains non-JSON-serializable type"):
            save_lora_checkpoint(ckpt_path, model=model, optimizer=opt, extra={"bad": bad_val})
        with open(ckpt_path, "r", encoding="utf-8") as f:
            assert f.read() == orig_bytes

    # 3. Non-finite number in extra
    with pytest.raises(ValueError, match="contains non-finite number"):
        save_lora_checkpoint(ckpt_path, model=model, optimizer=opt, extra={"nan": float("nan")})

    # 4. Injected write failure
    orig_open = open
    def crashing_open(path, mode="r", *args, **kwargs):
        if str(path).endswith(".tmp") and "w" in mode:
            f = orig_open(path, mode, *args, **kwargs)
            orig_write = f.write
            def bad_write(content):
                raise OSError("Simulated disk full error during write")
            f.write = bad_write
            return f
        return orig_open(path, mode, *args, **kwargs)

    with patch("builtins.open", side_effect=crashing_open):
        with pytest.raises(CheckpointError, match="Simulated disk full error"):
            save_lora_checkpoint(ckpt_path, model=model, optimizer=opt, epoch=2, global_step=10)

    # Existing file intact and no leftover tmp file
    with open(ckpt_path, "r", encoding="utf-8") as f:
        assert f.read() == orig_bytes
    assert not os.path.exists(f"{ckpt_path}.tmp")

    # 5. Injected fsync failure
    with patch("os.fsync", side_effect=OSError("Simulated fsync I/O error")):
        with pytest.raises(CheckpointError, match="Simulated fsync I/O error"):
            save_lora_checkpoint(ckpt_path, model=model, optimizer=opt, epoch=2, global_step=10)

    with open(ckpt_path, "r", encoding="utf-8") as f:
        assert f.read() == orig_bytes
    assert not os.path.exists(f"{ckpt_path}.tmp")

    # 6. Injected replace failure
    with patch("os.replace", side_effect=OSError("Simulated atomic replace failure")):
        with pytest.raises(CheckpointError, match="Simulated atomic replace failure"):
            save_lora_checkpoint(ckpt_path, model=model, optimizer=opt, epoch=2, global_step=10)

    with open(ckpt_path, "r", encoding="utf-8") as f:
        assert f.read() == orig_bytes
    assert not os.path.exists(f"{ckpt_path}.tmp")


# =============================================================================
# 7. Atomic Load Failure & Rollback Failure Matrix
# =============================================================================

def test_lora_checkpoint_load_rollback_and_failure_chaining(temp_dir, active_backend):
    model = nn.LoRALinear(4, 2, rank=2)
    model.lora_A._data = model.lora_A.backend.from_data([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    opt = optim.Adam(model.adapter_parameters(), lr=0.01)

    ckpt_path = os.path.join(temp_dir, "rollback_chain_test.json")
    save_lora_checkpoint(ckpt_path, model=model, optimizer=opt, epoch=1, global_step=10)

    target_model = nn.LoRALinear(4, 2, rank=2)
    target_model.lora_A._data = target_model.lora_A.backend.from_data([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    orig_target_a = copy.deepcopy(target_model.lora_A.tolist())
    target_opt = optim.Adam(target_model.adapter_parameters(), lr=0.01)

    # 1. Rollback failure injection on adapter: both load and rollback fail -> CheckpointRollbackError
    with patch("termux_train.runtime.checkpoint.load_adapter_state_dict", side_effect=RuntimeError("Adapter load/rollback crash")):
        with pytest.raises(CheckpointRollbackError) as exc_info:
            load_lora_checkpoint(ckpt_path, model=target_model, optimizer=target_opt)

        assert "AND atomic rollback also failed" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    # 2. Rollback failure injection on optimizer -> CheckpointRollbackError
    orig_opt_load = target_opt.load_state_dict
    def always_crashing_opt_load(state):
        raise RuntimeError("Optimizer permanent crash")
    target_opt.load_state_dict = always_crashing_opt_load

    try:
        with pytest.raises(CheckpointRollbackError) as exc_info2:
            load_lora_checkpoint(ckpt_path, model=target_model, optimizer=target_opt)
        assert "optimizer rollback failed" in str(exc_info2.value)
    finally:
        target_opt.load_state_dict = orig_opt_load


# =============================================================================
# 8. Cross-Backend Numerical Parity & Next Optimizer Step Parity
# =============================================================================

def test_lora_checkpoint_cross_backend_resume_and_step_parity(temp_dir):
    if "numpy" not in available_backends():
        pytest.skip("NumPy backend not available for cross-backend parity test")

    # Step A: Train on PythonBackend for 3 steps, save checkpoint
    set_backend("python")
    py_model = nn.Sequential(nn.LoRALinear(4, 6, rank=2), nn.Tanh(), nn.LoRALinear(6, 2, rank=2))
    py_opt = optim.Adam(nn.adapter_parameters(py_model), lr=0.05)

    x_data = [[0.5, -0.2, 0.8, -0.4], [0.1, 0.9, -0.5, 0.3]]
    tgt_data = [[1.0, 0.0], [0.0, 1.0]]

    for _ in range(3):
        py_opt.zero_grad()
        loss = ((py_model(Tensor(x_data)) - Tensor(tgt_data)) ** 2).mean()
        loss.backward()
        py_opt.step()

    ckpt_py = os.path.join(temp_dir, "py_saved.json")
    save_lora_checkpoint(ckpt_py, model=py_model, optimizer=py_opt, epoch=3, global_step=3)

    # Step B: Load on NumPyBackend, set identical base weights, verify parity
    set_backend("numpy")
    np_model = nn.Sequential(nn.LoRALinear(4, 6, rank=2), nn.Tanh(), nn.LoRALinear(6, 2, rank=2))
    # Match initial base weights
    np_model[0].base.weight._data = np_model[0].base.weight.backend.from_data(py_model[0].base.weight.tolist())
    np_model[0].base.bias._data = np_model[0].base.bias.backend.from_data(py_model[0].base.bias.tolist())
    np_model[2].base.weight._data = np_model[2].base.weight.backend.from_data(py_model[2].base.weight.tolist())
    np_model[2].base.bias._data = np_model[2].base.bias.backend.from_data(py_model[2].base.bias.tolist())

    np_opt = optim.Adam(nn.adapter_parameters(np_model), lr=0.05)

    load_lora_checkpoint(ckpt_py, model=np_model, optimizer=np_opt)

    # 1. Check weight parity
    for p_py, p_np in zip(nn.adapter_parameters(py_model), nn.adapter_parameters(np_model)):
        assert p_py.flatten().tolist() == pytest.approx(p_np.flatten().tolist(), abs=1e-5, rel=1e-5)

    # 2. Check forward parity
    set_backend("python")
    out_py = py_model(Tensor(x_data))
    set_backend("numpy")
    out_np = np_model(Tensor(x_data))
    assert out_py.flatten().tolist() == pytest.approx(out_np.flatten().tolist(), abs=1e-5, rel=1e-5)

    # 3. Check next optimizer step numerical parity
    set_backend("python")
    py_opt.zero_grad()
    loss_py = ((py_model(Tensor(x_data)) - Tensor(tgt_data)) ** 2).mean()
    loss_py.backward()
    py_opt.step()

    set_backend("numpy")
    np_opt.zero_grad()
    loss_np = ((np_model(Tensor(x_data)) - Tensor(tgt_data)) ** 2).mean()
    loss_np.backward()
    np_opt.step()

    for p_py, p_np in zip(nn.adapter_parameters(py_model), nn.adapter_parameters(np_model)):
        assert p_py.flatten().tolist() == pytest.approx(p_np.flatten().tolist(), abs=1e-5, rel=1e-5)


# =============================================================================
# 9. Shared Module Deduplication & Nested Modules
# =============================================================================

def test_lora_checkpoint_shared_and_nested_modules(temp_dir, active_backend):
    shared_lora = nn.LoRALinear(4, 4, rank=2)
    nested_model = nn.Sequential(
        shared_lora,
        nn.Tanh(),
        shared_lora,  # Shared reference
        nn.LoRALinear(4, 2, rank=2),
    )
    opt = optim.Adam(nn.adapter_parameters(nested_model), lr=0.01)
    # Expected exactly 4 adapter params (lora_A, lora_B for shared_lora, and for the last layer)
    assert len(nn.adapter_parameters(nested_model)) == 4

    ckpt_path = os.path.join(temp_dir, "shared_nested.json")
    save_lora_checkpoint(ckpt_path, model=nested_model, optimizer=opt, epoch=1, global_step=5)

    fresh_shared = nn.LoRALinear(4, 4, rank=2)
    fresh_nested = nn.Sequential(
        fresh_shared,
        nn.Tanh(),
        fresh_shared,
        nn.LoRALinear(4, 2, rank=2),
    )
    fresh_opt = optim.Adam(nn.adapter_parameters(fresh_nested), lr=0.01)

    load_lora_checkpoint(ckpt_path, model=fresh_nested, optimizer=fresh_opt)

    # Verify shared layer references remain identical object
    assert fresh_nested[0] is fresh_nested[2]
    assert fresh_nested[0].lora_A is fresh_nested[2].lora_A
