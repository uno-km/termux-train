"""
tests/test_runtime.py
=====================
Comprehensive unit tests for termux_train.runtime (Safe Checkpointing & MobileTrainer).
Tests:
  - Checkpoint round-trip serialization and deserialization
  - SHA256 integrity check and corruption detection
  - Atomic write protection (failure does not destroy existing checkpoint)
  - Atomic load rollback on model or optimizer restoration failure
  - MobileTrainer standard training loop execution
  - MobileTrainer periodic checkpointing
  - MobileTrainer explicit stop request
  - MobileTrainer checkpoint resume consistency
"""

import copy
import hashlib
import hmac
import json
import os
import shutil
import tempfile
import pytest
from termux_train import Tensor, nn, optim, available_backends, set_backend
from termux_train.runtime import (
    save_checkpoint,
    load_checkpoint,
    CheckpointError,
    CheckpointIntegrityError,
    CheckpointSchemaError,
    MobileTrainer,
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


# =============================================================================
# 1. Safe Checkpoint Core & Integrity Tests
# =============================================================================

def test_checkpoint_save_and_load_round_trip(temp_dir, active_backend):
    model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Perform 1 optimization step to populate optimizer state
    x = Tensor([[1.0, 2.0]])
    target = Tensor([[1.0]])
    loss = ((model(x) - target) ** 2).sum()
    loss.backward()
    optimizer.step()

    ckpt_path = os.path.join(temp_dir, "test_checkpoint.json")
    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        epoch=5,
        global_step=42,
        extra={"custom_metric": 0.123},
    )

    assert os.path.exists(ckpt_path)

    # Create fresh model & optimizer
    fresh_model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
    fresh_optimizer = optim.Adam(fresh_model.parameters(), lr=0.01)

    meta = load_checkpoint(ckpt_path, model=fresh_model, optimizer=fresh_optimizer)

    assert meta["epoch"] == 5
    assert meta["global_step"] == 42
    assert meta["extra"]["custom_metric"] == 0.123

    # Check model parameter parity
    for p_orig, p_loaded in zip(model.parameters(), fresh_model.parameters()):
        assert p_orig.flatten().tolist() == pytest.approx(p_loaded.flatten().tolist(), abs=1e-6)

    # Check optimizer state parity
    assert fresh_optimizer.state[0]["step"] == 1


def test_checkpoint_checksum_integrity_corruption_rejection(temp_dir, active_backend):
    model = nn.Linear(2, 2)
    ckpt_path = os.path.join(temp_dir, "corrupt_ckpt.json")
    save_checkpoint(ckpt_path, model=model, epoch=1, global_step=10)

    # Corrupt the payload inside the file
    with open(ckpt_path, "r", encoding="utf-8") as f:
        container = json.load(f)

    # Tamper with epoch without updating checksum
    container["payload"]["epoch"] = 999

    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(container, f)

    fresh_model = nn.Linear(2, 2)
    with pytest.raises(CheckpointIntegrityError, match="checksum mismatch"):
        load_checkpoint(ckpt_path, model=fresh_model)


def test_checkpoint_malformed_json_rejection(temp_dir, active_backend):
    ckpt_path = os.path.join(temp_dir, "malformed.json")
    with open(ckpt_path, "w", encoding="utf-8") as f:
        f.write("{ invalid json text ")

    fresh_model = nn.Linear(2, 2)
    with pytest.raises(CheckpointIntegrityError, match="Malformed"):
        load_checkpoint(ckpt_path, model=fresh_model)


def test_checkpoint_schema_mismatch_rejection(temp_dir, active_backend):
    ckpt_path = os.path.join(temp_dir, "bad_schema.json")
    save_checkpoint(ckpt_path, epoch=1, global_step=1)

    with open(ckpt_path, "r", encoding="utf-8") as f:
        container = json.load(f)

    container["payload"]["format"] = "unsupported-format"
    # Update checksum so integrity passes, but schema check catches it
    import hashlib
    container["checksum"] = hashlib.sha256(
        json.dumps(container["payload"], sort_keys=True, indent=2).encode("utf-8")
    ).hexdigest()

    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(container, f)

    with pytest.raises(CheckpointSchemaError, match="Unsupported checkpoint format"):
        load_checkpoint(ckpt_path)


def test_checkpoint_version_mismatch_rejection(temp_dir, active_backend):
    ckpt_path = os.path.join(temp_dir, "bad_version.json")
    save_checkpoint(ckpt_path, epoch=1, global_step=1)

    with open(ckpt_path, "r", encoding="utf-8") as f:
        container = json.load(f)

    container["payload"]["version"] = "99.0"
    import hashlib
    container["checksum"] = hashlib.sha256(
        json.dumps(container["payload"], sort_keys=True, indent=2).encode("utf-8")
    ).hexdigest()

    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(container, f)

    with pytest.raises(CheckpointSchemaError, match="Unsupported checkpoint version"):
        load_checkpoint(ckpt_path)


def test_checkpoint_atomic_load_rollback_on_failure(temp_dir, active_backend):
    model = nn.Linear(2, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Save valid checkpoint for 2x2 linear layer
    ckpt_path = os.path.join(temp_dir, "linear2x2.json")
    save_checkpoint(ckpt_path, model=model, optimizer=optimizer, epoch=1)

    # Now create incompatible model (Linear 4x4) and optimizer
    incompatible_model = nn.Linear(4, 4)
    incompatible_opt = optim.Adam(incompatible_model.parameters(), lr=0.05)

    orig_incompat_weight = copy.deepcopy(incompatible_model.weight.tolist())
    orig_incompat_lr = incompatible_opt.defaults["lr"]

    with pytest.raises(Exception):
        load_checkpoint(ckpt_path, model=incompatible_model, optimizer=incompatible_opt)

    # Atomic rollback: incompatible model and optimizer must remain completely untouched
    assert incompatible_model.weight.tolist() == orig_incompat_weight
    assert incompatible_opt.defaults["lr"] == orig_incompat_lr


def test_checkpoint_rollback_failure_raises_rollback_error(temp_dir, active_backend):
    from termux_train.runtime.checkpoint import CheckpointRollbackError
    model = nn.Linear(2, 2)
    ckpt_path = os.path.join(temp_dir, "corrupt_rollback.json")
    save_checkpoint(ckpt_path, model=model, epoch=1)

    # Broken model where both load_state_dict and rollback fail
    class BrokenModel(nn.Module):
        def state_dict(self):
            return {"dummy": 1}
        def load_state_dict(self, state):
            raise RuntimeError("load failed intentionally")

    broken = BrokenModel()
    with pytest.raises(CheckpointRollbackError, match="AND atomic rollback also failed"):
        load_checkpoint(ckpt_path, model=broken)


# =============================================================================
# 2. MobileTrainer Training & Resume Tests
# =============================================================================

def test_mobile_trainer_fit_and_periodic_checkpoint(temp_dir, active_backend):
    model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1), nn.Sigmoid())
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    target = Tensor([[0.0], [1.0], [1.0], [0.0]])

    trainer = MobileTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_dir,
        checkpoint_every_epochs=2,
    )

    res = trainer.fit(dataset=(x, target), epochs=4)

    assert res["epochs_completed"] == 4
    assert res["global_step"] == 4
    assert res["stopped_early"] is False
    assert len(res["history"]) == 4

    # Check periodic checkpoint files were created
    assert os.path.exists(os.path.join(temp_dir, "checkpoint_epoch_2.json"))
    assert os.path.exists(os.path.join(temp_dir, "checkpoint_epoch_4.json"))
    assert os.path.exists(os.path.join(temp_dir, "checkpoint_latest.json"))


def test_mobile_trainer_explicit_stop_request(temp_dir, active_backend):
    model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    x = Tensor([[0.0, 0.0], [1.0, 1.0]])
    target = Tensor([[0.0], [1.0]])

    trainer = MobileTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_dir,
    )

    def stop_on_epoch_2(info):
        if info["epoch"] >= 2:
            trainer.request_stop()

    res = trainer.fit(dataset=(x, target), epochs=10, on_epoch_end=stop_on_epoch_2)

    assert res["epochs_completed"] == 2
    assert res["stopped_early"] is True


def test_mobile_trainer_resume_exact_consistency(temp_dir, active_backend):
    # Train 5 epochs, save, then train 5 more from resume
    model1 = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1), nn.Sigmoid())
    optimizer1 = optim.Adam(model1.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    target = Tensor([[0.0], [1.0], [1.0], [0.0]])

    trainer1 = MobileTrainer(
        model=model1,
        optimizer=optimizer1,
        criterion=criterion,
        checkpoint_dir=temp_dir,
    )

    trainer1.fit(dataset=(x, target), epochs=5)
    ckpt_5 = os.path.join(temp_dir, "phase1_ckpt.json")
    trainer1.save(ckpt_5)

    # Resume into fresh trainer
    model2 = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1), nn.Sigmoid())
    optimizer2 = optim.Adam(model2.parameters(), lr=0.05)

    trainer2 = MobileTrainer(
        model=model2,
        optimizer=optimizer2,
        criterion=criterion,
        checkpoint_dir=temp_dir,
    )

    trainer2.fit(dataset=(x, target), epochs=5, resume_from=ckpt_5)

    assert trainer2.current_epoch == 10
    assert trainer2.global_step == 10


# =============================================================================
# 3. Dedicated Safe LoRA Checkpoint Integration Tests (SCRUM-311)
# =============================================================================

def test_lora_checkpoint_save_and_load_round_trip(temp_dir, active_backend):
    from termux_train.runtime import save_lora_checkpoint, load_lora_checkpoint

    model = nn.Sequential(
        nn.LoRALinear(4, 6, rank=2, alpha=2.0),
        nn.Tanh(),
        nn.Linear(6, 6),
        nn.LoRALinear(6, 2, rank=2, alpha=2.0),
    )
    optimizer = optim.Adam(nn.adapter_parameters(model), lr=0.01)

    # Run 3 training steps on adapter parameters
    x = Tensor([[1.0, 2.0, 3.0, 4.0]])
    target = Tensor([[0.5, -0.5]])
    for _ in range(3):
        optimizer.zero_grad()
        loss = ((model(x) - target) ** 2).sum()
        loss.backward()
        optimizer.step()

    ckpt_path = os.path.join(temp_dir, "lora_ckpt.json")
    save_lora_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        global_step=15,
        extra={"custom_eval_loss": 0.042},
    )

    assert os.path.exists(ckpt_path)

    # Inspect checkpoint JSON content directly
    with open(ckpt_path, "r", encoding="utf-8") as f:
        container = json.load(f)

    assert "checksum" in container
    assert "payload" in container
    payload = container["payload"]
    assert payload["format"] == "termux-train-lora-checkpoint"
    assert payload["version"] == "1.0"
    assert payload["epoch"] == 3
    assert payload["global_step"] == 15
    assert payload["extra"]["custom_eval_loss"] == 0.042
    assert "model_state" not in payload
    assert "base.weight" not in str(payload["adapter_state"])
    assert "base.bias" not in str(payload["adapter_state"])
    assert "_base_weight_snapshot" not in str(payload["adapter_state"])

    # Load into fresh model and optimizer
    fresh_model = nn.Sequential(
        nn.LoRALinear(4, 6, rank=2, alpha=2.0),
        nn.Tanh(),
        nn.Linear(6, 6),
        nn.LoRALinear(6, 2, rank=2, alpha=2.0),
    )
    orig_fresh_base_w0 = copy.deepcopy(fresh_model[0].base.weight.tolist())
    orig_fresh_base_w3 = copy.deepcopy(fresh_model[3].base.weight.tolist())

    fresh_optimizer = optim.Adam(nn.adapter_parameters(fresh_model), lr=0.01)

    meta = load_lora_checkpoint(ckpt_path, model=fresh_model, optimizer=fresh_optimizer)

    assert meta["epoch"] == 3
    assert meta["global_step"] == 15
    assert meta["extra"]["custom_eval_loss"] == 0.042

    # Adapter factors match trained model
    for p_orig, p_loaded in zip(nn.adapter_parameters(model), nn.adapter_parameters(fresh_model)):
        assert p_orig.flatten().tolist() == pytest.approx(p_loaded.flatten().tolist(), abs=1e-6)

    # Base weights are completely untouched
    assert fresh_model[0].base.weight.tolist() == orig_fresh_base_w0
    assert fresh_model[3].base.weight.tolist() == orig_fresh_base_w3

    # Optimizer step matches
    assert fresh_optimizer.state[0]["step"] == 3


def test_lora_checkpoint_unmerged_only_save_and_load_rejections(temp_dir, active_backend):
    from termux_train.runtime import save_lora_checkpoint, load_lora_checkpoint

    model = nn.Sequential(
        nn.LoRALinear(4, 6, rank=2),
        nn.LoRALinear(6, 2, rank=2),
    )
    optimizer = optim.Adam(nn.adapter_parameters(model), lr=0.01)
    ckpt_path = os.path.join(temp_dir, "unmerged_test.json")

    # 1. Merged model during save
    model[0].merge()
    with pytest.raises(RuntimeError, match="require all adapters to be unmerged"):
        save_lora_checkpoint(ckpt_path, model=model, optimizer=optimizer)
    assert not os.path.exists(ckpt_path)

    # 2. Unmerged but with stale snapshot during save
    model[0].unmerge()
    model[1]._base_weight_snapshot = model[1].base.weight.backend.from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    with pytest.raises(RuntimeError, match="require all adapters to be unmerged"):
        save_lora_checkpoint(ckpt_path, model=model, optimizer=optimizer)
    model[1]._base_weight_snapshot = None

    # Save valid checkpoint
    save_lora_checkpoint(ckpt_path, model=model, optimizer=optimizer)
    assert os.path.exists(ckpt_path)

    # 3. Merged target model during load
    fresh_model = nn.Sequential(
        nn.LoRALinear(4, 6, rank=2),
        nn.LoRALinear(6, 2, rank=2),
    )
    fresh_optimizer = optim.Adam(nn.adapter_parameters(fresh_model), lr=0.01)
    fresh_model[1].merge()

    with pytest.raises(RuntimeError, match="require all adapters to be unmerged"):
        load_lora_checkpoint(ckpt_path, model=fresh_model, optimizer=fresh_optimizer)


def test_lora_checkpoint_requires_lora_layer_and_adapter_only_optimizer(temp_dir, active_backend):
    from termux_train.runtime import save_lora_checkpoint, load_lora_checkpoint

    plain_model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
    plain_opt = optim.Adam(plain_model.parameters(), lr=0.01)
    ckpt_path = os.path.join(temp_dir, "no_lora.json")

    # 1. Plain model without LoRALinear raises ValueError
    with pytest.raises(ValueError, match="requires at least one LoRALinear layer"):
        save_lora_checkpoint(ckpt_path, model=plain_model, optimizer=plain_opt)

    # 2. Model with LoRALinear but optimizer tracking full model parameters (including base weights)
    lora_model = nn.Sequential(nn.LoRALinear(4, 6, rank=2), nn.Linear(6, 2))
    full_opt = optim.Adam(lora_model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="must exactly match model adapter parameters"):
        save_lora_checkpoint(ckpt_path, model=lora_model, optimizer=full_opt)

    # 3. Optimizer with reversed parameter order
    reversed_opt = optim.Adam(list(reversed(nn.adapter_parameters(lora_model))), lr=0.01)
    with pytest.raises(ValueError, match="must exactly match model adapter parameters"):
        save_lora_checkpoint(ckpt_path, model=lora_model, optimizer=reversed_opt)


def test_lora_checkpoint_checksum_integrity_and_tampering_rejections(temp_dir, active_backend):
    from termux_train.runtime import save_lora_checkpoint, load_lora_checkpoint

    model = nn.LoRALinear(4, 2, rank=2)
    optimizer = optim.Adam(model.adapter_parameters(), lr=0.01)
    ckpt_path = os.path.join(temp_dir, "integrity_test.json")

    save_lora_checkpoint(ckpt_path, model=model, optimizer=optimizer, epoch=2, global_step=20)

    # 1. Tamper payload epoch
    with open(ckpt_path, "r", encoding="utf-8") as f:
        container = json.load(f)
    bad_container = copy.deepcopy(container)
    bad_container["payload"]["epoch"] = 999

    bad_path = os.path.join(temp_dir, "bad_epoch.json")
    with open(bad_path, "w", encoding="utf-8") as f:
        json.dump(bad_container, f)

    fresh_model = nn.LoRALinear(4, 2, rank=2)
    fresh_opt = optim.Adam(fresh_model.adapter_parameters(), lr=0.01)
    with pytest.raises(CheckpointIntegrityError, match="checksum mismatch"):
        load_lora_checkpoint(bad_path, model=fresh_model, optimizer=fresh_opt)

    # 2. Tamper adapter value
    bad_container2 = copy.deepcopy(container)
    bad_container2["payload"]["adapter_state"]["lora_A"][0][0] += 1.0
    bad_path2 = os.path.join(temp_dir, "bad_adapter.json")
    with open(bad_path2, "w", encoding="utf-8") as f:
        json.dump(bad_container2, f)

    with pytest.raises(CheckpointIntegrityError, match="checksum mismatch"):
        load_lora_checkpoint(bad_path2, model=fresh_model, optimizer=fresh_opt)

    # 3. Corrupted / truncated JSON
    bad_path3 = os.path.join(temp_dir, "truncated.json")
    with open(bad_path3, "w", encoding="utf-8") as f:
        f.write('{"checksum": "1234", "payload": {')

    with pytest.raises(CheckpointIntegrityError, match="Malformed or unreadable"):
        load_lora_checkpoint(bad_path3, model=fresh_model, optimizer=fresh_opt)


def test_lora_checkpoint_schema_and_version_rejections(temp_dir, active_backend):
    from termux_train.runtime import save_lora_checkpoint, load_lora_checkpoint

    model = nn.LoRALinear(4, 2, rank=2)
    optimizer = optim.Adam(model.adapter_parameters(), lr=0.01)
    ckpt_path = os.path.join(temp_dir, "schema_test.json")
    save_lora_checkpoint(ckpt_path, model=model, optimizer=optimizer, epoch=1, global_step=5)

    with open(ckpt_path, "r", encoding="utf-8") as f:
        container = json.load(f)

    # Helper to re-save with valid checksum
    def save_valid_checksum(mod_payload, target_path):
        can_json = json.dumps(mod_payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        chk = hashlib.sha256(can_json.encode("utf-8")).hexdigest()
        with open(target_path, "w", encoding="utf-8") as f_out:
            json.dump({"checksum": chk, "payload": mod_payload}, f_out, indent=2)

    fresh_model = nn.LoRALinear(4, 2, rank=2)
    fresh_opt = optim.Adam(fresh_model.adapter_parameters(), lr=0.01)

    # 1. Format mismatch
    p1 = copy.deepcopy(container["payload"])
    p1["format"] = "wrong-format"
    p1_path = os.path.join(temp_dir, "p1.json")
    save_valid_checksum(p1, p1_path)
    with pytest.raises(CheckpointSchemaError, match="Unsupported LoRA checkpoint format"):
        load_lora_checkpoint(p1_path, model=fresh_model, optimizer=fresh_opt)

    # 2. Version mismatch
    p2 = copy.deepcopy(container["payload"])
    p2["version"] = "2.0"
    p2_path = os.path.join(temp_dir, "p2.json")
    save_valid_checksum(p2, p2_path)
    with pytest.raises(CheckpointSchemaError, match="Unsupported LoRA checkpoint version"):
        load_lora_checkpoint(p2_path, model=fresh_model, optimizer=fresh_opt)

    # 3. Epoch is bool or negative
    p3 = copy.deepcopy(container["payload"])
    p3["epoch"] = -1
    p3_path = os.path.join(temp_dir, "p3.json")
    save_valid_checksum(p3, p3_path)
    with pytest.raises(CheckpointSchemaError, match="Invalid epoch"):
        load_lora_checkpoint(p3_path, model=fresh_model, optimizer=fresh_opt)


def test_lora_checkpoint_atomic_load_rollback_on_failure(temp_dir, active_backend):
    from termux_train.runtime import save_lora_checkpoint, load_lora_checkpoint

    model = nn.LoRALinear(4, 2, rank=2)
    model.lora_A._data = model.lora_A.backend.from_data([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    opt = optim.Adam(model.adapter_parameters(), lr=0.01)

    ckpt_path = os.path.join(temp_dir, "rollback_test.json")
    save_lora_checkpoint(ckpt_path, model=model, optimizer=opt, epoch=1, global_step=10)

    # Create target model with distinct initial weights
    target_model = nn.LoRALinear(4, 2, rank=2)
    target_model.lora_A._data = target_model.lora_A.backend.from_data([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    orig_target_a = copy.deepcopy(target_model.lora_A.tolist())

    target_opt = optim.Adam(target_model.adapter_parameters(), lr=0.01)

    # Inject crash into optimizer load_state_dict only on first call
    attempts = 0
    orig_opt_load = target_opt.load_state_dict
    try:
        def crashing_opt_load(state):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("Simulated crash during optimizer state restore")
            return orig_opt_load(state)
        target_opt.load_state_dict = crashing_opt_load

        with pytest.raises(RuntimeError, match="Simulated crash during optimizer state restore"):
            load_lora_checkpoint(ckpt_path, model=target_model, optimizer=target_opt)
    finally:
        target_opt.load_state_dict = orig_opt_load

    # Adapter state 100% rolled back to pre-call state
    assert target_model.lora_A.tolist() == orig_target_a


def test_lora_checkpoint_atomic_save_failure_preserves_existing(temp_dir, active_backend):
    from termux_train.runtime import save_lora_checkpoint, CheckpointError

    model = nn.LoRALinear(4, 2, rank=2)
    opt = optim.Adam(model.adapter_parameters(), lr=0.01)
    ckpt_path = os.path.join(temp_dir, "existing_ckpt.json")

    # 1. Save initial valid checkpoint
    save_lora_checkpoint(ckpt_path, model=model, optimizer=opt, epoch=1, global_step=5)
    with open(ckpt_path, "r", encoding="utf-8") as f:
        orig_content = f.read()

    # 2. Attempt to save with un-serializable extra metadata (e.g. set object)
    with pytest.raises(CheckpointError):
        save_lora_checkpoint(ckpt_path, model=model, optimizer=opt, epoch=2, global_step=10, extra={"bad_set": {1, 2, 3}})

    # Existing file is preserved byte-for-byte
    with open(ckpt_path, "r", encoding="utf-8") as f:
        assert f.read() == orig_content


def test_lora_checkpoint_cross_backend_portability(temp_dir):
    from termux_train.runtime import save_lora_checkpoint, load_lora_checkpoint
    from termux_train.backend import set_backend

    if "numpy" not in available_backends():
        pytest.skip("NumPy backend not available for cross-backend test")

    # 1. Save on PythonBackend
    set_backend("python")
    py_model = nn.Sequential(nn.LoRALinear(4, 6, rank=2), nn.Tanh(), nn.LoRALinear(6, 2, rank=2))
    py_opt = optim.Adam(nn.adapter_parameters(py_model), lr=0.01)

    x_py = Tensor([[1.0, 2.0, 3.0, 4.0]])
    tgt_py = Tensor([[0.5, -0.5]])
    loss = ((py_model(x_py) - tgt_py) ** 2).sum()
    loss.backward()
    py_opt.step()

    ckpt_path = os.path.join(temp_dir, "cross_backend.json")
    save_lora_checkpoint(ckpt_path, model=py_model, optimizer=py_opt, epoch=1, global_step=1)

    # 2. Load on NumPyBackend
    set_backend("numpy")
    np_model = nn.Sequential(nn.LoRALinear(4, 6, rank=2), nn.Tanh(), nn.LoRALinear(6, 2, rank=2))
    np_opt = optim.Adam(nn.adapter_parameters(np_model), lr=0.01)

    load_lora_checkpoint(ckpt_path, model=np_model, optimizer=np_opt)

    # Verify adapter factor values match exactly
    for p_py, p_np in zip(nn.adapter_parameters(py_model), nn.adapter_parameters(np_model)):
        assert p_py.flatten().tolist() == pytest.approx(p_np.flatten().tolist(), abs=1e-6)
        assert p_np.backend.name == "numpy"

    assert np_opt.state[0]["step"] == 1
