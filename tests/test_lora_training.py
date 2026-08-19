"""
tests/test_lora_training.py
===========================
Comprehensive Test Suite for SCRUM-312: MobileTrainer LoRA Integration & Fine-tuning Lifecycle.

Sections:
  1. MobileTrainer LoRA constructor and API contracts
  2. Training, gradients, and base parameter invariance
  3. Periodic checkpointing and exact resume
  4. Failure injection and trainer-level rollback
  5. Teacher-student domain adaptation and convergence
  6. Continuous vs interrupted training equivalence
  7. Cross-backend resume and next-step parity
  8. Deployment merge lifecycle and re-training guards
  9. Callbacks, stop requests, and reentrant state machine
"""

import copy
import math
import os
import random
import shutil
import tempfile
import pytest
from unittest.mock import patch

from termux_train import Tensor, nn, optim, runtime, set_backend, available_backends
from termux_train.runtime import (
    MobileTrainer,
    save_lora_checkpoint,
    load_lora_checkpoint,
    CheckpointError,
    CheckpointIntegrityError,
    CheckpointSchemaError,
)


@pytest.fixture(params=["python"] + (["numpy"] if "numpy" in available_backends() else []))
def active_backend(request):
    set_backend(request.param)
    return request.param


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="termux_lora_train_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def build_pretrained_base():
    """Builds a deterministic 2-layer MLP base model."""
    random.seed(42)
    m = nn.Sequential(
        nn.Linear(4, 8),
        nn.Tanh(),
        nn.Linear(8, 2),
    )
    return m


def wrap_student_with_lora(base_model, rank=2, alpha=2.0):
    """Wraps Linear layers of base_model into LoRALinear with frozen base weights."""
    student = nn.Sequential(
        nn.LoRALinear.from_linear(base_model[0], rank=rank, alpha=alpha),
        nn.Tanh(),
        nn.LoRALinear.from_linear(base_model[2], rank=rank, alpha=alpha),
    )
    return student


# =============================================================================
# Section 1: MobileTrainer LoRA Constructor and API Contracts
# =============================================================================

def test_mobile_trainer_lora_constructor_validations(temp_dir, active_backend):
    base = build_pretrained_base()
    student = wrap_student_with_lora(base)
    opt = optim.Adam(nn.adapter_parameters(student), lr=0.01)
    crit = nn.MSELoss()

    # 1. lora_only strict boolean validation
    for bad_lora in [0, 1, None, "true", "True", [], {}]:
        with pytest.raises(TypeError, match="lora_only must be a bool"):
            MobileTrainer(model=student, optimizer=opt, criterion=crit, lora_only=bad_lora)

    # 2. checkpoint_every_epochs validation
    for bad_ep in [0, -1, -5, True, False, 1.5, "10"]:
        with pytest.raises(ValueError, match="checkpoint_every_epochs must be a positive integer"):
            MobileTrainer(
                model=student,
                optimizer=opt,
                criterion=crit,
                checkpoint_dir=temp_dir,
                checkpoint_every_epochs=bad_ep,
                lora_only=True,
            )

    # 3. checkpoint_every_steps validation
    for bad_st in [0, -1, -10, True, False, 2.5, "5"]:
        with pytest.raises(ValueError, match="checkpoint_every_steps must be a positive integer"):
            MobileTrainer(
                model=student,
                optimizer=opt,
                criterion=crit,
                checkpoint_dir=temp_dir,
                checkpoint_every_steps=bad_st,
                lora_only=True,
            )

    # 4. checkpoint_dir required when scheduling active
    for bad_dir in [None, "", "   "]:
        with pytest.raises(ValueError, match="checkpoint_dir must be a non-empty string"):
            MobileTrainer(
                model=student,
                optimizer=opt,
                criterion=crit,
                checkpoint_dir=bad_dir,
                checkpoint_every_epochs=5,
                lora_only=True,
            )

    # 5. Generic mode backward compatibility (lora_only=False works with plain model)
    plain_model = build_pretrained_base()
    plain_opt = optim.Adam(plain_model.parameters(), lr=0.01)
    generic_trainer = MobileTrainer(
        model=plain_model,
        optimizer=plain_opt,
        criterion=crit,
        checkpoint_dir=temp_dir,
        lora_only=False,
    )
    assert generic_trainer._checkpoint_mode == "full"
    assert generic_trainer.lora_only is False


def test_mobile_trainer_lora_preflight_rejections(temp_dir, active_backend):
    crit = nn.MSELoss()

    # 1. Model without LoRALinear rejected in lora_only=True mode
    plain_model = build_pretrained_base()
    plain_opt = optim.Adam(plain_model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="requires at least one LoRALinear layer"):
        MobileTrainer(model=plain_model, optimizer=plain_opt, criterion=crit, lora_only=True)

    # 2. Merged model rejected
    base = build_pretrained_base()
    student = wrap_student_with_lora(base)
    student[0].merge()
    opt = optim.Adam(nn.adapter_parameters(student), lr=0.01)
    with pytest.raises(RuntimeError, match="require all adapters to be unmerged"):
        MobileTrainer(model=student, optimizer=opt, criterion=crit, lora_only=True)
    student[0].unmerge()

    # 3. Stale snapshot rejected
    student[0]._base_weight_snapshot = student[0].base.weight.backend.from_data([[1.0] * 8] * 4)
    with pytest.raises(RuntimeError, match="require all adapters to be unmerged"):
        MobileTrainer(model=student, optimizer=opt, criterion=crit, lora_only=True)
    student[0]._base_weight_snapshot = None

    # 4. Full model optimizer (including base weights) rejected
    full_opt = optim.Adam(student.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="must exactly match model adapter parameters"):
        MobileTrainer(model=student, optimizer=full_opt, criterion=crit, lora_only=True)

    # 5. Reversed parameter order rejected
    rev_opt = optim.Adam(list(reversed(nn.adapter_parameters(student))), lr=0.01)
    with pytest.raises(ValueError, match="must exactly match model adapter parameters"):
        MobileTrainer(model=student, optimizer=rev_opt, criterion=crit, lora_only=True)

    # 6. Optimizer for another model rejected
    other_student = wrap_student_with_lora(build_pretrained_base())
    other_opt = optim.Adam(nn.adapter_parameters(other_student), lr=0.01)
    with pytest.raises(ValueError, match="must exactly match model adapter parameters"):
        MobileTrainer(model=student, optimizer=other_opt, criterion=crit, lora_only=True)


# =============================================================================
# Section 2: Training, Gradients, and Base Invariance
# =============================================================================

def test_mobile_trainer_lora_gradient_and_base_invariance(temp_dir, active_backend):
    base = build_pretrained_base()
    student = wrap_student_with_lora(base)
    opt = optim.Adam(nn.adapter_parameters(student), lr=0.05)
    crit = nn.MSELoss()

    orig_base0_w = copy.deepcopy(student[0].base.weight.tolist())
    orig_base0_b = copy.deepcopy(student[0].base.bias.tolist())
    orig_base2_w = copy.deepcopy(student[2].base.weight.tolist())
    orig_base2_b = copy.deepcopy(student[2].base.bias.tolist())

    base0_w_id = id(student[0].base.weight)
    base0_b_id = id(student[0].base.bias)

    orig_adapter0_b = copy.deepcopy(student[0].lora_B.tolist())

    trainer = MobileTrainer(model=student, optimizer=opt, criterion=crit, lora_only=True)

    x = Tensor([[1.0, -1.0, 2.0, -2.0], [0.5, 0.5, -0.5, -0.5]])
    target = Tensor([[1.0, 0.0], [0.0, 1.0]])

    # Fit for 3 epochs
    res = trainer.fit(dataset=(x, target), epochs=3)

    assert res["epochs_completed"] == 3
    assert res["global_step"] == 3
    assert len(res["history"]) == 3

    # 1. Base gradients are None and requires_grad=False
    assert student[0].base.weight.grad is None
    assert student[0].base.bias.grad is None
    assert student[0].base.weight.requires_grad is False
    assert student[0].base.bias.requires_grad is False
    assert student[2].base.weight.grad is None
    assert student[2].base.bias.grad is None

    # 2. Base values are exactly identical before vs after training
    assert student[0].base.weight.tolist() == orig_base0_w
    assert student[0].base.bias.tolist() == orig_base0_b
    assert student[2].base.weight.tolist() == orig_base2_w
    assert student[2].base.bias.tolist() == orig_base2_b

    # 3. Base Parameter identities are preserved
    assert id(student[0].base.weight) == base0_w_id
    assert id(student[0].base.bias) == base0_b_id

    # 4. Adapter weights were updated
    assert student[0].lora_B.tolist() != orig_adapter0_b


# =============================================================================
# Section 3: Periodic Checkpointing and Exact Resume
# =============================================================================

def test_mobile_trainer_lora_periodic_checkpointing_and_naming(temp_dir, active_backend):
    base = build_pretrained_base()
    student = wrap_student_with_lora(base)
    opt = optim.Adam(nn.adapter_parameters(student), lr=0.05)
    crit = nn.MSELoss()

    trainer = MobileTrainer(
        model=student,
        optimizer=opt,
        criterion=crit,
        checkpoint_dir=temp_dir,
        checkpoint_every_epochs=5,
        lora_only=True,
    )

    x = Tensor([[1.0, 0.0, -1.0, 0.5]])
    target = Tensor([[0.5, -0.5]])

    trainer.fit(dataset=(x, target), epochs=12)

    files = sorted(os.listdir(temp_dir))
    assert "checkpoint_epoch_5.json" in files
    assert "checkpoint_epoch_10.json" in files
    assert "checkpoint_latest.json" in files
    assert "checkpoint_epoch_1.json" not in files
    assert "checkpoint_epoch_12.json" not in files

    # Verify latest checkpoint is valid LoRA schema with trainer history
    meta = load_lora_checkpoint(os.path.join(temp_dir, "checkpoint_latest.json"))
    assert meta["epoch"] == 12
    assert meta["global_step"] == 12
    assert "_trainer_history" in meta["extra"]
    assert len(meta["extra"]["_trainer_history"]) == 12


def test_mobile_trainer_lora_exact_resume_and_history_continuity(temp_dir, active_backend):
    base = build_pretrained_base()
    student1 = wrap_student_with_lora(base)
    opt1 = optim.Adam(nn.adapter_parameters(student1), lr=0.05)
    crit = nn.MSELoss()

    trainer1 = MobileTrainer(
        model=student1,
        optimizer=opt1,
        criterion=crit,
        checkpoint_dir=temp_dir,
        checkpoint_every_epochs=5,
        lora_only=True,
    )

    x = Tensor([[0.5, -0.5, 1.0, -1.0]])
    target = Tensor([[1.0, 0.0]])

    trainer1.fit(dataset=(x, target), epochs=5)
    assert trainer1.current_epoch == 5
    assert trainer1.global_step == 5
    assert len(trainer1.history) == 5

    ckpt_5 = os.path.join(temp_dir, "checkpoint_epoch_5.json")

    # Resume into fresh trainer with identical pretrained base
    fresh_base = build_pretrained_base()
    student2 = wrap_student_with_lora(fresh_base)
    opt2 = optim.Adam(nn.adapter_parameters(student2), lr=0.05)

    trainer2 = MobileTrainer(
        model=student2,
        optimizer=opt2,
        criterion=crit,
        checkpoint_dir=temp_dir,
        checkpoint_every_epochs=5,
        lora_only=True,
    )

    res2 = trainer2.fit(dataset=(x, target), epochs=5, resume_from=ckpt_5)

    assert trainer2.current_epoch == 10
    assert trainer2.global_step == 10
    assert res2["epochs_completed"] == 10
    assert res2["global_step"] == 10
    assert len(res2["history"]) == 10

    # Verify history monotonicity
    for idx, item in enumerate(res2["history"]):
        assert item["epoch"] == idx + 1
        assert item["global_step"] == idx + 1


# =============================================================================
# Section 4: Failure Injection and Trainer-level Rollback
# =============================================================================

def test_mobile_trainer_lora_periodic_save_failure_policy(temp_dir, active_backend):
    base = build_pretrained_base()
    student = wrap_student_with_lora(base)
    opt = optim.Adam(nn.adapter_parameters(student), lr=0.05)
    crit = nn.MSELoss()

    trainer = MobileTrainer(
        model=student,
        optimizer=opt,
        criterion=crit,
        checkpoint_dir=temp_dir,
        checkpoint_every_epochs=2,
        lora_only=True,
    )

    x = Tensor([[1.0, 2.0, 3.0, 4.0]])
    target = Tensor([[0.0, 1.0]])

    # Successful first epoch
    trainer.fit(dataset=(x, target), epochs=1)
    assert trainer.current_epoch == 1
    assert trainer.global_step == 1

    # Inject failure on second epoch periodic save (os.fsync crash)
    with patch("os.fsync", side_effect=OSError("Simulated periodic disk fsync failure")):
        with pytest.raises(CheckpointError, match="Simulated periodic disk fsync failure"):
            trainer.fit(dataset=(x, target), epochs=1)

    # Step succeeded before save failed -> state accurately reflects completed computation
    assert trainer.global_step == 2
    assert trainer.current_epoch == 2


def test_mobile_trainer_lora_resume_failure_trainer_rollback(temp_dir, active_backend):
    base = build_pretrained_base()
    student = wrap_student_with_lora(base)
    opt = optim.Adam(nn.adapter_parameters(student), lr=0.05)
    crit = nn.MSELoss()

    trainer = MobileTrainer(model=student, optimizer=opt, criterion=crit, lora_only=True)
    trainer.current_epoch = 7
    trainer.global_step = 42
    trainer.history = [{"epoch": 7, "loss": 0.123}]

    # Create corrupted checkpoint file
    bad_path = os.path.join(temp_dir, "corrupted.json")
    with open(bad_path, "w", encoding="utf-8") as f:
        f.write('{"checksum": "invalid", "payload": {}}')

    with pytest.raises(CheckpointIntegrityError):
        trainer.resume(bad_path)

    # Trainer state is 100% rolled back to pre-call snapshot
    assert trainer.current_epoch == 7
    assert trainer.global_step == 42
    assert trainer.history == [{"epoch": 7, "loss": 0.123}]


def test_mobile_trainer_lora_format_mismatch_rejections(temp_dir, active_backend):
    crit = nn.MSELoss()

    # 1. Create generic checkpoint
    plain_model = build_pretrained_base()
    plain_opt = optim.Adam(plain_model.parameters(), lr=0.01)
    plain_trainer = MobileTrainer(model=plain_model, optimizer=plain_opt, criterion=crit, lora_only=False)
    generic_ckpt = os.path.join(temp_dir, "generic.json")
    plain_trainer.save(generic_ckpt)

    # 2. Create LoRA checkpoint
    student = wrap_student_with_lora(build_pretrained_base())
    student_opt = optim.Adam(nn.adapter_parameters(student), lr=0.01)
    lora_trainer = MobileTrainer(model=student, optimizer=student_opt, criterion=crit, lora_only=True)
    lora_ckpt = os.path.join(temp_dir, "lora.json")
    lora_trainer.save(lora_ckpt)

    # Rejection 1: LoRA trainer resuming generic checkpoint -> CheckpointError
    with pytest.raises(CheckpointError):
        lora_trainer.resume(generic_ckpt)

    # Rejection 2: Generic trainer resuming LoRA checkpoint -> CheckpointError
    with pytest.raises(CheckpointError):
        plain_trainer.resume(lora_ckpt)


# =============================================================================
# Section 5: Teacher-Student Domain Adaptation and Convergence
# =============================================================================

def test_mobile_trainer_lora_teacher_student_domain_adaptation(temp_dir, active_backend):
    random.seed(1234)

    # 1. Teacher model: Represents target adapted behavior (affine domain transformation)
    teacher = build_pretrained_base()

    # 2. Student base model: Identical pretrained base
    student_base = build_pretrained_base()
    student = wrap_student_with_lora(student_base, rank=2, alpha=4.0)
    opt = optim.Adam(nn.adapter_parameters(student), lr=0.08)
    crit = nn.MSELoss()

    # Generate synthetic training and evaluation datasets
    # Input points X in [-1.0, 1.0]
    train_x_data = [
        [0.8, -0.5, 0.3, -0.2],
        [-0.4, 0.9, -0.7, 0.1],
        [0.2, 0.3, 0.8, -0.9],
        [-0.6, -0.8, 0.4, 0.5],
        [0.1, -0.2, -0.3, 0.7],
    ]
    eval_x_data = [
        [0.5, -0.3, 0.1, 0.4],
        [-0.2, 0.6, -0.4, -0.1],
    ]

    train_x = Tensor(train_x_data)
    eval_x = Tensor(eval_x_data)

    # Teacher targets with domain shift delta
    shift_matrix = Tensor([[0.8, -0.3], [0.2, 0.7]])
    train_target = teacher(train_x) @ shift_matrix
    eval_target = teacher(eval_x) @ shift_matrix

    # Detach targets so teacher requires no grad
    train_target = Tensor(train_target.tolist(), requires_grad=False)
    eval_target = Tensor(eval_target.tolist(), requires_grad=False)

    # Measure pre-adaptation initial losses
    initial_train_loss = crit(student(train_x), train_target).item()
    initial_eval_loss = crit(student(eval_x), eval_target).item()

    assert initial_train_loss > 1e-3
    assert initial_eval_loss > 1e-3

    trainer = MobileTrainer(
        model=student,
        optimizer=opt,
        criterion=crit,
        checkpoint_dir=temp_dir,
        checkpoint_every_epochs=10,
        lora_only=True,
    )

    # Train for 40 epochs
    res = trainer.fit(dataset=(train_x, train_target), epochs=40)

    final_train_loss = res["history"][-1]["loss"]
    final_eval_loss = crit(student(eval_x), eval_target).item()

    # Assert significant convergence (>90% loss reduction)
    assert final_train_loss < initial_train_loss * 0.10
    assert final_eval_loss < initial_eval_loss * 0.15


# =============================================================================
# Section 6: Continuous vs Interrupted Training Equivalence
# =============================================================================

def test_mobile_trainer_lora_continuous_vs_interrupted_equivalence(temp_dir, active_backend):
    x = Tensor([[0.2, -0.4, 0.6, -0.8], [-0.3, 0.5, -0.7, 0.9]])
    target = Tensor([[0.8, -0.2], [-0.5, 0.4]])

    # 1. Run Continuous: 20 epochs in one pass
    base_cont = build_pretrained_base()
    student_cont = wrap_student_with_lora(base_cont, rank=2, alpha=2.0)
    opt_cont = optim.Adam(nn.adapter_parameters(student_cont), lr=0.05)
    crit = nn.MSELoss()

    trainer_cont = MobileTrainer(model=student_cont, optimizer=opt_cont, criterion=crit, lora_only=True)
    res_cont = trainer_cont.fit(dataset=(x, target), epochs=20)

    # 2. Run Interrupted: 10 epochs -> save -> fresh model -> resume -> 10 epochs
    base_int = build_pretrained_base()
    student_int = wrap_student_with_lora(base_int, rank=2, alpha=2.0)
    opt_int = optim.Adam(nn.adapter_parameters(student_int), lr=0.05)

    trainer_int1 = MobileTrainer(
        model=student_int,
        optimizer=opt_int,
        criterion=crit,
        checkpoint_dir=temp_dir,
        checkpoint_every_epochs=10,
        lora_only=True,
    )
    trainer_int1.fit(dataset=(x, target), epochs=10)

    ckpt_10 = os.path.join(temp_dir, "checkpoint_epoch_10.json")

    # Fresh resume instance
    fresh_base = build_pretrained_base()
    student_resumed = wrap_student_with_lora(fresh_base, rank=2, alpha=2.0)
    opt_resumed = optim.Adam(nn.adapter_parameters(student_resumed), lr=0.05)

    trainer_resumed = MobileTrainer(
        model=student_resumed,
        optimizer=opt_resumed,
        criterion=crit,
        checkpoint_dir=temp_dir,
        lora_only=True,
    )
    res_int = trainer_resumed.fit(dataset=(x, target), epochs=10, resume_from=ckpt_10)

    # 3. Compare counters and history
    assert res_cont["epochs_completed"] == res_int["epochs_completed"] == 20
    assert res_cont["global_step"] == res_int["global_step"] == 20
    assert len(res_cont["history"]) == len(res_int["history"]) == 20

    # 4. Compare final adapter parameter values
    for p_c, p_i in zip(nn.adapter_parameters(student_cont), nn.adapter_parameters(student_resumed)):
        assert p_c.flatten().tolist() == pytest.approx(p_i.flatten().tolist(), abs=1e-5, rel=1e-5)

    # 5. Compare forward output
    out_cont = student_cont(x)
    out_int = student_resumed(x)
    assert out_cont.flatten().tolist() == pytest.approx(out_int.flatten().tolist(), abs=1e-5, rel=1e-5)


# =============================================================================
# Section 7: Cross-Backend Resume
# =============================================================================

def test_mobile_trainer_lora_cross_backend_resume(temp_dir):
    if "numpy" not in available_backends():
        pytest.skip("NumPy backend not available for cross-backend test")

    x_data = [[0.5, -0.3, 0.8, -0.2], [-0.1, 0.7, -0.4, 0.6]]
    tgt_data = [[1.0, 0.0], [0.0, 1.0]]

    # Step A: Python training for 5 epochs -> Save LoRA checkpoint
    set_backend("python")
    py_base = build_pretrained_base()
    py_student = wrap_student_with_lora(py_base, rank=2, alpha=2.0)
    py_opt = optim.Adam(nn.adapter_parameters(py_student), lr=0.05)
    crit = nn.MSELoss()

    py_trainer = MobileTrainer(
        model=py_student,
        optimizer=py_opt,
        criterion=crit,
        checkpoint_dir=temp_dir,
        checkpoint_every_epochs=5,
        lora_only=True,
    )
    py_trainer.fit(dataset=(Tensor(x_data), Tensor(tgt_data)), epochs=5)

    ckpt_path = os.path.join(temp_dir, "checkpoint_epoch_5.json")

    # Step B: NumPy resume with matching base weights -> Continue 5 epochs
    set_backend("numpy")
    np_base = build_pretrained_base()
    np_student = wrap_student_with_lora(np_base, rank=2, alpha=2.0)
    np_opt = optim.Adam(nn.adapter_parameters(np_student), lr=0.05)

    np_trainer = MobileTrainer(
        model=np_student,
        optimizer=np_opt,
        criterion=crit,
        checkpoint_dir=temp_dir,
        lora_only=True,
    )
    res_np = np_trainer.fit(
        dataset=(Tensor(x_data), Tensor(tgt_data)),
        epochs=5,
        resume_from=ckpt_path,
    )

    assert res_np["epochs_completed"] == 10
    assert res_np["global_step"] == 10
    assert len(res_np["history"]) == 10


# =============================================================================
# Section 8: Deployment Merge Lifecycle and Guards
# =============================================================================

def test_mobile_trainer_lora_deployment_merge_and_guards(temp_dir, active_backend):
    base = build_pretrained_base()
    student = wrap_student_with_lora(base, rank=2, alpha=2.0)
    opt = optim.Adam(nn.adapter_parameters(student), lr=0.05)
    crit = nn.MSELoss()

    trainer = MobileTrainer(model=student, optimizer=opt, criterion=crit, lora_only=True)
    x = Tensor([[0.5, -0.5, 1.0, -1.0]])
    target = Tensor([[1.0, -1.0]])

    trainer.fit(dataset=(x, target), epochs=5)

    # 1. Pre-merge inference evaluation
    pre_merge_out = student(x).flatten().tolist()
    orig_base0_w = copy.deepcopy(student[0].base.weight.tolist())

    # 2. Transactional merge
    nn.merge_lora_adapters(student)
    assert student[0].merged is True
    assert student[2].merged is True

    # 3. Post-merge inference parity
    post_merge_out = student(x).flatten().tolist()
    assert pre_merge_out == pytest.approx(post_merge_out, abs=1e-5, rel=1e-5)

    # 4. Merged state guards: trainer must reject fit, save, and resume
    with pytest.raises(RuntimeError, match="require all adapters to be unmerged"):
        trainer.fit(dataset=(x, target), epochs=1)

    with pytest.raises(RuntimeError, match="require all adapters to be unmerged"):
        trainer.save(os.path.join(temp_dir, "merged_save.json"))

    # 5. Unmerge restores original base and permits training resumption
    nn.unmerge_lora_adapters(student)
    assert student[0].merged is False
    assert student[0].base.weight.tolist() == orig_base0_w

    # Fit resumes without error
    res = trainer.fit(dataset=(x, target), epochs=2)
    assert res["epochs_completed"] == 7


# =============================================================================
# Section 9: Callbacks, Stop Requests, and Reentrant State Machine
# =============================================================================

def test_mobile_trainer_lora_callbacks_stop_and_reentrancy(temp_dir, active_backend):
    base = build_pretrained_base()
    student = wrap_student_with_lora(base)
    opt = optim.Adam(nn.adapter_parameters(student), lr=0.01)
    crit = nn.MSELoss()

    trainer = MobileTrainer(model=student, optimizer=opt, criterion=crit, lora_only=True)
    x = Tensor([[1.0, 2.0, 3.0, 4.0]])
    target = Tensor([[0.5, -0.5]])

    step_events = []
    epoch_events = []

    def on_step(info):
        step_events.append(info)

    def on_epoch(info):
        epoch_events.append(info)

    # 1. Callback execution tracking
    trainer.fit(dataset=(x, target), epochs=3, on_step_end=on_step, on_epoch_end=on_epoch)
    assert len(step_events) == 3
    assert len(epoch_events) == 3
    assert step_events[0]["global_step"] == 1
    assert epoch_events[0]["epoch"] == 1

    # 2. Stop request gracefully stops training
    stop_student = wrap_student_with_lora(build_pretrained_base())
    stop_opt = optim.Adam(nn.adapter_parameters(stop_student), lr=0.01)
    stop_trainer = MobileTrainer(model=stop_student, optimizer=stop_opt, criterion=crit, lora_only=True)

    def stop_on_step(info):
        if info["global_step"] == 2:
            stop_trainer.request_stop()

    res_stop = stop_trainer.fit(dataset=(x, target), epochs=5, on_step_end=stop_on_step)
    assert res_stop["stopped_early"] is True
    assert res_stop["global_step"] == 2

    # 3. Reentrant fit call rejection
    reentrant_trainer = MobileTrainer(model=student, optimizer=opt, criterion=crit, lora_only=True)
    def reentrant_step(info):
        reentrant_trainer.fit(dataset=(x, target), epochs=1)

    with pytest.raises(RuntimeError, match="Reentrant fit"):
        reentrant_trainer.fit(dataset=(x, target), epochs=2, on_step_end=reentrant_step)
