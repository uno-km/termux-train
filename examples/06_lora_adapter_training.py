"""
examples/06_lora_adapter_training.py
====================================
Sprint 6 Milestone: On-Device LoRA Adapter Fine-Tuning & Deployment Merge Demo.
Demonstrates:
  1. Pre-trained Base Model setup with frozen weights
  2. LoRA low-rank adapter injection via LoRALinear.from_linear()
  3. Parameter efficiency analysis (Trainable Adapter vs Total Parameters)
  4. On-device fine-tuning with MobileTrainer (lora_only=True)
  5. Periodic atomic LoRA checkpointing (checkpoint_epoch_X.json)
  6. Process interruption simulation & exact transactional resume from checkpoint_latest.json
  7. Teacher-Student domain adaptation convergence verification
  8. Base weight invariance validation
  9. Transactional merge for zero-overhead inference deployment
"""

import sys
import os
import shutil
import tempfile
import random
import copy
import math

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError) as rec_err:
        sys.stderr.write(f'[termux-train] Notice: stream reconfigure failed: {rec_err}\n')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from termux_train import Tensor, nn, optim, runtime, set_backend, get_backend, available_backends


def run_lora_finetuning_demo(backend_name: str):
    set_backend(backend_name)
    print("=" * 80)
    print(f"🎯 Running On-Device LoRA Adapter Fine-Tuning Demo on Backend: [{get_backend().name}]")
    print("=" * 80)

    random.seed(42)
    checkpoint_dir = tempfile.mkdtemp(prefix="termux_lora_demo_")

    try:
        # 1. Setup Deterministic Teacher (Target Domain Transformation) & Pre-trained Base
        print("\n▶️ [Step 1]: Setting up Pre-trained Base Model and Teacher Task...")
        base_model = nn.Sequential(
            nn.Linear(4, 16),
            nn.Tanh(),
            nn.Linear(16, 2),
        )
        pretrained_base_state = copy.deepcopy(base_model.state_dict())

        # Generate synthetic domain adaptation dataset (4D input -> 2D shifted target)
        train_x_data = [
            [0.5, -0.2, 0.8, -0.4],
            [-0.3, 0.7, -0.5, 0.2],
            [0.1, 0.4, 0.6, -0.8],
            [-0.6, -0.5, 0.3, 0.5],
            [0.2, -0.1, -0.4, 0.6],
        ]
        eval_x_data = [
            [0.4, -0.3, 0.2, 0.3],
            [-0.2, 0.5, -0.3, -0.1],
        ]

        train_x = Tensor(train_x_data)
        eval_x = Tensor(eval_x_data)

        # Domain transformation target
        shift_matrix = Tensor([[0.8, -0.4], [0.3, 0.7]])
        teacher_model = nn.Sequential(
            nn.Linear(4, 16),
            nn.Tanh(),
            nn.Linear(16, 2),
        )
        teacher_model.load_state_dict(pretrained_base_state)
        train_target = Tensor((teacher_model(train_x) @ shift_matrix).tolist(), requires_grad=False)
        eval_target = Tensor((teacher_model(eval_x) @ shift_matrix).tolist(), requires_grad=False)

        # 2. Inject LoRA Adapters (Rank=2, Alpha=4.0)
        print("\n▶️ [Step 2]: Injecting LoRA Low-Rank Adapters (Rank=2, Alpha=4.0)...")
        student = nn.Sequential(
            nn.LoRALinear.from_linear(base_model[0], rank=2, alpha=4.0),
            nn.Tanh(),
            nn.LoRALinear.from_linear(base_model[2], rank=2, alpha=4.0),
        )

        # Parameter counts
        total_params = sum(math.prod(p.shape) for p in student.parameters())
        adapter_params = sum(math.prod(p.shape) for p in nn.adapter_parameters(student))
        trainable_ratio = (adapter_params / total_params) * 100.0

        print(f"   ✓ Total Model Parameters: {total_params}")
        print(f"   ✓ Trainable LoRA Adapter Parameters: {adapter_params}")
        print(f"   ✓ Trainable Parameter Ratio: {trainable_ratio:.2f}% (Frozen Base: {100.0 - trainable_ratio:.2f}%)")

        # Snapshot base weights for invariance verification
        orig_base0_w = copy.deepcopy(student[0].base.weight.tolist())
        orig_base2_w = copy.deepcopy(student[2].base.weight.tolist())

        # 3. Measure Initial Pre-adaptation Performance
        criterion = nn.MSELoss()
        initial_train_loss = criterion(student(train_x), train_target).item()
        initial_eval_loss = criterion(student(eval_x), eval_target).item()
        print(f"\n📊 [Baseline Evaluation Before Fine-tuning]:")
        print(f"   ✓ Initial Train Loss: {initial_train_loss:.6f}")
        print(f"   ✓ Initial Eval Loss:  {initial_eval_loss:.6f}")

        # 4. Phase 1 Training: Initial 20 Epochs with MobileTrainer
        print("\n▶️ [Step 3]: Fine-tuning LoRA Adapters (Phase 1: 20 Epochs)...")
        optimizer = optim.Adam(nn.adapter_parameters(student), lr=0.08)
        trainer = runtime.MobileTrainer(
            model=student,
            optimizer=optimizer,
            criterion=criterion,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every_epochs=10,
            lora_only=True,
        )

        res1 = trainer.fit(dataset=(train_x, train_target), epochs=20)
        print(f"   ✓ Phase 1 Completed: Epoch {res1['epochs_completed']}, Global Step: {res1['global_step']}")
        print(f"   ✓ Phase 1 Loss: {res1['history'][-1]['loss']:.6f}")

        saved_files = sorted(os.listdir(checkpoint_dir))
        print(f"   ✓ Saved Checkpoint Files: {saved_files}")
        assert "checkpoint_epoch_10.json" in saved_files
        assert "checkpoint_epoch_20.json" in saved_files
        assert "checkpoint_latest.json" in saved_files

        # 5. Simulate Process Interruption & Memory Eviction
        print("\n⚡ [Step 4]: Simulating Process Interruption / Memory Eviction...")
        del trainer
        del student
        del optimizer

        # 6. Reconstruct Compatible Pre-trained Base and Resume Training
        print("\n🔄 [Step 5]: Reconstructing Base Model & Resuming from 'checkpoint_latest.json'...")
        fresh_base = nn.Sequential(
            nn.Linear(4, 16),
            nn.Tanh(),
            nn.Linear(16, 2),
        )
        fresh_base.load_state_dict(pretrained_base_state)
        resumed_student = nn.Sequential(
            nn.LoRALinear.from_linear(fresh_base[0], rank=2, alpha=4.0),
            nn.Tanh(),
            nn.LoRALinear.from_linear(fresh_base[2], rank=2, alpha=4.0),
        )
        resumed_opt = optim.Adam(nn.adapter_parameters(resumed_student), lr=0.08)

        resumed_trainer = runtime.MobileTrainer(
            model=resumed_student,
            optimizer=resumed_opt,
            criterion=criterion,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every_epochs=10,
            lora_only=True,
        )

        latest_ckpt = os.path.join(checkpoint_dir, "checkpoint_latest.json")
        res2 = resumed_trainer.fit(
            dataset=(train_x, train_target),
            epochs=20,
            resume_from=latest_ckpt,
        )

        print(f"   ✓ Phase 2 Resumed & Finished: Total Epochs: {res2['epochs_completed']}, Global Step: {res2['global_step']}")
        print(f"   ✓ Total History Length: {len(res2['history'])} epochs recorded")

        # 7. Post-adaptation Convergence Verification
        final_train_loss = res2["history"][-1]["loss"]
        final_eval_loss = criterion(resumed_student(eval_x), eval_target).item()
        train_reduction = ((initial_train_loss - final_train_loss) / initial_train_loss) * 100.0
        eval_reduction = ((initial_eval_loss - final_eval_loss) / initial_eval_loss) * 100.0

        print(f"\n📊 [Post-adaptation Evaluation After 40 Epochs]:")
        print(f"   ✓ Final Train Loss: {final_train_loss:.6f} ({train_reduction:.2f}% reduction)")
        print(f"   ✓ Final Eval Loss:  {final_eval_loss:.6f} ({eval_reduction:.2f}% reduction)")

        assert final_train_loss < initial_train_loss * 0.10, "Train loss failed to converge below 10% of initial loss"
        assert final_eval_loss < initial_eval_loss * 0.15, "Eval loss failed to converge below 15% of initial loss"

        # 8. Base Parameter Invariance Check
        print("\n🔒 [Step 6]: Verifying Base Weight Invariance...")
        assert resumed_student[0].base.weight.tolist() == orig_base0_w
        assert resumed_student[2].base.weight.tolist() == orig_base2_w
        print("   ✓ Base weights remained strictly immutable throughout all 40 training epochs!")

        # 9. Deployment Merge & Inference Parity Check
        print("\n🚀 [Step 7]: Executing Transactional Merge for Zero-Overhead Inference Deployment...")
        pre_merge_predictions = resumed_student(eval_x).flatten().tolist()

        nn.merge_lora_adapters(resumed_student)
        assert resumed_student[0].merged is True
        assert resumed_student[2].merged is True
        print("   ✓ All LoRA layers merged into base weight matrices (merged=True).")

        post_merge_predictions = resumed_student(eval_x).flatten().tolist()
        max_diff = max(abs(a - b) for a, b in zip(pre_merge_predictions, post_merge_predictions))
        print(f"   ✓ Pre-merge vs Post-merge Prediction Maximum Difference: {max_diff:.8e}")
        assert max_diff < 1e-5, "Merged model predictions differ from adapter model predictions"

        print(f"\n✅ LoRA Adapter Fine-Tuning & Deployment Lifecycle Completed Successfully on [{backend_name}]!")

    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


def main():
    print("=" * 80)
    print("📱 termux-train: On-Device LoRA Adapter Fine-Tuning & Deployment Demo")
    print("=" * 80)

    # 1. Pure Python Backend (Always run)
    run_lora_finetuning_demo("python")

    # 2. NumPy Backend (If available)
    if "numpy" in available_backends():
        print()
        run_lora_finetuning_demo("numpy")


if __name__ == "__main__":
    main()
