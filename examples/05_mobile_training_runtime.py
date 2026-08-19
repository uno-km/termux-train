"""
examples/05_mobile_training_runtime.py
======================================
Sprint 5 Milestone: Mobile Training Runtime & Safe Checkpoint Save/Resume Demo.
Demonstrates training lifecycle with MobileTrainer, periodic atomic checkpoints,
interruption simulation, and seamless exact resume to full convergence.
"""

import sys
import os
import shutil
import tempfile
import random

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from termux_train import Tensor, nn, optim, runtime, set_backend, get_backend, available_backends

def run_mobile_runtime_demo(backend_name: str):
    set_backend(backend_name)
    print("=" * 75)
    print(f"📱 Running MobileTrainer & Safe Checkpoint Demo on Backend: [{get_backend().name}]")
    print("=" * 75)

    random.seed(42)
    checkpoint_dir = tempfile.mkdtemp(prefix="termux_train_ckpt_")

    try:
        # 1. XOR Dataset
        x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        target = Tensor([[0.0], [1.0], [1.0], [0.0]])

        # 2. Phase 1: Initial Training with MobileTrainer (Stop after Epoch 25)
        print("\n▶️ [Phase 1]: Initial Training (Target: 25 Epochs with Periodic Checkpointing)...")
        model = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        optimizer = optim.Adam(model.parameters(), lr=0.05)
        criterion = nn.MSELoss()

        trainer = runtime.MobileTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every_epochs=10,
        )

        res1 = trainer.fit(dataset=(x, target), epochs=25)
        print(f"   ✓ Phase 1 Completed: Epoch {res1['epochs_completed']}, Global Step: {res1['global_step']}")
        print(f"   ✓ Latest Loss: {res1['history'][-1]['loss']:.6f}")

        # Check saved checkpoint files
        saved_files = sorted(os.listdir(checkpoint_dir))
        print(f"   ✓ Checkpoint Directory Files: {saved_files}")
        assert "checkpoint_epoch_10.json" in saved_files
        assert "checkpoint_epoch_20.json" in saved_files
        assert "checkpoint_latest.json" in saved_files

        # 3. Phase 2: Simulate Process Interruption / Memory Eviction
        print("\n⚡ [Phase 2]: Simulating Process Interruption & Memory Eviction...")
        del trainer
        del model
        del optimizer

        # 4. Phase 3: Resume Training from Checkpoint
        print("\n🔄 [Phase 3]: Resuming Training from Checkpoint ('checkpoint_latest.json')...")
        fresh_model = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        fresh_optimizer = optim.Adam(fresh_model.parameters(), lr=0.05)

        resume_trainer = runtime.MobileTrainer(
            model=fresh_model,
            optimizer=fresh_optimizer,
            criterion=criterion,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every_epochs=10,
        )

        latest_ckpt = os.path.join(checkpoint_dir, "checkpoint_latest.json")
        res2 = resume_trainer.fit(
            dataset=(x, target),
            epochs=25,
            resume_from=latest_ckpt,
        )

        print(f"   ✓ Phase 3 Resumed & Finished: Total Epochs: {res2['epochs_completed']}, Global Step: {res2['global_step']}")
        print(f"   ✓ Final Resumed Loss: {res2['history'][-1]['loss']:.6f}")

        # 5. Final Evaluation
        final_pred = fresh_model(x)
        final_loss = criterion(final_pred, target).item()
        pred_vals = [row[0] for row in final_pred.tolist()]
        target_vals = [row[0] for row in target.tolist()]

        correct = 0
        print("\n   [Post-Resume Predictions vs Targets]:")
        for i, (pv, tv) in enumerate(zip(pred_vals, target_vals)):
            p_class = 1 if pv >= 0.5 else 0
            t_class = int(tv)
            is_ok = (p_class == t_class)
            if is_ok:
                correct += 1
            print(f"   - Input {x.tolist()[i]} -> Pred: {pv:.4f} (Class: {p_class}) | Target: {t_class} [{'PASS' if is_ok else 'FAIL'}]")

        accuracy = correct / len(target_vals)
        print(f"\n   📊 Final Loss: {final_loss:.6f} | Accuracy: {accuracy * 100:.1f}%")

        assert final_loss < 0.01, f"Final loss ({final_loss}) above threshold"
        assert accuracy == 1.0, f"Accuracy ({accuracy}) not 100%"
        print(f"   ✅ [{get_backend().name}] Mobile Training Runtime & Checkpoint Resume 100% SUCCESS!\n")

    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

def main():
    for b in ["python"] + (["numpy"] if "numpy" in available_backends() else []):
        run_mobile_runtime_demo(b)

if __name__ == "__main__":
    main()
