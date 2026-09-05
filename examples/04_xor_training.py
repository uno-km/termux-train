"""
examples/04_xor_training.py
===========================
Sprint 4 Milestone: Non-linear XOR Problem Optimization & Convergence Demo.
Verifies training convergence with nn.Sequential, Tanh/Sigmoid, MSELoss, and Adam/SGD/AdamW optimizers.
"""

import sys
import os
import random

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError) as rec_err:
        sys.stderr.write(f'[termux-train] Notice: stream reconfigure failed: {rec_err}\n')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from termux_train import Tensor, nn, optim, set_backend, get_backend, available_backends

def train_xor(backend_name: str, optimizer_factory, optimizer_name: str, max_epochs: int = 2000):
    set_backend(backend_name)
    print("=" * 70)
    print(f"🧠 Training XOR on Backend: [{get_backend().name}] | Optimizer: [{optimizer_name}]")
    print("=" * 70)

    # 1. Deterministic Seeding for reproducible initialization
    random.seed(42)

    # 2. XOR Dataset
    x = Tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    target = Tensor(
        [
            [0.0],
            [1.0],
            [1.0],
            [0.0],
        ]
    )

    # 3. Model: Non-linear MLP with Tanh & Sigmoid
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.Tanh(),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    )

    optimizer = optimizer_factory(model.parameters())
    criterion = nn.MSELoss()

    initial_loss = None
    final_epoch = max_epochs

    # Measure initial loss before any step
    init_pred = model(x)
    initial_loss = criterion(init_pred, target).item()
    print(f"   [Epoch    0/{max_epochs}] Initial Pre-Step Loss: {initial_loss:.6f}")

    # 4. Training Loop
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        prediction = model(x)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()

        # Post-step evaluation on the updated parameter state
        if epoch % 10 == 0 or epoch == 1:
            eval_pred = model(x)
            eval_loss = criterion(eval_pred, target).item()
            eval_classes = [1 if row[0] >= 0.5 else 0 for row in eval_pred.tolist()]

            if epoch % 200 == 0 or epoch == 1:
                print(f"   [Epoch {epoch:4d}/{max_epochs}] Post-Step MSE Loss: {eval_loss:.6f}")

            if eval_loss < 0.01 and eval_classes == [0, 1, 1, 0]:
                print(f"   🎯 Early stopping target reached at epoch {epoch}: Post-Step Loss = {eval_loss:.6f}")
                final_epoch = epoch
                break

    # 5. Final Evaluation on final model state
    final_pred = model(x)
    final_loss = criterion(final_pred, target).item()
    pred_values = [row[0] for row in final_pred.tolist()]
    target_values = [row[0] for row in target.tolist()]

    correct = 0
    print("\n   [Final Predictions vs Targets on Post-Step Model]:")
    for i, (pv, tv) in enumerate(zip(pred_values, target_values)):
        p_class = 1 if pv >= 0.5 else 0
        t_class = int(tv)
        is_ok = (p_class == t_class)
        if is_ok:
            correct += 1
        print(f"   - Input {x.tolist()[i]} -> Pred: {pv:.4f} (Class: {p_class}) | Target: {t_class} [{'PASS' if is_ok else 'FAIL'}]")

    accuracy = correct / len(target_values)
    print(f"\n   📊 Epochs: {final_epoch} | Initial Loss: {initial_loss:.6f} -> Final Loss: {final_loss:.6f}")
    print(f"   📊 Accuracy: {accuracy * 100:.1f}%")

    assert final_loss < initial_loss, "Training failed to reduce loss"
    assert final_loss < 0.03, f"Final loss ({final_loss}) above convergence threshold 0.03"
    assert accuracy == 1.0, f"Accuracy ({accuracy}) not 100%"
    print(f"   ✅ [{get_backend().name} - {optimizer_name}] XOR Convergence 100% SUCCESS!\n")

def main():
    for b in ["python"] + (["numpy"] if "numpy" in available_backends() else []):
        # 1. Adam Convergence
        train_xor(
            backend_name=b,
            optimizer_name="Adam(lr=0.05)",
            optimizer_factory=lambda params: optim.Adam(params, lr=0.05),
        )
        # 2. SGD with Momentum Convergence
        train_xor(
            backend_name=b,
            optimizer_name="SGD(lr=0.5, momentum=0.9)",
            optimizer_factory=lambda params: optim.SGD(params, lr=0.5, momentum=0.9),
        )
        # 3. AdamW Convergence
        train_xor(
            backend_name=b,
            optimizer_name="AdamW(lr=0.05, weight_decay=1e-4)",
            optimizer_factory=lambda params: optim.AdamW(params, lr=0.05, weight_decay=1e-4),
        )

if __name__ == "__main__":
    main()
