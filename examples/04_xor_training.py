"""
examples/04_xor_training.py
===========================
Sprint 4 Milestone: Non-linear XOR Problem Optimization & Convergence Demo.
Verifies training convergence with nn.Sequential, Tanh/Sigmoid, MSELoss, and Adam/SGD optimizers.
"""

import sys
import os
import random

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from termux_train import Tensor, nn, optim, set_backend, get_backend, available_backends

def train_xor(backend_name: str, optimizer_cls=optim.Adam, lr: float = 0.05, max_epochs: int = 2000):
    set_backend(backend_name)
    print("=" * 70)
    print(f"🧠 Training XOR on Backend: [{get_backend().name}] | Optimizer: [{optimizer_cls.__name__}]")
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

    optimizer = optimizer_cls(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    initial_loss = None
    final_loss = None

    # 4. Training Loop
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        prediction = model(x)
        loss = criterion(prediction, target)

        current_loss = loss.item()
        if initial_loss is None:
            initial_loss = current_loss

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == 1:
            print(f"   [Epoch {epoch:4d}/{max_epochs}] MSE Loss: {current_loss:.6f}")

        if current_loss < 0.01:
            print(f"   🎯 Early stopping target reached at epoch {epoch}: Loss = {current_loss:.6f}")
            final_loss = current_loss
            break

    if final_loss is None:
        final_loss = current_loss

    # 5. Evaluation & Accuracy Check
    preds = model(x)
    pred_values = [row[0] for row in preds.tolist()]
    target_values = [row[0] for row in target.tolist()]

    correct = 0
    print("\n   [Final Predictions vs Targets]:")
    for i, (pv, tv) in enumerate(zip(pred_values, target_values)):
        p_class = 1 if pv >= 0.5 else 0
        t_class = int(tv)
        is_ok = (p_class == t_class)
        if is_ok:
            correct += 1
        print(f"   - Input {x.tolist()[i]} -> Pred: {pv:.4f} (Class: {p_class}) | Target: {t_class} [{'PASS' if is_ok else 'FAIL'}]")

    accuracy = correct / len(target_values)
    print(f"\n   📊 Initial Loss: {initial_loss:.6f} -> Final Loss: {final_loss:.6f}")
    print(f"   📊 Accuracy: {accuracy * 100:.1f}%")

    assert final_loss < initial_loss, "Training failed to reduce loss"
    assert final_loss < 0.03, f"Final loss ({final_loss}) above convergence threshold 0.03"
    assert accuracy == 1.0, f"Accuracy ({accuracy}) not 100%"
    print(f"   ✅ [{get_backend().name} - {optimizer_cls.__name__}] XOR Convergence 100% SUCCESS!\n")

def main():
    for b in ["python"] + (["numpy"] if "numpy" in available_backends() else []):
        # 1. Adam Convergence
        train_xor(backend_name=b, optimizer_cls=optim.Adam, lr=0.05, max_epochs=2000)
        # 2. SGD with Momentum Convergence
        train_xor(backend_name=b, optimizer_cls=optim.SGD, lr=0.5, max_epochs=2000)

if __name__ == "__main__":
    main()
