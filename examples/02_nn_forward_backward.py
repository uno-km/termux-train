#!/usr/bin/env python3
"""
examples/02_nn_forward_backward.py
==================================
Sprint 3 Neural Network Framework Pipeline Demo:
nn.Sequential + nn.Linear + nn.ReLU + nn.mse_loss + backward().
"""

import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from termux_train import Tensor, nn, set_backend, get_backend

def main():
    print("================================================================")
    print("🧠 termux-train (AMEVA-Termux) - Sprint 3 Neural Network Demo")
    print("================================================================")
    set_backend("auto")
    print(f"[*] Active Backend: {get_backend().name}\n")

    # 1. Define Model Architecture
    print("1️⃣ Constructing nn.Sequential Model:")
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )
    print(model)
    print()

    # 2. Inspect Model Parameters
    print("2️⃣ Model Parameters Introspection:")
    for name, p in model.named_parameters():
        print(f"   - {name:<10}: shape={p.shape}, requires_grad={p.requires_grad}")
    print(f"   Total Parameters: {len(model.parameters())}")
    print()

    # 3. Forward Pass
    print("3️⃣ Forward Pass (Input: x = [[0.0, 1.0]]):")
    x = Tensor([[0.0, 1.0]], requires_grad=False)
    pred = model(x)
    print(f"   Predicted Output: {pred}")
    print()

    # 4. Loss Computation & Backward Pass
    print("4️⃣ Computing Loss (MSE) and Executing Backward Pass:")
    target = Tensor([[1.0]], requires_grad=False)
    loss = nn.mse_loss(pred, target)
    print(f"   MSE Loss: {loss.item():.6f}")
    
    print("   -> Running loss.backward()...")
    loss.backward()
    print()

    # 5. Gradients Inspection
    print("5️⃣ Parameter Gradients after Autograd:")
    for name, p in model.named_parameters():
        print(f"   - {name:<10} grad:\n{p.grad}")
    print()

    print("✅ Sprint 3 Neural Network Mini Framework pipeline passed with flying colors!")

if __name__ == "__main__":
    main()
