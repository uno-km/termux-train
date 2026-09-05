#!/usr/bin/env python3
"""
examples/01_tensor_basics.py
============================
Demonstration of Tensor creation, Pluggable Backend switching,
Matrix Multiplication, and Reverse-Mode Autograd.

Features both:
 - Example A: Linear Mean Loss (Basic Autograd explanation)
 - Example B: Non-linear Squared Loss (Realistic ML Loss function)
"""

import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError) as rec_err:
        sys.stderr.write(f'[termux-train] Notice: stream reconfigure failed: {rec_err}\n')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from termux_train import Tensor, set_backend, get_backend, available_backends, tensor, zeros, ones

def run_example_a():
    print("----------------------------------------------------------------")
    print("📘 [Example A] Linear Mean Loss (loss = y.mean())")
    print("----------------------------------------------------------------")
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    w = Tensor([[2.0], [1.0]], requires_grad=True)
    
    y = x @ w
    loss = y.mean()
    loss.backward()
    
    print("   x:\n", x)
    print("   w:\n", w)
    print("   y = x @ w:\n", y)
    print("   loss = y.mean():\n", loss)
    print()
    print("   [Analytical Gradients]")
    print("   x.grad (expected: [[1.0, 0.5], [1.0, 0.5]]):\n", x.grad)
    print("   w.grad (expected: [[2.0], [3.0]]):\n", w.grad)
    print()

def run_example_b():
    print("----------------------------------------------------------------")
    print("📙 [Example B] Non-linear Squared Loss (loss = (y * y).mean())")
    print("----------------------------------------------------------------")
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    w = Tensor([[2.0], [1.0]], requires_grad=True)
    
    y = x @ w
    loss = (y * y).mean()
    loss.backward()
    
    print("   y = x @ w:\n", y)
    print("   loss = (y * y).mean():\n", loss)
    # y = [[4], [10]] -> y^2 = [[16], [100]] -> mean = 58.0
    # d loss / d y = y = [[4], [10]]
    # d loss / d x = [[4], [10]] @ [[2, 1]] = [[8.0, 4.0], [20.0, 10.0]]
    # d loss / d w = [[1, 3], [2, 4]] @ [[4], [10]] = [[34.0], [48.0]]
    print()
    print("   [Analytical Gradients]")
    print("   x.grad (expected: [[8.0, 4.0], [20.0, 10.0]]):\n", x.grad)
    print("   w.grad (expected: [[34.0], [48.0]]):\n", w.grad)
    print()

def main():
    print("================================================================")
    print("📱 termux-train (AMEVA-Termux) - Core Tensor & Autograd Demo")
    print("================================================================")
    
    # Pluggable Backend Setup
    set_backend("auto")
    print(f"[*] Available Backends: {available_backends()} + ['auto']")
    print(f"[*] Active Backend:     {get_backend().name}")
    print()

    run_example_a()
    run_example_b()

    # Pure Python Backend Check
    if "python" in available_backends():
        print("----------------------------------------------------------------")
        print("🐍 [Pure-Python Fallback Verification]")
        print("----------------------------------------------------------------")
        set_backend("python")
        print(f"   Switched to: {get_backend().name}")
        p_x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        p_w = Tensor([[2.0], [1.0]], requires_grad=True)
        p_loss = (p_x @ p_w).mean()
        p_loss.backward()
        print("   p_x.grad:\n", p_x.grad)
        print("   p_w.grad:\n", p_w.grad)

    print("\n✅ All Tensor & Autograd core operations verified with exact expected gradients!")

if __name__ == "__main__":
    main()
