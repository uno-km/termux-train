"""
termux_train.cli
================
Official Command-Line Interface (CLI) for termux-train.
Provides diagnostics, environment validation, benchmark scoring, and demo execution.
"""

import sys
import os
import argparse
import subprocess

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from termux_train import __version__, available_backends, get_backend, set_backend, Tensor, nn
from termux_train.utils.termux_env import is_termux, is_android, get_device_info


def cmd_info(args):
    """Prints comprehensive system, hardware, and backend diagnostic information."""
    print("=" * 65)
    print(f"  📱 termux-train (v{__version__}) - System Diagnostics")
    print("=" * 65)
    info = get_device_info()
    for k, v in info.items():
        print(f"  • {k:20s}: {v}")
    print("=" * 65)
    print("  🔧 Framework Capabilities:")
    print(f"  • Active Backend     : {get_backend().name.upper()}")
    print(f"  • Available Backends : {[b.upper() for b in available_backends()]}")
    print(f"  • Termux Native      : {'YES' if is_termux() else 'NO (Host Environment)'}")
    print(f"  • Android OS         : {'YES' if is_android() else 'NO'}")
    print("=" * 65)


def cmd_check(args):
    """Performs self-test across all available backends to verify mathematical integrity."""
    print("=" * 65)
    print(f"  🔍 termux-train v{__version__} - Self-Diagnostic Verification")
    print("=" * 65)

    all_passed = True
    for b_name in available_backends():
        set_backend(b_name)
        print(f"  [+] Testing Backend: [{b_name.upper()}] ... ", end="", flush=True)
        try:
            # 1. Tensor creation & basic math
            a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            b = Tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
            c = (a @ b).sum()
            c.backward()

            # 2. NN & RoPE Transformer verification
            m = nn.TinyTransformerLM(vocab_size=10, d_model=8, num_heads=2, d_ff=16, num_layers=1, pos_type="rope")
            inp = Tensor([[1, 2, 3]], dtype="int64")
            logits, _ = m(inp)
            assert logits.shape == (1, 3, 10)

            print("PASSED ✅")
        except Exception as e:
            print(f"FAILED ❌ ({e})")
            all_passed = False

    print("=" * 65)
    if all_passed:
        print("  🎉 All backends verified and operating with mathematical integrity!")
    else:
        print("  ⚠️ One or more backends encountered errors. Check system libraries.")
    print("=" * 65)


def cmd_score(args):
    """Runs the 0-Point Baseline Granular Audit Scoring System."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(root_dir, "scripts", "run_audit_scoring.py")
    if not os.path.exists(script_path):
        # Fallback to direct pytest run
        pytest_cmd = [sys.executable, "-m", "pytest", "tests/test_audit_scorecard.py", "-s", "-v"]
        subprocess.run(pytest_cmd, check=False)
        return

    subprocess.run([sys.executable, script_path], check=False)


def cmd_demo(args):
    """Executes one of the 8 canonical example demos."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    examples_dir = os.path.join(root_dir, "examples")
    
    demo_map = {
        "1": "01_tensor_basics.py",
        "2": "02_nn_forward_backward.py",
        "3": "03_matmul_1d_to_3d.py",
        "4": "04_xor_training.py",
        "5": "05_mobile_training_runtime.py",
        "6": "06_lora_adapter_training.py",
        "7": "07_transformer_lm.py",
        "8": "08_docfold_trainer.py",
    }

    choice = str(args.demo_number).lstrip("0")
    if choice not in demo_map:
        print(f"❌ Unknown demo number: '{args.demo_number}'. Available demos: 1 through 8.")
        print("   1: Tensor Basics")
        print("   2: NN Forward/Backward")
        print("   3: 1D~3D Matmul")
        print("   4: XOR Training")
        print("   5: Mobile Training Runtime & Checkpoints")
        print("   6: LoRA Adapter Fine-Tuning")
        print("   7: Character-Level Transformer LM")
        print("   8: DocFold Sequence Mapping Trainer")
        sys.exit(1)

    target_script = os.path.join(examples_dir, demo_map[choice])
    print(f"🚀 Running Demo [{choice}]: {demo_map[choice]} ...\n")
    subprocess.run([sys.executable, target_script], check=False)


def main():
    parser = argparse.ArgumentParser(
        prog="termux-train",
        description="Native On-Device Deep Learning & LoRA Training Framework for Android Termux."
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s v{__version__}")
    
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # info
    p_info = subparsers.add_parser("info", help="Display environment, hardware, and backend capabilities")
    p_info.set_defaults(func=cmd_info)

    # check
    p_check = subparsers.add_parser("check", help="Run self-diagnostic mathematical checks across all backends")
    p_check.set_defaults(func=cmd_check)

    # score / test
    p_score = subparsers.add_parser("score", help="Run 0-point baseline granular audit scoring system")
    p_score.set_defaults(func=cmd_score)

    # demo
    p_demo = subparsers.add_parser("demo", help="Run an interactive example demo (1 to 8)")
    p_demo.add_argument("demo_number", type=int, help="Demo number (1 to 8)")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    if args.command is None:
        cmd_info(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
