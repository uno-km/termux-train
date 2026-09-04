"""
termux_train.runtime.benchmarker
=================================
Production-Grade Standalone Benchmark Runner — stdin JSON IPC Architecture.
Eliminates python -c inline script injection and ARG_MAX limits.
Open-Source under Apache License 2.0.
"""

import sys
import os
import time
import json
import argparse

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    # Non-fatal console encoding setup fallback
    pass

from termux_train import randn, get_backend


def run_benchmark(cfg: dict) -> None:
    """
    Executes GEMM and full autograd step benchmarks.
    dim and iters are validated integers, passed via stdin JSON — no shell injection path.
    """
    dim = int(cfg.get("dim", 256))
    iters = int(cfg.get("iters", 10))

    if not (2 <= dim <= 4096):
        raise ValueError(f"Invalid dim={dim}. Must be between 2 and 4096.")
    if not (1 <= iters <= 1000):
        raise ValueError(f"Invalid iters={iters}. Must be between 1 and 1000.")

    # Warmup
    a = randn((dim, dim), requires_grad=True)
    b = randn((dim, dim), requires_grad=True)
    c = (a @ b).sum()
    c.backward()

    # Forward GEMM Latency
    t0 = time.perf_counter()
    for _ in range(iters):
        z = a @ b  # noqa: F841
    gemm_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    # Full Autograd Step Latency (Forward + Backward)
    t0 = time.perf_counter()
    for _ in range(iters):
        a.grad = None
        b.grad = None
        out = (a @ b).sum()
        out.backward()
    autograd_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    # GFLOPS: 2 * N^3 (valid only for square matmul N×N @ N×N)
    safe_gemm_ms = max(gemm_ms, 1e-6)
    gflops = (2.0 * (dim ** 3)) / (safe_gemm_ms / 1000.0) / 1e9

    result = {
        "dimension": f"{dim}x{dim}",
        "iterations": iters,
        "backend": get_backend().name.upper(),
        "gemmLatencyMs": round(gemm_ms, 3),
        "autogradStepLatencyMs": round(autograd_ms, 3),
        "throughputGflops": round(gflops, 3)
    }
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description="termux-train production benchmark runner")
    parser.add_argument("--stdin-json", action="store_true", help="Read benchmark config from stdin JSON")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    if args.stdin_json:
        raw = sys.stdin.read()
        cfg = json.loads(raw)
    else:
        cfg = {"dim": args.dim, "iters": args.iters}

    try:
        run_benchmark(cfg)
    except Exception as exc:
        print(f"__FATAL__:{str(exc)}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
