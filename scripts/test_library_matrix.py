#!/usr/bin/env python3
"""
scripts/test_library_matrix.py
==============================
Test and analyze library installability matrix across:
 1. Host PC Development Environment (Windows/Linux PC)
 2. Android Termux Native Environment (Bionic arm64)
Generates reports/library_install_matrix.md.
"""

import sys
import os
import importlib
import json

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError) as rec_err:
        sys.stderr.write(f'[termux-train] Notice: stream reconfigure failed: {rec_err}\n')

LIBRARIES_TO_CHECK = [
    {
        "name": "numpy",
        "role": "Fast C-accelerated array operations (Tier-2 Backend)",
        "essential_for_mvp": False,
        "termux_status": "✅ pkg install python-numpy (Recommended)",
        "termux_notes": "Official Termux repo provides pre-built binary. Fast SIMD vector ops."
    },
    {
        "name": "scipy",
        "role": "Scientific computing & special math functions",
        "essential_for_mvp": False,
        "termux_status": "⚠️ pkg install python-scipy",
        "termux_notes": "Heavy Fortran/BLAS build. Optional, not required for termux-train."
    },
    {
        "name": "pytest",
        "role": "Automated unit testing suite",
        "essential_for_mvp": True,
        "termux_status": "✅ pip install pytest",
        "termux_notes": "Pure Python package. Runs cleanly inside Termux."
    },
    {
        "name": "tokenizers",
        "role": "Hugging Face Rust-based BPE/WordPiece Tokenizer",
        "essential_for_mvp": False,
        "termux_status": "❌ Requires Rust/Cargo on Termux (Heavy build)",
        "termux_notes": "Replaced by termux-train built-in pure Python lightweight tokenizer."
    },
    {
        "name": "sentencepiece",
        "role": "C++ based Tokenizer backend",
        "essential_for_mvp": False,
        "termux_status": "❌ Requires cmake/clang build",
        "termux_notes": "Replaced by termux-train built-in pure Python lightweight tokenizer."
    },
    {
        "name": "transformers",
        "role": "Hugging Face model architecture loading",
        "essential_for_mvp": False,
        "termux_status": "⚠️ pip install transformers --no-deps",
        "termux_notes": "Optional. termux-train provides native tiny-transformer."
    },
    {
        "name": "peft",
        "role": "Hugging Face Parameter-Efficient Fine-Tuning",
        "essential_for_mvp": False,
        "termux_status": "⚠️ pip install peft --no-deps",
        "termux_notes": "Optional. termux-train provides native LoRALinear."
    },
    {
        "name": "torch",
        "role": "Reference PyTorch (Desktop/Server Reference Only)",
        "essential_for_mvp": False,
        "termux_status": "❌ 공식 배포 중단 (Zero-PyTorch Core)",
        "termux_notes": "termux-train operates with 100% Zero-PyTorch Core dependency."
    }
]

def check_host_libraries():
    results = []
    print("📦 Inspecting Host PC Library Environment...")
    for lib in LIBRARIES_TO_CHECK:
        item = dict(lib)
        try:
            mod = importlib.import_module(lib["name"])
            version = getattr(mod, "__version__", "Installed")
            item["host_status"] = "Installed"
            item["host_version"] = version
            item["host_success"] = True
            print(f" [+] {lib['name']:<15}: AVAILABLE (v{version})")
        except ImportError as e:
            item["host_status"] = "Not Installed"
            item["host_version"] = "N/A"
            item["host_success"] = False
            item["host_error"] = str(e)
            print(f" [-] {lib['name']:<15}: NOT FOUND ({e})")
        results.append(item)
    return results

def generate_report(results):
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "library_install_matrix.md")
    
    host_rows = []
    termux_rows = []
    for r in results:
        host_icon = "✅" if r["host_success"] else "⚠️"
        essential = "🔴 필수 (MVP)" if r["essential_for_mvp"] else "⚪ 선택 (Optional)"
        
        host_rows.append(
            f"| `{r['name']}` | {host_icon} {r['host_status']} | `{r['host_version']}` | {essential} | {r['role']} |"
        )
        termux_rows.append(
            f"| `{r['name']}` | {r['termux_status']} | {essential} | {r['termux_notes']} |"
        )
        
    md = f"""# 📦 Multi-Environment Library Installability Matrix

> **Project**: `termux-train` (AMEVA-Termux)
> **Core Principle**: **Zero Mandatory Heavy Binary Dependencies for termux-train MVP**

---

## 💻 1. Host PC Development Environment Matrix (Windows / Linux)
*로컬 개발 및 크로스 플랫폼 단위 테스트/검증 환경*

| Library | Status | Installed Version | MVP Importance | Purpose / Role |
| :--- | :--- | :--- | :--- | :--- |
{chr(10).join(host_rows)}

---

## 📱 2. Android Termux Native Environment Matrix (Bionic arm64)
*실제 스마트폰 Termux 네이티브 실행 환경 기준 (No CUDA / Bionic libc)*

| Library | Termux Install Method & Availability | MVP Importance | Termux Architecture Strategy & Notes |
| :--- | :--- | :--- | :--- |
{chr(10).join(termux_rows)}

---

## 🎯 Architecture Decision on Dependencies (ADR-001)

1. **`torch` 의존성 완전 제거 (100% Zero-PyTorch Core)**:
   - Host PC에 설치된 PyTorch(예: CUDA 11.8)는 순수 **알고리즘 비교 검증 레퍼런스용**이며, `termux-train` 런타임은 PyTorch에 일절 의존하지 않습니다.
2. **`numpy` Pluggable 가속 (Tier-2 Acceleration)**:
   - Termux에서는 `pkg install python-numpy`로 사전 빌드된 바이너리를 설치하여 즉시 C-level 가속을 활성화합니다.
   - NumPy가 없는 환경에서는 Pure-Python Fallback으로 100% 동일하게 동작합니다.
3. **독립 내장 모듈 (Self-Contained Modules)**:
   - 무거운 Rust 빌드가 필요한 `tokenizers` 대신, `termux-train` 내장 초경량 **Char/Word-level Tokenizer**를 기본 제공합니다.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
        
    print(f"\n✅ Dual-Environment library installability matrix saved to: {report_path}")

if __name__ == "__main__":
    res = check_host_libraries()
    generate_report(res)
