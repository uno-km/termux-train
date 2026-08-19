# 📦 Multi-Environment Library Installability Matrix

> **Project**: `termux-train` (AMEVA-Termux)
> **Core Principle**: **Zero Mandatory Heavy Binary Dependencies for termux-train MVP**

---

## 💻 1. Host PC Development Environment Matrix (Windows / Linux)
*로컬 개발 및 크로스 플랫폼 단위 테스트/검증 환경*

| Library | Status | Installed Version | MVP Importance | Purpose / Role |
| :--- | :--- | :--- | :--- | :--- |
| `numpy` | ✅ Installed | `2.4.4` | ⚪ 선택 (Optional) | Fast C-accelerated array operations (Tier-2 Backend) |
| `scipy` | ⚠️ Not Installed | `N/A` | ⚪ 선택 (Optional) | Scientific computing & special math functions |
| `pytest` | ✅ Installed | `9.1.1` | 🔴 필수 (MVP) | Automated unit testing suite |
| `tokenizers` | ⚠️ Not Installed | `N/A` | ⚪ 선택 (Optional) | Hugging Face Rust-based BPE/WordPiece Tokenizer |
| `sentencepiece` | ⚠️ Not Installed | `N/A` | ⚪ 선택 (Optional) | C++ based Tokenizer backend |
| `transformers` | ⚠️ Not Installed | `N/A` | ⚪ 선택 (Optional) | Hugging Face model architecture loading |
| `peft` | ⚠️ Not Installed | `N/A` | ⚪ 선택 (Optional) | Hugging Face Parameter-Efficient Fine-Tuning |
| `torch` | ✅ Installed | `2.7.1+cu118` | ⚪ 선택 (Optional) | Reference PyTorch (Desktop/Server Reference Only) |

---

## 📱 2. Android Termux Native Environment Matrix (Bionic arm64)
*실제 스마트폰 Termux 네이티브 실행 환경 기준 (No CUDA / Bionic libc)*

| Library | Termux Install Method & Availability | MVP Importance | Termux Architecture Strategy & Notes |
| :--- | :--- | :--- | :--- |
| `numpy` | ✅ pkg install python-numpy (Recommended) | ⚪ 선택 (Optional) | Official Termux repo provides pre-built binary. Fast SIMD vector ops. |
| `scipy` | ⚠️ pkg install python-scipy | ⚪ 선택 (Optional) | Heavy Fortran/BLAS build. Optional, not required for termux-train. |
| `pytest` | ✅ pip install pytest | 🔴 필수 (MVP) | Pure Python package. Runs cleanly inside Termux. |
| `tokenizers` | ❌ Requires Rust/Cargo on Termux (Heavy build) | ⚪ 선택 (Optional) | Replaced by termux-train built-in pure Python lightweight tokenizer. |
| `sentencepiece` | ❌ Requires cmake/clang build | ⚪ 선택 (Optional) | Replaced by termux-train built-in pure Python lightweight tokenizer. |
| `transformers` | ⚠️ pip install transformers --no-deps | ⚪ 선택 (Optional) | Optional. termux-train provides native tiny-transformer. |
| `peft` | ⚠️ pip install peft --no-deps | ⚪ 선택 (Optional) | Optional. termux-train provides native LoRALinear. |
| `torch` | ❌ 공식 배포 중단 (Zero-PyTorch Core) | ⚪ 선택 (Optional) | termux-train operates with 100% Zero-PyTorch Core dependency. |

---

## 🎯 Architecture Decision on Dependencies (ADR-001)

1. **`torch` 의존성 완전 제거 (100% Zero-PyTorch Core)**:
   - Host PC에 설치된 PyTorch(예: CUDA 11.8)는 순수 **알고리즘 비교 검증 레퍼런스용**이며, `termux-train` 런타임은 PyTorch에 일절 의존하지 않습니다.
2. **`numpy` Pluggable 가속 (Tier-2 Acceleration)**:
   - Termux에서는 `pkg install python-numpy`로 사전 빌드된 바이너리를 설치하여 즉시 C-level 가속을 활성화합니다.
   - NumPy가 없는 환경에서는 Pure-Python Fallback으로 100% 동일하게 동작합니다.
3. **독립 내장 모듈 (Self-Contained Modules)**:
   - 무거운 Rust 빌드가 필요한 `tokenizers` 대신, `termux-train` 내장 초경량 **Char/Word-level Tokenizer**를 기본 제공합니다.
