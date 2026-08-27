# Architecture Boundaries & Security Contracts

> **Standard:** AMEVA-Termux Engine Architecture Policy  
> **Target:** `termux-train` (Android Termux Pure Python/C-Extension ML Framework)

---

## 1. Zero-Heap Allocation & In-Place Computation Policy
- Core forward/backward passes on tensors MUST minimize dynamic heap allocation in execution hot loops.
- In-place gradient accumulation (`grad += ...`) is strictly preferred to avoid garbage collection pressure during mobile training runs.

## 2. Platform Isolation Boundaries
- `termux-train` operates strictly within unprivileged Android user space.
- Any root privilege escalation or privileged shell execution (`su`) is strictly prohibited.
- Native shared libraries (`.so`) MUST link only against Android Bionic libc and NDK system symbols.

## 3. Dependency Footprint
- Core autograd, layers, and optimizers must maintain zero external dependencies beyond NumPy.
- Optional backends (Safetensors, OpenBLAS) must degrade gracefully when unavailable.
