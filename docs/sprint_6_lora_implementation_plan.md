# Sprint 6 Implementation Plan: On-Device LoRA Adapter

## Overview
Sprint 6 implements parameter-efficient fine-tuning through native On-Device Low-Rank Adaptation (LoRA) for `termux-train`.
The implementation freezes a base Linear projection:
\[
W \in \mathbb{R}^{d_{\mathrm{in}} \times d_{\mathrm{out}}}
\]
and trains low-rank adapter factors:
\[
A \in \mathbb{R}^{d_{\mathrm{in}} \times r},
\qquad
B \in \mathbb{R}^{r \times d_{\mathrm{out}}}
\]
with:
\[
\gamma = \frac{\alpha}{r}
\]
No mandatory dependency on PyTorch, PEFT, transformers, or bitsandbytes is introduced.

---

## Architectural Principles and Strict Rules

### 1. Weight Shape Convention
The implementation follows the native `termux-train` Linear convention:
- `base.weight.shape == (in_features, out_features)`
- `lora_A.shape == (in_features, rank)`
- `lora_B.shape == (rank, out_features)`
- `scaling == alpha / rank`

### 2. Forward Formulation
While unmerged:
\[
\operatorname{output}
=
\operatorname{base}(x)
+
((x @ A) @ B)\operatorname{scaling}
\]
While merged, the adapter delta is already applied to the base weight, so forward returns `base(x)` without applying the adapter path again.

### 3. Initialization Contract
`lora_A` is initialized from:
\[
A_{ij}
\sim
\mathcal{U}
\left(
-\frac{1}{\sqrt{\mathrm{in\_features}}},
\frac{1}{\sqrt{\mathrm{in\_features}}}
\right)
\]
`lora_B` is initialized to exact zeros:
\[
B = 0
\]
Therefore:
\[
\Delta W = A @ B = 0
\]
at initialization, ensuring initial output identity under the tested numerical contract.

### 4. Base Parameter Invariance
- `base.weight.requires_grad = False`
- If `base.bias is not None`, `base.bias.requires_grad = False`
- Base gradients are not created:
  - `base.weight.grad is None`
  - `base.bias.grad is None` when bias exists
- Base parameter values remain unchanged across adapter-only optimizer steps.
- Base and adapter `Parameter` object identities remain stable.

### 5. Adapter-only Optimization Contract
LoRA optimizers must use:
- `layer.adapter_parameters()` for one `LoRALinear`
- `adapter_parameters(model)` for a nested model
Generic `module.parameters()` must not be used as the primary LoRA fine-tuning parameter source.
The adapter parameter count is:
\[
r(d_{\mathrm{in}} + d_{\mathrm{out}})
\]

### 6. Backend Policy
- Pure Python fallback is mandatory.
- NumPy acceleration is optional.
- Adapter state must be portable between PythonBackend and NumPyBackend.
- Loading state must not change the target parameter backend.
- Cross-backend trainable Tensor operations remain prohibited.

### 7. External Dependency Policy
The runtime must not depend on:
- `torch`
- `peft`
- `transformers`
- `bitsandbytes`

---

## Phase 1: SCRUM-308 - LoRALinear Core
- **Status**: Host Complete
- **Commit**: `4b016ea`
- **Commit Message**: `Add frozen-base LoRALinear core`
- **Host Tests**: `336 passed, 1 warning`
- **Android Termux Gate**: `PENDING`

Implemented:
- `LoRALinear`
- `LoRALinear.from_linear()`
- instance adapter parameter helpers
- recursive adapter parameter helpers
- strict constructor validation
- frozen base parameters
- 1D, 2D, and 3D forward support
- PythonBackend and NumPyBackend parity tests

---

## Phase 2: SCRUM-309 - Adapter-only State Serialization
- **Status**: Host Complete
- **Base Commit**: `881bce2` (`Add atomic LoRA adapter state serialization`)
- **Hardening Focus**: Crash-resilient commit-failure rollback, strict key/metadata type validation
- **Host Tests**: `359 passed, 1 warning`
- **Android Termux Gate**: `PENDING`

Implemented & Hardened:
- `LoRALinear.adapter_state_dict()`
- `LoRALinear.load_adapter_state_dict(state_dict, strict=True)`
- `adapter_state_dict(module)`
- `load_adapter_state_dict(module, state_dict, strict=True)`
- Validation-atomic & commit-failure-atomic rollback guarantees
- Single-layer pre-commit native snapshot and exception rollback
- Recursive multi-layer pre-commit snapshot and exception rollback
- Strict metadata bool, type, finite, and value checking (`in_features`, `out_features`, `rank`, `alpha`)
- Container schema validation (`adapters` dict check, string key enforcement)
- Parameter identity (`id(lora_A)`, `id(lora_B)`, `id(base.weight)`, `id(base.bias)`) preservation
- `requires_grad` and optimizer parameter reference preservation
- PythonBackend ↔ NumPyBackend cross-backend portability

---

## Phase 3: SCRUM-310 - Transactional Merge and Unmerge
- **Status**: Current Task
Implement:
- `merge()`
- `unmerge()`
- exact pre-merge base snapshot
- strict merge lifecycle policy
- atomic failure rollback
- Parameter identity preservation
- prevention of adapter double application

Commit:
`Add transactional LoRA merge lifecycle`

---

## Phase 4: SCRUM-311 - Safe LoRA Checkpoint Integration
Implement:
- unmerged-only adapter checkpoint policy
- adapter checkpoint schema
- atomic save and load
- corruption detection
- rollback integration

Commit:
`Integrate safe LoRA adapter checkpointing`

---

## Phase 5: SCRUM-312 - MobileTrainer and Toy Fine-tuning
Implement:
- adapter-only MobileTrainer flow
- teacher-student adaptation experiment
- convergence tests
- save and resume tests
- `examples/06_lora_adapter_training.py`

Commit:
`Add LoRA mobile fine-tuning example`

---

## SCRUM-308 Verification Result
- `py -3 -m pytest tests/test_lora.py -v`
  - pytest collection 기준 100 test cases passed
- `py -3 -m pytest tests/ -v`
  - `336 passed, 1 warning`
- `git diff --check`
  - PASS
- `git status --short`
  - CLEAN
- Android Termux Gate
  - PENDING
