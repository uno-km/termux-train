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
- **Hardening Commit 1**: `1ad912f` (`Harden atomic LoRA adapter state loading`)
- **Hardening Commit 2**: `5eabce5` (`Harden LoRA model adapter container schema`)
- **Host Tests**: `363 passed, 1 warning`
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
- Model adapter container schema validation (`_validate_model_adapter_container`, string key enforcement, `version="1.0"`, single-layer container unwrap)
- Parameter identity (`id(lora_A)`, `id(lora_B)`, `id(base.weight)`, `id(base.bias)`) preservation
- `requires_grad` and optimizer parameter reference preservation
- PythonBackend ↔ NumPyBackend cross-backend portability

---

## Phase 3: SCRUM-310 - Transactional Merge and Unmerge
- **Status**: Host Complete
- **Base Commit**: `6971097` (`Add transactional LoRA merge lifecycle`)
- **Hardening Commit**: `45fd1da` (`Harden LoRA merge lifecycle invariants`)
- **Host Tests**: `401 passed, 1 warning`
- **Android Termux Gate**: `PENDING`

Implemented & Hardened:
- `LoRALinear.merge()` & `LoRALinear.unmerge()`
- `merge_lora_adapters(module)` & `unmerge_lora_adapters(module)` recursive module helpers
- Merged forward removes LoRA-specific projection operations and executes only the base linear path
- Forward parity is verified with explicit tolerance (`abs=1e-6, rel=1e-6`)
- Exact merge-time backend-native deep snapshot restoration (`_base_weight_snapshot`)
- Unmerge restores snapshot exactly under successful backend assignment (no delta subtraction drift)
- State-pair lifecycle invariant enforcement: rejects unmerged state with unexpected stale snapshot
- Snapshot shape validation before unmerge assignment across single and recursive paths
- Computed delta and merged_weight shape validation before merge commit
- Runtime validation of scaling factor (finite, strictly positive numeric scalar)
- Strict lifecycle transition policy: double merge rejection and invalid unmerge rejection
- Adapter double-application prevention during merged forward
- Transactional single-layer commit rollback and multi-layer model rollback
- Rollback assignment failure detection and explicit exception reporting
- Parameter identity preservation (`id(base.weight)`, `id(lora_A)`, `id(lora_B)`)
- Optimizer adapter reference preservation (`optimizer.params[0] is layer.lora_A`)
- Backend identity and `requires_grad` preservation
- Adapter mutation after merge isolation: exact base restoration verified
- Module deduplication (`visited_ids`) preventing double-merge on shared layers
- Empty module no-op guarantee

---

## Phase 4: SCRUM-311 - Safe LoRA Checkpoint Integration
- **Status**: Host Complete
- **Base Commit**: `834aba8` (`Integrate safe LoRA adapter checkpointing`)
- **Hardening Commit**: `Harden LoRA checkpoint contracts`
- **Host Tests**: `436 passed, 1 warning`
- **Android Termux Gate**: `PENDING`

Implemented & Hardened:
- `save_lora_checkpoint()` & `load_lora_checkpoint()` public APIs with complete combination matrix (A: model+optimizer, B: model only, C: metadata-only validation mode, D: model=None+optimizer rejected)
- Dedicated `termux-train-lora-checkpoint` schema version `1.0` with exact outer and payload key validation
- Adapter-only model state serialization via `adapter_state_dict(model)`
- Adapter-only optimizer identity and ordering validation via `adapter_parameters(model)`
- Strict unmerged-only policy on saving and loading with stale snapshot and corrupted state pair rejection
- Base weight, base bias, and merge snapshot exclusion from payload
- SHA-256 integrity verification over canonical JSON serialization (`sort_keys=True, separators=(',', ':'), allow_nan=False`)
- Crash-safe temp write (`<path>.tmp`), `flush`, `os.fsync`, and atomic `os.replace`
- Comprehensive file-system failure injection protection (serialization, open, write, flush, fsync, replace)
- Two-phase transactional load with full atomic rollback on adapter or optimizer restoration failure
- Multi-failure aggregation and exception chaining reporting with `CheckpointRollbackError`
- Cross-backend adapter checkpoint portability and next optimizer step numerical parity (PythonBackend ↔ NumPyBackend)
- Shared module deduplication, nested container support, and extra metadata deep-copy isolation
- Parameter identity, optimizer reference, and `requires_grad` preservation

---

## Phase 5: SCRUM-312 - MobileTrainer and Toy Fine-tuning
- **Status**: Current Task
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
