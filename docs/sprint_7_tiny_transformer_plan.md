# Sprint 7 Implementation Plan: Tiny Transformer & CharLM Toy Engine

## Baseline Confirmation
- **Start Baseline**: `2179a4ab8d6f71626d6c7cc0e6a105d5cbe22394` (`Record SCRUM-313 implementation commit`)
- **Host Test Baseline**: `486 passed, 1 warning`
- **Working Tree**: `CLEAN`
- **Branch**: `main`
- **Sprint 6 Status**: `Host Complete, Android Device Validation Pending`
- **Current Stage**: `Sprint 7 Host In Progress (Tiny Transformer & CharLM, Device Validation Pending)`

---

## 1. Architectural Foundation Decisions (ADR)

### 1.1 ADR 7.1: Tensor Dtype & Non-Differentiable Integer Tensor Foundation (Gate 7.1)
- **Problem**: In current `PythonBackend` and `NumPyBackend`, all Tensor data is implicitly cast to `float` / `np.float32`. Therefore, `Tensor([1, 2])` and `Tensor([1.0, 2.0])` are indistinguishable post-construction, preventing runtime validation of integer-only indices in `nn.Embedding` and class targets in `nn.CrossEntropyLoss`.
- **Decision (Strict Dtype System)**:
  - Add explicit `dtype` attribute to `Tensor` (`dtype: str = "float32" | "int64" | "bool"`).
  - Integer (`int64`) and Boolean (`bool`) tensors strictly enforce `requires_grad=False`. Attempting `requires_grad=True` on non-float tensors raises `ValueError`.
  - `nn.Embedding` input strictly requires `int64` Tensor.
  - `nn.CrossEntropyLoss` target strictly requires `int64` Tensor.
  - Serialization, factories (`zeros`, `ones`, `tensor`), and `tolist()` preserve explicit dtype.

### 1.2 ADR 7.2: Transpose API Semantics & Axis Manipulation
- **Contract**:
  - Preserve existing `Tensor.transpose(*axes)` contract (expects full permutation of length `ndim`).
  - Introduce `Tensor.swapaxes(dim0, dim1)` as an explicit 2-axis swap helper.
  - In Multi-Head Attention, explicitly use `x.transpose(0, 2, 1, 3)` for 4D shape permutations without ambiguity.

### 1.3 ADR 7.3: Generalized N-D Batched Matmul with Unbroadcasting Backward
- **Forward Contract**:
  - Operands $A$ of shape $(\dots, M, K)$ and $B$ of shape $(\dots, K, N)$ broadcast leading batch dimensions $(\dots)$ using standard right-aligned broadcasting.
  - 1D promotion rules: $1\text{D} @ 2\text{D} \to 1\text{D}$; $2\text{D} @ 1\text{D} \to 1\text{D}$; $1\text{D} @ 1\text{D} \to \text{scalar}$.
- **Backward Contract**:
  - Promoted computation: $dA_{\text{prom}} = G \times B^T$, $dB_{\text{prom}} = A^T \times G$.
  - Unbroadcasting: Broadcast batch axes are summed along broadcasted dimensions back to the exact shape of $A$ and $B$.

### 1.4 ADR 7.4: Numerical Stability & Attention Masking Contracts
- **LogSumExp & Softmax**:
  - `Tensor.max(axis, keepdims)`: Implements deterministic subgradient.
  - `Tensor.logsumexp(axis, keepdims)`: $m + \log\left(\sum \exp(x - m)\right)$ for numerical stability.
  - `Tensor.log_softmax(axis)`: $x - x.\text{logsumexp}(axis, \text{keepdims}=\text{True})$.
  - `Tensor.softmax(axis)`: $\exp(x.\text{log\_softmax}(axis))$.
- **Attention Masking**:
  - Additive finite sentinel ($-10^9$ for float64, $-10^4$ for float32) or masked softmax.
  - Requires at least one unmasked key per row; causal self-attention validates diagonal availability.

---

## 2. Sprint 7 Isolation Gates and Execution Order

```
Gate 7.0: SCRUM-313 - Lightweight Tokenizers Hardening (Complete)
  ├── BaseTokenizer with strict versioned JSON schema validation
  ├── CharTokenizer (exact round-trip for known vocab, unknown fallback)
  ├── ByteTokenizer (260-token fixed vocab, UTF-8 round-trip, strict decode error handling)
  ├── WordTokenizer (lossless regex lexer preserving whitespace/punctuation)
  └── Subprocess zero-dependency isolation test
  ↓
Gate 7.1: SCRUM-315A - Tensor Dtype Foundation
  ├── Tensor dtype property ("float32", "int64", "bool")
  ├── Non-differentiable int64/bool tensor invariants (requires_grad=False)
  └── Backend dtype preservation
  ↓
Gate 7.2: SCRUM-315 - Transformer Math Spec & Core Primitives
  ├── docs/tiny_transformer_spec.md
  ├── Generalized N-D Batched Matmul (Python & NumPy Backends)
  ├── exp, sqrt, max (subgradient), swapaxes
  └── logsumexp, log_softmax, softmax, causal masking
  ↓
Gate 7.3: SCRUM-314 - nn.Embedding Layer
  ├── int64 token index validation & out-of-bounds rejection
  └── Forward lookup & scatter-add backward gradient accumulation
  ↓
Gate 7.4: SCRUM-316A - nn.LayerNorm (Isolated Component)
  ├── Mean/Variance normalization over trailing dimension
  ├── Learnable gamma (ones), beta (zeros) parameters
  └── Analytical & autograd backward verification
  ↓
Gate 7.5: SCRUM-316B - nn.MultiHeadAttention (Isolated Component)
  ├── Q, K, V linear projections & head split/merge
  ├── Scaled dot-product attention with causal mask
  └── Output projection & shape assertion pipeline
  ↓
Gate 7.6: SCRUM-316C - nn.TransformerBlock (Isolated Component)
  ├── Pre-LN topology with Residual connections
  └── 2-Layer FeedForward MLP (Linear -> Tanh/ReLU -> Linear)
  ↓
Gate 7.7: SCRUM-317 - CharLM Autoregressive Language Model Demo
  ├── nn.CrossEntropyLoss (LogSumExp target gather)
  ├── Learned Positional Embedding
  ├── CharLM model architecture & greedy autoregressive text generation
  └── MobileTrainer integration & convergence verification
  ↓
Gate 7.8: SCRUM-318 & SCRUM-319 - DocFold Dataset Pipeline & Toy Trainer
  ├── DocFold JSONL dataset parser & grammar
  └── Sequence mapping toy trainer & on-device overfitting convergence
```

---

## 3. Current Phase Status

### Gate 7.0 / Phase 1: SCRUM-313 - Lightweight Tokenizer Interface (Host Complete)
- **Status**: Host Complete
- **Product Commit**: `f3e673d` (`Add deterministic lightweight tokenizers`)
- **Traceability Commit**: `2179a4a` (`Record SCRUM-313 implementation commit`)
- **Hardening Commit**: `b9a3640` (`Harden tokenization schema, decode semantics, and subprocess isolation`)
- **Host Tests**: `486 passed, 1 warning` (25 tokenization tests 100% PASS)
- **Jira Status**: `검토 중 (Ready for Device Validation)`
- **Android Termux Gate**: `PENDING`

### Gate 7.1 & Gate 7.2 / Phase 2: SCRUM-315 - Transformer Math Spec, Tensor Dtype & N-D Batched Matmul (Host Complete)
- **Status**: Host Complete
- **Product Commit**: `6a368c2` (`Add Tensor dtype foundation, iterative autograd, and generalized ND matmul`)
- **Traceability Commit**: `363bb9c` (`Record SCRUM-315 implementation commit`)
- **Hardening Commit**: `9a3ff76` (`Harden Dtype promotion, inplace mutation guards, IEEE 754 compliance, and atomic transactions`)
- **Lifecycle & Setup Commit**: `55b916a` (`Add Big-Tech autograd lifecycle: no_grad, in-flight DAG release, selective saving, and NumPy setup`)
- **Autograd Correctness Commit**: `747d26f` (`Harden Autograd correctness: 1D dot grad, monotonic version invalidation, tie subgradient, and thread-safe ContextVar`)
- **Audit Polish Commit**: `ab0f910` (`Harden Autograd: conditional closure definitions, max tie spec alignment, and leaf/multithread test coverage`)
- **Host Tests**: `552 passed, 6 warnings in 7.52s` (66 dtype, transformer math, hardening, lifecycle & correctness tests 100% PASS)
- **Test Evidence**: `reports/junit_test_report.xml`
- **Files**:
  - `docs/tiny_transformer_spec.md`
  - `termux_train/tensor.py`
  - `termux_train/backend/__init__.py`
  - `termux_train/backend/base.py`
  - `termux_train/backend/python_backend.py`
  - `termux_train/backend/numpy_backend.py`
  - `termux_train/nn/module.py`
  - `termux_train/nn/lora.py`
  - `termux_train/optim/optimizer.py`
  - `scripts/setup_termux.sh`
  - `tests/test_dtype.py`
  - `tests/test_transformer_math.py`
  - `tests/test_audit_hardening.py`
  - `tests/test_autograd_lifecycle.py`
  - `tests/test_autograd_correctness.py`
- **SCRUM-315 Ticket Stage**: `검토 중 (Ready for Device Validation)`
- **Sprint 7 Overall Stage**: `진행 중 (Host In Progress, Device Validation Pending)`
- **Android Termux Gate**: `PENDING`
