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
- **Deep Hardening Commit**: `9a74a95` (`Harden Core ML Compiler: all-neginf LogSumExp/Softmax NaN defense, 3-color cyclic DAG check, N-D Linear, and Fused CrossEntropyLoss`)
- **VersionCounter & Contract Commit**: `92ba6b9` (`Harden Autograd & Losses: _VersionCounter shared alias invalidation, strict loss shape/bounds contracts, and all-ignore handling`)
- **SCRUM-315 Ticket Stage**: `검토 중 (Ready for Device Validation)`
- **Android Termux Gate**: `PENDING`

### Phase 3: SCRUM-314 - Embedding Layer (nn.Embedding) (Host Complete)
- **Status**: Host Complete
- **Product Commit**: `5f3ede1` (`Implement nn.Embedding layer with backward gradient accumulation and padding_idx support (SCRUM-314)`)
- **Host Tests**: 11 passed (1D/2D/3D forward lookup, duplicate index gradient accumulation, padding_idx masking, state_dict roundtrip)
- **SCRUM-314 Ticket Stage**: `검토 중 (Ready for Device Validation)`
- **Android Termux Gate**: `PENDING`

### Phase 4: SCRUM-316 - Tiny Transformer Block (Attention + FeedForward) (Host Complete)
- **Status**: Host Complete
- **Product Commit**: `ef807c0` (`Implement LayerNorm, MultiHeadAttention, TransformerBlock, and TinyTransformerLM (SCRUM-316)`)
- **Host Tests**: 8 passed (LayerNorm, MultiHeadAttention with causal mask, FeedForward MLP, Pre-LN TransformerBlock, TinyTransformerLM)
- **SCRUM-316 Ticket Stage**: `검토 중 (Ready for Device Validation)`
- **Android Termux Gate**: `PENDING`

### Phase 5: SCRUM-317 - Character-Level Autoregressive LM Demo (Host Complete)
- **Status**: Host Complete
- **Product Commit**: `90e497c` (`Add Character-Level Autoregressive LM Demo with AdamW and CrossEntropyLoss (SCRUM-317)`)
- **Demo Script**: `examples/05_transformer_lm.py` (Loss: 3.2729 -> 0.4184 in 12.64s, text generation validated)
- **SCRUM-317 Ticket Stage**: `검토 중 (Ready for Device Validation)`
- **Android Termux Gate**: `PENDING`

### Phase 6: SCRUM-318 - DocFold Toy Dataset Pipeline (Host Complete)
- **Status**: Host Complete
- **Product Commit**: `6954b05` (`Implement DocFold toy dataset pipeline with JSONL streaming and batch generator (SCRUM-318)`)
- **Host Tests**: 3 passed (DocFoldRecord serialization, JSONL roundtrip, padded batch generation)
- **SCRUM-318 Ticket Stage**: `검토 중 (Ready for Device Validation)`
- **Android Termux Gate**: `PENDING`

### Phase 7: SCRUM-319 - DocFold Sequence Mapping Toy Trainer (Host Complete)
- **Status**: Host Complete
- **Product Commit**: `49eeb71` (`Add DocFold Sequence Mapping Toy Trainer with TinyTransformerLM (SCRUM-319)`)
- **Demo Script**: `examples/06_docfold_trainer.py` (Loss: 3.4077 -> 0.0783 in 11.10s, structured symbol generation validated)
- **SCRUM-319 Ticket Stage**: `검토 중 (Ready for Device Validation)`
- **Android Termux Gate**: `PENDING`

---

## 4. Overall Sprint 7 Metrics & Verification
- **Total Host Tests**: **600 passed, 7 warnings in 15.54s (100% PASS, Exit Code 0)**
- **Test Evidence**: `reports/junit_test_report.xml`
- **Sprint 7 Stage**: `Host Complete (All 7 Tickets Hardened & Verified, Ready for Device Gate)`
- **Android Termux Gate**: `PENDING` (Done 전환 대기)
