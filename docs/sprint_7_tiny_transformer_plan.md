# Sprint 7 Implementation Plan: Tiny Transformer & CharLM Toy Trainer

## Baseline Confirmation
- **Base Commit**: `2467f5b35172cfa913f5d0051fe117aec338e0cd` (`Docs: clarify LoRA checkpoint base model reconstruction boundary in README`)
- **Host Test Baseline**: `461 passed, 1 warning in 6.44s`
- **Working Tree**: `CLEAN`
- **Branch**: `main`
- **Sprint 6 Status**: `Host Complete, Android Device Validation Pending`
- **Current Stage**: `Sprint 7 Host In Progress (Tiny Transformer & CharLM, Device Validation Pending)`

---

## 1. Architectural Foundation Decisions (ADR)

### 1.1 ADR 7.1: Integer / Index Representation for Embedding & Losses
- **Problem**: In current `PythonBackend` and `NumPyBackend`, all Tensor data is cast to `float` / `np.float32`. Therefore, `Tensor([1, 2])` and `Tensor([1.0, 2.0])` are indistinguishable post-construction, preventing strict runtime validation of integer-only indices in `nn.Embedding` and class targets in `nn.CrossEntropyLoss`.
- **Decision (Two-Tier Index Contract)**:
  1. **Tier 1 (Sprint 7 Phase 1~3)**: Public `nn.Embedding` and `nn.CrossEntropyLoss` accept either raw integer indices (`List[int]`, `List[List[int]]`) or integer-validated index buffers (`int` type check on all elements).
  2. **Tier 2 (Foundation Extension)**: Add explicit `dtype` tracking to `Tensor` (`dtype="float32"` by default, `dtype="int64"` for integer tensors). Integer tensors strictly enforce `requires_grad=False` and reject non-integer / floating-point operations.

### 1.2 ADR 7.2: Transpose API Semantics
- **Problem**: `Tensor.transpose(*axes)` currently expects a full permutation tuple of length equal to `ndim` (e.g. `x.transpose(0, 2, 1, 3)` for 4D attention shape transformation). Writing `x.transpose(1, 2)` on a 4D tensor is an invalid permutation in the existing API.
- **Decision**:
  - Preserve the existing `Tensor.transpose(*axes)` full permutation contract without breaking backward compatibility.
  - Introduce `Tensor.swapaxes(dim0, dim1)` as an explicit 2-axis swap helper.
  - In Attention projection, explicitly use `x.transpose(0, 2, 1, 3)` and inverse `x.transpose(0, 2, 1, 3)`.

### 1.3 ADR 7.3: Generalized N-D Batched Matmul Specification
- **Forward Contract**:
  - Operands $A$ of shape $(\dots, M, K)$ and $B$ of shape $(\dots, K, N)$ broadcast leading batch dimensions $(\dots)$ using standard NumPy-compatible right-aligned broadcasting.
  - 1D promotion: 1D @ 2D -> 1D vector; 2D @ 1D -> 1D vector; 1D @ 1D -> scalar.
- **Backward Contract**:
  - Upstream gradient $G$ of shape $(\text{broadcast\_batch}, M, N)$ computes:
    $$dA_{\text{promoted}} = G \times B^T, \quad dB_{\text{promoted}} = A^T \times G$$
  - Broadcast axes are unbroadcast (summed along broadcasted dimensions) back to the exact shape of $A$ and $B$.

### 1.4 ADR 7.4: Softmax & Causal Masking Numerical Contracts
- **Softmax Stability**: Uses max-subtraction stabilization:
  $$m = x.\text{max}(\text{axis}=-1, \text{keepdims}=\text{True}), \quad \text{Softmax}(x) = \frac{\exp(x - m)}{\sum \exp(x - m)}$$
- **Max Reduction**: Implemented as detached constant for stabilization without tie-gradient explosion.
- **Causal Mask**: Additive finite mask with sentinel value $-10^9$ (or $-10^4$ for float32 precision) to prevent `NaN` during float subtract while ensuring future token attention weights are zero within float32 epsilon.

---

## 2. Reordered Sprint 7 Execution Phases

```
Foundation & Phase 1: SCRUM-313 - Lightweight Tokenizers (Base, Char, Byte, Word)
  ↓
Foundation & Phase 2: SCRUM-315 - Tiny Transformer Math Spec & Core Tensor Primitives (4D Matmul, exp, max, softmax, mask)
  ↓
Phase 3: SCRUM-314 - nn.Embedding Layer (Forward lookup, Scatter-Add backward)
  ↓
Phase 4: SCRUM-316 - nn.LayerNorm, nn.MultiHeadAttention, nn.TransformerBlock (Pre-LN, Residual, FFN)
  ↓
Phase 5: SCRUM-317 - nn.CrossEntropyLoss, CharLM Model, Autoregressive Training & Generation Demo
  ↓
Phase 6: SCRUM-318 - DocFold JSONL Dataset Pipeline
  ↓
Phase 7: SCRUM-319 - DocFold Sequence Mapping Toy Trainer & Convergence Gate
```

---

## 3. Phase Details & Commit Strategy

### Phase 1: SCRUM-313 - Lightweight Tokenizer Interface (Host Complete)
- **Status**: Host Complete
- **Commit**: `f3e673d` (`Add deterministic lightweight tokenizers`)
- **Host Tests**: `483 passed, 1 warning` (22 tokenization tests PASS)
- **Files**:
  - `termux_train/tokenization/__init__.py`
  - `termux_train/tokenization/base.py`
  - `termux_train/tokenization/char.py`
  - `termux_train/tokenization/byte.py`
  - `termux_train/tokenization/word.py`
  - `tests/test_tokenization.py`
- **Scope**:
  - `BaseTokenizer` abstract base class with deterministic vocabulary management and special token constants (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`).
  - `CharTokenizer`: Character-level tokenizer with exact round-trip on known-vocabulary characters, unknown fallback to `<UNK>`, BOS/EOS options.
  - `ByteTokenizer`: UTF-8 byte-level tokenizer with complete 0~255 byte vocabulary, exact round-trip for valid UTF-8 strings.
  - `WordTokenizer`: Whitespace and punctuation preserving tokenizer with exact round-trip on known words without losing layout.
  - Pure Python, zero Tensor / NumPy dependencies.

### Phase 2: SCRUM-315 - Tiny Transformer Math Spec & Tensor Operators
- **Files**:
  - `docs/tiny_transformer_spec.md`
  - `termux_train/tensor.py`
  - `termux_train/backend/python_backend.py`
  - `termux_train/backend/numpy_backend.py`
  - `tests/test_tensor.py`, `tests/test_autograd.py`
- **Scope**: Generalized N-D Batched Matmul, `exp`, `max` reduction, `swapaxes`, `softmax`, causal mask utilities.
- **Commit Message**: `Add generalized ND matmul and transformer tensor primitives`

### Phase 3: SCRUM-314 - Embedding Layer (`nn.Embedding`)
- **Files**:
  - `termux_train/nn/embedding.py`
  - `termux_train/nn/__init__.py`
  - `tests/test_nn.py`
- **Scope**: Forward lookup, scatter-add backward gradient accumulation, shape and index validation.
- **Commit Message**: `Add Embedding layer with scatter-add autograd`

### Phase 4: SCRUM-316 - Tiny Transformer Block (`nn.LayerNorm`, `nn.MultiHeadAttention`, `nn.TransformerBlock`)
- **Files**:
  - `termux_train/nn/normalization.py` (`LayerNorm`)
  - `termux_train/nn/attention.py` (`MultiHeadAttention`)
  - `termux_train/nn/transformer.py` (`TransformerBlock`)
  - `termux_train/nn/__init__.py`
  - `tests/test_nn.py`
- **Commit Message**: `Add LayerNorm, MultiHeadAttention, and TransformerBlock`

### Phase 5: SCRUM-317 - CharLM Autoregressive Language Model Demo
- **Files**:
  - `termux_train/nn/loss.py` (`CrossEntropyLoss`)
  - `termux_train/nn/models/charlm.py`
  - `examples/07_charlm_autoregressive_training.py`
  - `tests/test_training.py`
- **Commit Message**: `Add CharLM autoregressive model and training demo`

### Phase 6: SCRUM-318 - DocFold Toy Dataset Pipeline
- **Files**:
  - `termux_train/data/docfold.py`
  - `tests/test_data.py`
- **Commit Message**: `Add DocFold dataset pipeline and grammar tokenizer`

### Phase 7: SCRUM-319 - DocFold Sequence Mapping Toy Trainer
- **Files**:
  - `examples/08_docfold_sequence_mapping.py`
  - `reports/docfold_training_report.md`
- **Commit Message**: `Add DocFold sequence mapping toy trainer`

---

## 4. Resource Bounds & Metric Definitions

1. **Analytical Lower-Bound Estimate**:
   - Parameter scalar count: $\approx 25\text{K} \sim 50\text{K}$ parameters.
   - Nominal float32 weights: $\approx 100\text{KB} \sim 200\text{KB}$.
2. **Measured Host Metrics**:
   - Wall-clock step time on host CPU.
   - Peak Python traced memory via `tracemalloc`.
3. **Measured Android Metrics (Device Gate Pending)**:
   - Peak resident memory (RSS).
   - Per-epoch training latency on ARM architecture.
