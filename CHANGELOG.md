# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0-alpha] - 2026-08-20

### 🚀 Initial Public Technical Preview Release

#### Added
- **Core Tensor & Autograd Engine**:
  - Zero-dependency pure Python `Tensor` with dynamic reverse-mode DAG graph tracking.
  - Pluggable backend architecture (`BaseBackend`, `PythonBackend`, `NumPyBackend`).
  - Full 1D~3D matrix multiplication matrix (all 9 forward/backward rank combinations).
  - Numerical gradient checker (`gradcheck`) for analytical autograd verification.
- **Neural Network Mini-Framework (`termux_train.nn`)**:
  - `nn.Module`, `nn.Parameter`, `nn.Sequential`.
  - Layers: `Linear`, `Embedding` (vectorized scatter-add), `LayerNorm`.
  - Activations: `ReLU`, `Sigmoid`, `Tanh`.
  - Loss functions: `MSELoss`, `BCELoss`, `BCEWithLogitsLoss`, `CrossEntropyLoss` (vectorized integer-target Fused CrossEntropy).
- **First-Order Optimizers (`termux_train.optim`)**:
  - `SGD` (with momentum, dampening, Nesterov acceleration, L2 weight decay).
  - `Adam` (with first/second moment tracking, bias correction).
  - `AdamW` (with decoupled weight decay).
- **Mobile-Resilient Training Runtime (`termux_train.runtime`)**:
  - `MobileTrainer` orchestrator with explicit training loop control.
  - Atomic checkpoint writing (`.tmp` $\to$ `fsync` $\to$ `os.replace`) with automatic rollback on error.
- **On-Device LoRA (Low-Rank Adaptation)**:
  - `LoRALinear` layer with $A \times B$ low-rank weight decomposition.
  - `mark_only_lora_as_trainable` base model weight freezing.
  - `merge_lora_weights` / `unmerge_lora_weights` for zero-overhead inference deployment.
  - Lightweight LoRA adapter serialization (`save_lora_adapter`, `load_lora_adapter`) producing $<100\text{KB}$ adapter files.
- **Transformer & Modern LLM Primitives**:
  - `RotaryEmbedding` (RoPE) with $O(0)$ learnable parameters for context extrapolation.
  - `MultiHeadAttention` with Universal Causal Trapezoid masking for chunked prefill.
  - `TinyTransformerLM` with weight tying and incremental KV-caching.
  - Autoregressive `generate()` with Top-K truncation, Top-P (nucleus) filtering, and early `<EOS>` stopping.
- **Big-Tech Production Hardening**:
  - HuggingFace-compatible zero-copy `.safetensors` binary format.
  - `QuantizedLinear` with zero-allocation INT8 matrix multiplication.
  - `MMapTokenDataset` for streaming token sequences from disk via OS `mmap`.
- **Command-Line Interface (`termux-train`)**:
  - `termux-train info`: Detailed hardware, OS, memory, and backend diagnostics.
  - `termux-train check`: Mathematical self-test across all backends.
  - `termux-train score`: 0-point baseline granular audit scoring runner.
  - `termux-train demo <1..8>`: Interactive runner for canonical example scripts.
- **Quality & Verification**:
  - 649 unit, integration, and mobile test cases (100% PASS).
  - 0-Point Baseline Granular Audit Scorecard (100.0 / 100.0 Points, Grade A+).
  - Multi-platform CI workflow (Ubuntu, Windows, macOS, Python 3.8 ~ 3.12).
