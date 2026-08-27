# Sprints 0 – 3.5 Integrated Verification Checklist

> **Repository:** `termux-train`  
> **Focus:** Autograd Engine, Layer Primitives, Optimizers, and Memory Boundaries

---

## ✅ Completed Milestones

### Sprint 0: Foundation & Build Infrastructure
- [x] Pure-Python / NumPy NDArray tensor wrapper with autograd graph tracking.
- [x] Node backward topological sorting and gradient accumulation.
- [x] In-tree test harness with numerical gradient checking.

### Sprint 1: Linear, Activations & Loss Functions
- [x] `Linear` layer with weight and bias gradient computation.
- [x] `ReLU`, `GELU`, `Sigmoid`, and `Softmax` activation kernels.
- [x] `CrossEntropyLoss` and `MSELoss` numerically stabilized functions.

### Sprint 2: Optimizers & Learning Rate Scheduling
- [x] `SGD` with momentum and Nesterov acceleration.
- [x] `Adam` and `AdamW` with decoupled weight decay.
- [x] Cosine annealing learning rate scheduler.

### Sprint 3: Layer Normalization & Attention Primitives
- [x] `LayerNorm` and `RMSNorm` for transformer architectures.
- [x] Multi-Head Attention (MHA) self-attention module.
- [x] Rotary Position Embeddings (RoPE) implementation.

### Sprint 3.5: Mobile Stability & Memory Hardening
- [x] Peak RSS monitoring and memory budget enforcement.
- [x] Zero-leak tensor cycle collection in backward tape.
