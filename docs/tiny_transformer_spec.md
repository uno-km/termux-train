# Tiny Transformer & CharLM Mathematical Specification

## 1. Scope and Design Philosophy
This document specifies the exact numerical, mathematical, and tensor contracts for the **Tiny Transformer & CharLM Engine** in `termux-train`.
It is designed specifically for constrained mobile runtime environments (Android Termux) with:
- Zero binary / C++ / Rust dependencies (Pure Python Tier-1, Optional NumPy Tier-2).
- Strict numerical stability against floating-point overflow and underflow.
- Exact autograd reverse-mode gradient derivation for all operators.

---

## 2. Tensor Data Model & Dtype Hierarchy

### 2.1 Supported Dtypes
- `"float32"`: Default 32-bit floating point for continuous tensors and neural network weights.
- `"int64"`: 64-bit integer for token IDs, embedding lookup indices, and classification targets.
- `"bool"`: Boolean tensor for masks and condition predicates.

### 2.2 Dtype Autograd Invariants
- `requires_grad=True` is **only** permitted on floating-point tensors (`"float32"`).
- Attempting to set `requires_grad=True` on `"int64"` or `"bool"` tensors raises `ValueError`.

---

## 3. Core Math & Tensor Operators

### 3.1 Generalized N-D Batched Matrix Multiplication
For tensors $A$ of shape $(\dots, M, K)$ and $B$ of shape $(\dots, K, N)$:
- Leading batch dimensions $(\dots)$ are broadcasted according to standard right-aligned broadcasting rules.
- 1D operands are promoted:
  - $1\text{D} @ 2\text{D}$: $(K,) \to (1, K) @ (K, N) \to (1, N) \to (N,)$
  - $2\text{D} @ 1\text{D}$: $(M, K) @ (K,) \to (M, K) @ (K, 1) \to (M, 1) \to (M,)$
  - $1\text{D} @ 1\text{D}$: $(K,) @ (K,) \to \text{scalar}$
- **Autograd Reverse Derivative**:
  $$dA_{\text{prom}} = G_{\text{prom}} \times B_{\text{prom}}^T, \quad dB_{\text{prom}} = A_{\text{prom}}^T \times G_{\text{prom}}$$
  where $G_{\text{prom}}$ is the upstream gradient. The gradients are unbroadcasted (summed along broadcasted batch axes) to recover the exact shapes of $A$ and $B$.

### 3.2 Maximum with Subgradient
- Forward: $y = \max(x, \text{axis}, \text{keepdims})$
- Backward: Upstream gradient is equidistributed across tied maximum positions (conserving total subgradient mass):
  $$\frac{\partial y}{\partial x_i} = \frac{g \cdot \mathbf{1}(x_i = \max(x))}{\sum_j \mathbf{1}(x_j = \max(x))}$$
  where $\mathbf{1}(\cdot)$ is the indicator function.

### 3.3 Stable Log-Sum-Exp & Softmax
- Max-shifted Log-Sum-Exp:
  $$m = \max(x, \text{axis}, \text{keepdims}=\text{True}).\text{detach}()$$
  $$\text{logsumexp}(x) = m + \log\left(\sum \exp(x - m)\right)$$
- Log-Softmax:
  $$\text{log\_softmax}(x) = x - \text{logsumexp}(x)$$
- Softmax:
  $$\text{softmax}(x) = \exp(\text{log\_softmax}(x)) = \frac{\exp(x - m)}{\sum \exp(x - m)}$$

---

## 4. Multi-Head Attention Specification

For input $X \in \mathbb{R}^{B \times S \times D}$ with $H$ heads and head dimension $d_k = D / H$:

### 4.1 Projections & 4D Head Splitting
$$Q = X W_q, \quad K = X W_k, \quad V = X W_v \quad (W_q, W_k, W_v \in \mathbb{R}^{D \times D})$$
$$Q_{\text{4D}} = Q.\text{reshape}(B, S, H, d_k).\text{transpose}(0, 2, 1, 3) \in \mathbb{R}^{B \times H \times S \times d_k}$$
$$K_{\text{4D}} = K.\text{reshape}(B, S, H, d_k).\text{transpose}(0, 2, 1, 3) \in \mathbb{R}^{B \times H \times S \times d_k}$$
$$V_{\text{4D}} = V.\text{reshape}(B, S, H, d_k).\text{transpose}(0, 2, 1, 3) \in \mathbb{R}^{B \times H \times S \times d_k}$$

### 4.2 Scaled Dot-Product & Causal Mask
$$\text{Scores} = \frac{Q_{\text{4D}} \times K_{\text{4D}}^T}{\sqrt{d_k}} \in \mathbb{R}^{B \times H \times S \times S}$$
$$\text{CausalMask}_{i, j} = \begin{cases} 0.0 & \text{if } j \le i \\ -10^4 & \text{if } j > i \end{cases}$$
$$\text{AttentionWeights} = \text{Softmax}(\text{Scores} + \text{CausalMask}, \text{dim}=-1)$$
$$\text{Context}_{\text{4D}} = \text{AttentionWeights} \times V_{\text{4D}} \in \mathbb{R}^{B \times H \times S \times d_k}$$

### 4.3 Head Merging & Output Projection
$$\text{Context} = \text{Context}_{\text{4D}}.\text{transpose}(0, 2, 1, 3).\text{reshape}(B, S, D)$$
$$\text{Output} = \text{Context} \times W_o \quad (W_o \in \mathbb{R}^{D \times D})$$

---

## 5. Layer Normalization Specification

For input $x \in \mathbb{R}^{\dots \times D}$ with parameters $\gamma, \beta \in \mathbb{R}^D$:
$$\mu = \frac{1}{D}\sum_{i=1}^D x_i, \quad \sigma^2 = \frac{1}{D}\sum_{i=1}^D (x_i - \mu)^2$$
$$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$
where $\epsilon = 10^{-5}$ is added strictly **inside** the square root.

---

## 6. Pre-LN Transformer Block Architecture

```
         Input x
            │
      ┌─────┴─────┐
      │           ▼
      │        LayerNorm 1
      │           ▼
      │      MultiHeadAttention
      │           ▼
      └───► (+) ◄─┘
            │ (Residual 1)
      ┌─────┴─────┐
      │           ▼
      │        LayerNorm 2
      │           ▼
      │      FeedForward (Linear -> Tanh -> Linear)
      │           ▼
      └───► (+) ◄─┘
            │ (Residual 2)
            ▼
         Output y
```

---

## 7. Cross-Entropy Loss Specification

For logits $z \in \mathbb{R}^{N \times C}$ and integer target indices $y \in \{0, \dots, C-1\}^N$:
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \text{log\_softmax}(z)_{i, y_i} = \frac{1}{N}\sum_{i=1}^N \left( -z_{i, y_i} + \text{logsumexp}(z_i) \right)$$
- Gradient:
  $$\frac{\partial \mathcal{L}}{\partial z_{i, c}} = \frac{1}{N}\left( \text{Softmax}(z)_{i, c} - \mathbf{1}(c = y_i) \right)$$
