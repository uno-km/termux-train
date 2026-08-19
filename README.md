# 📱 termux-train (AMEVA-Termux)
> **Native On-Device Deep Learning & Autograd Training Framework for Android Termux**
> *Zero PyTorch Dependency · Pure Python Autograd Core · Pluggable NumPy Acceleration · Mobile-Resilient Runtime · On-Device LoRA*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Android%20%7C%20Termux%20%7C%20Linux%20arm64-green.svg)](https://termux.dev)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-orange.svg)](https://www.python.org/)

---

## 💡 What is termux-train?

`termux-train` (also known as `AMEVA-Termux`) is a lightweight, self-contained deep learning and automatic differentiation (Autograd) training engine built specifically for **Android Termux native environments**.

While standard mobile ML frameworks (TFLite, ONNX Runtime Mobile, ExecuTorch, NCNN) only support **inference**, `termux-train` enables **full on-device training, backpropagation, and LoRA fine-tuning** directly on smartphone hardware without requiring heavy PyTorch binaries or PRoot container virtualization.

> ⚠️ **Disclaimer**: `termux-train` is an independent project engineered by the AMEVA team. It is not affiliated with, endorsed by, or sponsored by PyTorch or Meta.

---

## 📦 Package Naming & Import Policy

- **Distribution Name**: `termux-train`
- **Python Import Package**: `termux_train`
- **Legacy Name (`termux_torch`)**: Unsupported and completely removed.

```python
# Official and only supported import package
from termux_train import Tensor, nn
```

---

## 🏛️ Core Architectural Pillars (핵심 개발 철학)

| 계층 (Layer) | 개발 방식 및 주체 | 핵심 특징 |
| :--- | :--- | :--- |
| **1. Tensor API** | **우리가 직접 설계** | PyTorch 호환 0-Dependency 순수 텐서 인터페이스 |
| **2. Autograd Graph** | **우리가 직접 설계** | Dynamic Reverse-Mode DAG 그래프 및 위상정렬 미분 엔진 |
| **3. Backend Engine** | **Python Fallback + NumPy Optional** | 0-Dep 순수 파이썬 기본 탑재 + NumPy C-가속 플러그형 백엔드 |
| **4. MobileTrainer** | **우리가 직접 개발** | 배터리(<20%)/발열(>43°C)/RAM 가드 + 원자적 체크포인트 복구 |
| **5. On-Device LoRA** | **우리가 직접 개발** | 스마트폰 환경 저메모리 Base Weight Freeze + Low-Rank Adapter 파인튜닝 |

---

## ⚡ Core Tensor & Autograd Quickstart

```python
from termux_train import Tensor, set_backend, get_backend

# 1. Pluggable Backend (auto / numpy / python)
set_backend("auto")  # Uses NumPy if available, fallback to Pure-Python
print("Active Backend:", get_backend().name)

# 2. Tensor Creation with Autograd Tracking
x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
w = Tensor([[2.0], [1.0]], requires_grad=True)
y = x @ w  # Shape (2, 1) -> [[4.0], [10.0]]
```

### 📘 [Example A] Linear Mean Loss (`loss = y.mean()`)
*기본 Autograd 수식 및 해석적 그래디언트 설명용*

```python
loss_a = y.mean()  # loss = (4.0 + 10.0) / 2 = 7.0
loss_a.backward()

print("x.grad:\n", x.grad)
# Output: [[1.0, 0.5],
#          [1.0, 0.5]]

print("w.grad:\n", w.grad)
# Output: [[2.0],
#          [3.0]]
```

### 📙 [Example B] Non-linear Squared Loss (`loss = (y * y).mean()`)
*실제 머신러닝 손실 함수(MSE 등) 형태의 비선형 손실 데모용*

```python
# Reset gradients
x.zero_grad()
w.zero_grad()

y = x @ w
loss_b = (y * y).mean()  # loss = (16.0 + 100.0) / 2 = 58.0
loss_b.backward()

print("x.grad:\n", x.grad)
# Output: [[8.0, 4.0],
#          [20.0, 10.0]]

print("w.grad:\n", w.grad)
# Output: [[34.0],
#          [48.0]]
```

---

## ⚡ Matmul Support (1D~3D Complete Matrix)

`termux_train` supports all matrix multiplication rank combinations where both operands are between 1D and 3D:

- **`1D @ 1D`**: `(K,) @ (K,) -> ()` (0D Scalar Dot Product)
- **`1D @ 2D`**: `(K,) @ (K, N) -> (N,)` (1D Vector)
- **`1D @ 3D`**: `(K,) @ (B, K, N) -> (B, N)` (2D Batch Matrix)
- **`2D @ 1D`**: `(M, K) @ (K,) -> (M,)` (1D Vector)
- **`2D @ 2D`**: `(M, K) @ (K, N) -> (M, N)` (2D Matrix Multiplication)
- **`2D @ 3D`**: `(M, K) @ (B, K, N) -> (B, M, N)` (3D Batched Matrix)
- **`3D @ 1D`**: `(B, M, K) @ (K,) -> (B, M)` (2D Matrix)
- **`3D @ 2D`**: `(B, M, K) @ (K, N) -> (B, M, N)` (3D Sequence / LoRA Projection)
- **`3D @ 3D`**: `(B, M, K) @ (B, K, N) -> (B, M, N)` (3D Transformer Attention Product)

*Note: Scalar operands (0D) and 4D+ general ND matmuls are not supported yet.*

---

## 🧠 Neural Network Training Quickstart

```python
from termux_train import Tensor, nn, optim

# 1. Define Model
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.Tanh(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

# 2. Setup Optimizer & Data
optimizer = optim.Adam(model.parameters(), lr=0.05)
criterion = nn.MSELoss()

x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
target = Tensor([[0.0], [1.0], [1.0], [0.0]])

# 3. Mobile Training Loop (XOR Convergence)
for epoch in range(1000):
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | MSE Loss: {loss.item():.6f}")
```

---

## ⚡ Optimizers

`termux_train.optim` provides first-order optimizers with full state serialization and pure Python / NumPy backend parity:

- **`optim.SGD`**: Stochastic Gradient Descent with L2 weight decay, momentum factor $\mu$, dampening, and optional Nesterov accelerated gradient.
- **`optim.Adam`**: Adaptive Moment Estimation with coupled L2 weight decay and numerical bias correction.
- **`optim.AdamW`**: Decoupled Weight Decay Adam for modern Transformer & neural network training.

---

## 📱 Mobile-Resilient Training with MobileTrainer *(Planned: Sprint 5)*

```python
# [Planned Sprint 5 Preview]
from termux_train.runtime import MobileTrainer

trainer = MobileTrainer(
    model=model,
    optimizer=optimizer,
    checkpoint_dir="./checkpoints",
    checkpoint_every=100,
    pause_if_battery_below=20,      # Auto-pause when battery is low (<20%)
    pause_if_temperature_above=43,  # Auto-throttle when phone heats up (>43°C)
    max_ram_mb=2048
)

trainer.fit(dataset, epochs=10)
```

---

## 🏗️ Architecture

```
termux-train/
├── termux_train/
│   ├── __init__.py           # Tensor, nn, optim, utils exports
│   ├── tensor.py             # Pure-Python Tensor Data Model & DAG Graph
│   ├── backend/              # Pluggable Compute Backends (Base, Python, NumPy)
│   ├── nn/                   # Module, Parameter, Linear, Sequential, Activations, Losses
│   ├── optim/                # First-Order Optimizers: SGD (Momentum, Nesterov), Adam, AdamW
│   ├── runtime/              # [Sprint 5 Planned] MobileTrainer, Safe Checkpoint, Battery/Thermal Guard
│   └── utils/                # Termux Environment Probe, Numerical Gradcheck
├── scripts/                  # Device Setup & Diagnostics Scripts, Code Exporter
├── examples/                 # Basics, NN Forward/Backward, 1D~3D Matmul, XOR Training Demos
└── tests/                    # 216 Unit, Backend, Autograd, NN, Optim, Training, Gradcheck Test Suites
```

---

## 🗺️ Sprint Roadmap (Scrum Tracked)

- [x] **Sprint 0**: Governance, Environment Setup & Termux Diagnostics (`SCRUM-262` ~ `SCRUM-267`)
- [x] **Sprint 1**: Pluggable Backend & Tensor Core (`SCRUM-268` ~ `SCRUM-274`)
- [x] **Sprint 2**: Dynamic DAG Autograd Engine & Gradcheck (`SCRUM-275` ~ `SCRUM-286`)
- [x] **Sprint 3**: NN Mini Framework & Linear Layers (`SCRUM-287` ~ `SCRUM-295`)
- [x] **Sprint 3.5**: Stabilization & Autograd Policy Hardening (`SCRUM-332` ~ `SCRUM-335`)
- [x] **Sprint 3.6**: Core Semantics Hardening (Reduction axes, ND Transpose) (`SCRUM-336` ~ `SCRUM-338`)
- [x] **Sprint 3.7**: Cross-Backend Auto-Conversion & Matmul Contract Validation (`SCRUM-339` ~ `SCRUM-342`)
- [x] **Sprint 3.8**: Initial 1D~3D Matmul Core Suite & Autograd Hardening (`SCRUM-343` ~ `SCRUM-350`)
- [x] **Sprint 3.9**: Complete 1D~3D Matmul Rank Matrix, 9 Forward/Backward Combinations & Linear 1D~3D Support (`SCRUM-351`)
- [x] **Sprint 4**: Optimizers (`SGD`, `Adam`, `AdamW`) & XOR Convergence MVP (`SCRUM-296` ~ `SCRUM-300`)
- [ ] **Sprint 5**: Mobile Training Runtime & Thermal/Battery Guard (`SCRUM-301` ~ `SCRUM-307`)
- [ ] **Sprint 6**: On-Device LoRA Adapter (`SCRUM-308` ~ `SCRUM-312`)
- [ ] **Sprint 7**: Tiny Transformer & CharLM Toy Trainer (General 4D ND Matmul & Multi-Head Attention) (`SCRUM-313` ~ `SCRUM-319`)
- [ ] **Sprint 8**: Packaging, Full Test Suite & v0.1.0-alpha Release (`SCRUM-320` ~ `SCRUM-325`)
- [ ] **Sprint 9+**: ARM NEON & OpenCL Hardware Acceleration (`SCRUM-326` ~ `SCRUM-331`)

---

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE) for details.
