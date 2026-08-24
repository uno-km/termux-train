# termux-train (AMEVA-Termux)

> **Native On-Device Deep Learning & LoRA Training Framework for Android Termux**  
> *Zero PyTorch Dependency · Pure Python Autograd Core · Pluggable NumPy Acceleration · Mobile-Resilient Runtime · On-Device LoRA · SafeTensors · RoPE Transformer*

<div align="center">

[![Official Documentation](https://img.shields.io/badge/docs-uno--km.vercel.app%2Flib%2Ftrain-004499?style=for-the-badge&logo=vercel)](https://uno-km.vercel.app/lib/train/)
[![PyPI - Version](https://img.shields.io/pypi/v/termux-train.svg?color=0066cc&logo=pypi&logoColor=white&style=for-the-badge)](https://pypi.org/project/termux-train/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/termux-train.svg?logo=python&logoColor=white&style=for-the-badge)](https://pypi.org/project/termux-train/)
[![Audit Scorecard](https://img.shields.io/badge/Audit%20Scorecard-100.0%20%2F%20100.0%20(Grade%20A%2B)-brightgreen.svg?style=for-the-badge)](https://uno-km.vercel.app/lib/train/benchmarks.html)
[![Open Collective](https://img.shields.io/badge/Open_Collective-AOSF_Fund-004499?style=flat&logo=opencollective)](https://opencollective.com/ameva-fund)
[![GitHub Sponsors](https://img.shields.io/badge/GitHub_Sponsors-uno--km-ea4aaa?style=flat&logo=githubsponsors)](https://github.com/sponsors/uno-km)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)](LICENSE)
[![AMEVA Foundation](https://img.shields.io/badge/Foundation-AOSF_Tier_1-orange?style=for-the-badge)](https://uno-km.vercel.app/docs/foundation/)

### Ultra-lightweight On-Device Tensor & DAG Autograd Deep Learning Framework
**An Official Tier 1 Top-Level Open-Source Project of the AMEVA Foundation (AOSF)**

[Official Documentation](https://uno-km.vercel.app/lib/train/) • [PyPI Package](https://pypi.org/project/termux-train/) • [Issue Tracker](https://github.com/uno-km/termux-train/issues)

</div>

---

## Engineered by AMEVA Open-Source Foundation (AOSF)

`termux-train` is an official Tier 1 Top-Level Project (TLP) of the **AMEVA Open-Source Foundation (AOSF)**.  
The AMEVA Foundation democratizes AI research by eliminating cloud egress lock-in and high-cost GPU monopolies, empowering developers worldwide to **train, fine-tune (LoRA), and evaluate neural networks directly on everyday smartphones (Android Termux) and ARM64 edge hardware 100% locally**.

### AMEVA On-Device Open-Source Ecosystem
- **[Termux-BitNet](https://github.com/uno-km/termux-bitnet)**: Ultra-low power 1.58-bit LLM on-device inference engine with ARM64 DotProd SIMD.
- **[Termux-Diffusion](https://uno-km.vercel.app/lib/diffusion/)**: Android Termux native on-device AI image generation engine.
- **[Termux-Playwright](https://uno-km.vercel.app/lib/playwright/)**: Non-root mobile headless Chromium browser automation.
- **[Termux-STT](https://uno-km.vercel.app/lib/stt/)**: Integrated on-device speech-to-text and 128d speaker diarization.
- **[AMEVA-Forge](https://uno-km.vercel.app/lib/forge/)**: Browser-native WebGPU deep learning tensor engine.
- **[AMEVA Workstation](https://ameva-workstation-web-core.vercel.app/)**: 100% client-side local AI and document intelligence.

---

## What is termux-train?

`termux-train` (also known as `AMEVA-Termux`) is a lightweight, self-contained deep learning and automatic differentiation (Autograd) training engine built specifically for **Android Termux native environments** and resource-constrained edge devices.

While standard mobile ML frameworks (TFLite, ONNX Runtime Mobile, ExecuTorch, NCNN) only support **inference**, `termux-train` enables **full on-device training, backpropagation, RoPE Transformers, and LoRA fine-tuning** directly on smartphone hardware without requiring heavy PyTorch binaries or PRoot container virtualization.

> **Disclaimer**: `termux-train` is an independent project engineered by the AMEVA team. It is not affiliated with, endorsed by, or sponsored by PyTorch, Meta, or the Termux project.

---

## 5-Minute Quickstart

### 1. Installation via PyPI

```bash
# In Android Termux:
pkg update && pkg install python python-numpy git
pip install termux-train
```

### 2. End-to-End On-Device Training (Python SDK)

```python
import termux_train as tt
import termux_train.nn as nn
import termux_train.optim as optim

# 1. Define Model Architecture
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = MLP()
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# 2. Synthetic Data
x_train = tt.randn(32, 4)
y_train = tt.randn(32, 1)

# 3. Training Loop with Autograd
for epoch in range(10):
    optimizer.zero_grad()
    predictions = model(x_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.6f}")
```

---

## License

<<<<<<< Updated upstream
| File | Demo Name | Description |
| :--- | :--- | :--- |
| `01_tensor_basics.py` | **Tensor Basics** | Tensor creation, shapes, dtypes, elementwise math |
| `02_nn_forward_backward.py` | **NN Forward/Backward** | Custom `nn.Module`, Linear layers, manual backprop |
| `03_matmul_1d_to_3d.py` | **1D~3D Matmul Matrix** | All 9 matrix multiplication rank combinations |
| `04_xor_training.py` | **XOR Convergence** | Non-linear MLP training with Adam/AdamW |
| `05_mobile_training_runtime.py` | **MobileTrainer & Checkpoint** | Atomic crash recovery, state_dict deep copy |
| `06_lora_adapter_training.py` | **LoRA Adapter Fine-Tuning** | Parameter freezing, low-rank adaptation, weight merge |
| `07_transformer_lm.py` | **Character-Level Transformer LM** | RoPE Transformer trained on Shakespeare corpus |
| `08_docfold_trainer.py` | **DocFold Sequence Mapping** | Structured document entity extraction & token generation |
| `09_tiny_whisper_lora.py` | **Tiny Whisper LoRA Fine-Tuning** | Audio 80-mel feature Speech-to-Text LoRA training (<30KB) |

---

## 🛡️ 0-Point Baseline Granular Audit Scorecard

`termux-train` is rigorously verified by a **0-Point Baseline Granular Scoring Protocol** that evaluates 5 core pillars of production integrity:

```text
======================================================================
  🛡️  [termux-train] Production Granular Audit Scoring System
  ⭐ Baseline: 0.0 Points | Target: 100.0 Points
======================================================================
  🏆 AUDIT SCORECARD: 100.0 / 100.0 POINTS (PERFECT GRADE A+)
  ⏱️ Total Audit Execution Time: 18.43s
  ✅ Pillar 1 (Autograd & Math Stability)     : 20.0 / 20.0 pts
  ✅ Pillar 2 (Transformer & RoPE)           : 20.0 / 20.0 pts
  ✅ Pillar 3 (Memory & Allocation Safety)   : 20.0 / 20.0 pts
  ✅ Pillar 4 (Performance & Latency)        : 20.0 / 20.0 pts
  ✅ Pillar 5 (Crash Resilience & Checkpoints): 20.0 / 20.0 pts
======================================================================
```

---

## 🏗️ Architecture Layout

```text
termux-train/
├── termux_train/
│   ├── __init__.py           # Top-level exports (Tensor, nn, optim, runtime, checkpoint, data)
│   ├── tensor.py             # Pure-Python Tensor Data Model & Dynamic Autograd DAG
│   ├── cli.py                # Official Command-Line Interface (info, check, score, demo)
│   ├── backend/              # Pluggable Compute Backends (BaseBackend, PythonBackend, NumPyBackend)
│   ├── nn/                   # Linear, LoRALinear, Embedding, LayerNorm, Attention, Transformer, RoPE
│   ├── optim/                # First-Order Optimizers: SGD (Momentum/Nesterov), Adam, AdamW
│   ├── checkpoint/           # SafeTensors zero-copy binary, LoRA adapter I/O, atomic JSON
│   ├── data/                 # MMapTokenDataset (zero-copy memory mapped disk streaming)
│   ├── runtime/              # MobileTrainer orchestrator & atomic rollback recovery
│   ├── tokenization/         # Pure-Python Lightweight Tokenizers (Base, Char, Byte, Word)
│   └── utils/                # Termux Environment Diagnostics, Numerical Gradcheck
├── .github/workflows/        # Multi-Platform CI & GitHub Pages Deployment
├── docs/                     # Official Documentation Website
├── examples/                 # 9 Canonical Demos (01_tensor_basics.py ~ 09_tiny_whisper_lora.py)
├── scripts/                  # 0-point audit scoring runner, source code exporter
└── tests/                    # 649 unit, integration, mobile, and scorecard test suites
```

---

## ⚖️ Disclaimer (면책 조항)

> **Disclaimer:**  
> *termux-train (AMEVA-Termux) is an independent open-source project developed for the Android Termux environment and is not officially affiliated with, endorsed by, or sponsored by the Termux project, PyTorch, or Meta.*  
> 
> *(본 프로젝트는 안드로이드 Termux 환경을 위해 개발된 독립적인 오픈소스 라이브러리이며, Termux 공식 프로젝트 및 PyTorch, Meta와 직접적인 제휴 관계가 아닙니다.)*

---

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE) for details.


---

## 💖 Sponsorship & Community Backing

AMEVA is an independent open-source public good governed under the **AMEVA Open-Source Foundation (AOSF)**. All sponsorship funds are 100% publicly audited and dedicated to physical ARM64 testbeds and CI/CD GPU runners.

- **Open Collective (Non-Profit 501(c)(6))**: [https://opencollective.com/ameva-fund](https://opencollective.com/ameva-fund)
- **GitHub Sponsors**: [https://github.com/sponsors/uno-km](https://github.com/sponsors/uno-km)
- **Official Foundation Portal**: [https://uno-km.vercel.app/docs/foundation/sponsorship.html](https://uno-km.vercel.app/docs/foundation/sponsorship.html)
=======
Apache License 2.0. Copyright (c) 2026 uno-km (AMEVA Foundation).
>>>>>>> Stashed changes
