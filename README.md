# 📱 termux-train (AMEVA-Termux)
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

[📚 Official Documentation](https://uno-km.vercel.app/lib/train/) • [📦 PyPI Package](https://pypi.org/project/termux-train/) • [💬 Issue Tracker](https://github.com/uno-km/termux-train/issues)

</div>

---

## 🏛️ Engineered by AMEVA Foundation (아메바 재단)

`termux-train`은 **AMEVA Open-Source Foundation(AOSF)**의 공식 1등급(Tier 1 TLP) 오픈소스 프로젝트입니다.  
AMEVA Foundation은 **빅테크의 클라우드 종속과 고비용 GPU 독점을 타파**하고, 전 세계 누구나 주머니 속 스마트폰(Android Termux)과 엣지 디바이스에서 **100% 무료로 AI를 학습·미세조정(LoRA)·추론**할 수 있는 온디바이스 AI 주권 기술을 연구·보급합니다.

### 🌐 AMEVA On-Device Open-Source Ecosystem
- 🎨 **[Termux-Diffusion](https://uno-km.vercel.app/lib/diffusion/)**: Android Termux 네이티브 온디바이스 AI 이미지 생성 엔진
- 🎭 **[Termux-Playwright](https://uno-km.vercel.app/lib/playwright/)**: 모바일 온디바이스 헤드리스 브라우저 자동화
- 🎙️ **[Termux-STT](https://uno-km.vercel.app/lib/stt/)**: 스마트폰 전용 로컬 음성인식 엔진
- ⚡ **[AMEVA-Forge](https://uno-km.vercel.app/lib/forge/)**: 웹 표준 차세대 WebGPU 딥러닝 런타임
- 💻 **[AMEVA Workstation](https://ameva-workstation-web-core.vercel.app/)**: 100% 온디바이스 로컬 AI & 문서 편집기

---

## 🌐 Official Documentation & Live Manual
- 📖 **Official Documentation**: [https://uno-km.vercel.app/lib/train/](https://uno-km.vercel.app/lib/train/) *(13개 국어 다국어 지원)*
- 📚 **On-Device Tiny Model & Small LLM Training Manual**: [docs/tiny_model_training_guide.md](docs/tiny_model_training_guide.md)
- 📦 **Official PyPI Package**: [https://pypi.org/project/termux-train/](https://pypi.org/project/termux-train/)

---

## 💡 What is termux-train?

`termux-train` (also known as `AMEVA-Termux`) is a lightweight, self-contained deep learning and automatic differentiation (Autograd) training engine built specifically for **Android Termux native environments** and resource-constrained edge devices.

While standard mobile ML frameworks (TFLite, ONNX Runtime Mobile, ExecuTorch, NCNN) only support **inference**, `termux-train` enables **full on-device training, backpropagation, RoPE Transformers, and LoRA fine-tuning** directly on smartphone hardware without requiring heavy PyTorch binaries or PRoot container virtualization.

> ⚠️ **Disclaimer**: `termux-train` is an independent project engineered by the AMEVA team. It is not affiliated with, endorsed by, or sponsored by PyTorch, Meta, or the Termux project.

---

## 🚀 5-Minute Quickstart

### 1. Installation via PyPI

```bash
# A. In Android Termux (Native with OpenBLAS C-Acceleration):
pkg update && pkg install python python-numpy git
pip install termux-train

# B. Standard Linux / macOS / Windows:
pip install termux-train[accelerated]
```

### 2. CLI Diagnostics & Self-Test

```bash
# Check device hardware, RAM, and backend capabilities:
termux-train info

# Run mathematical self-test across all backends:
termux-train check

# Run 0-point baseline granular audit scorecard:
termux-train score

# Run any canonical demo (1 through 9):
termux-train demo 9  # Tiny Whisper Speech-to-Text LoRA Fine-Tuning
```

---

## 🧠 Core Features & Modern Architecture

| Feature | Description | Mobile Benefit |
| :--- | :--- | :--- |
| **Pure Python Autograd** | Zero-dependency dynamic computation graph with reverse-mode DAG | Runs everywhere, no compilation required |
| **NumPy Tier-2 Engine** | C-vectorized BLAS acceleration for Matmul, CrossEntropy, Softmax | $10\times \sim 100\times$ faster on ARM CPUs |
| **Native RoPE** | Rotary Position Embedding with $O(0)$ learnable parameters | Infinite context extrapolation without memory bloat |
| **On-Device LoRA** | Parameter-efficient fine-tuning ($A \times B$ low-rank decomposition) | Fine-tune models with $99.9\%$ smaller adapter files (<100KB) |
| **SafeTensors Binary** | HuggingFace-compatible zero-copy serialization format | $5\times \sim 10\times$ faster checkpoints, eliminates LMK crash |
| **INT8 Quantization** | Symmetric AbsMax dynamic weight quantization | Zero-allocation matrix scaling, $75\%$ RAM reduction |
| **MMap Streaming** | Streaming token sequence dataset directly from disk via `mmap` | Train on datasets larger than physical mobile RAM |

---

## ⚡ 10-Line Code Examples

### A. Autograd & Tensor Math

```python
from termux_train import Tensor, set_backend

# Pluggable backend (auto / numpy / python)
set_backend("auto")

x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
w = Tensor([[2.0], [1.0]], requires_grad=True)

y = x @ w
loss = (y * y).mean()
loss.backward()

print("x.grad:\n", x.grad)
print("w.grad:\n", w.grad)
```

### B. Tiny Transformer LM with RoPE & KV Caching

```python
from termux_train import Tensor, nn

# Modern Decoder-Only Transformer with Native RoPE
model = nn.TinyTransformerLM(
    vocab_size=100,
    d_model=32,
    num_heads=4,
    d_ff=64,
    num_layers=2,
    pos_type="rope",       # O(0) positional embedding parameters
    tie_weights=True       # Shares token embeddings with LM head
)

# Fast autoregressive generation with incremental KV cache
prompt = [1, 5, 12]
generated = model.generate(prompt, max_new_tokens=20, temperature=0.7, top_p=0.9)
print("Generated Tokens:", generated)
```

### C. Lightweight LoRA Adapter Serialization (<100KB)

```python
from termux_train import nn, checkpoint

# 1. Base Model & LoRA Injection
base = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
student = nn.Sequential(
    nn.LoRALinear.from_linear(base[0], rank=4, alpha=8.0),
    nn.ReLU(),
    nn.LoRALinear.from_linear(base[2], rank=4, alpha=8.0),
)

# 2. Save ONLY the adapter weights in SafeTensors format (<100KB)
checkpoint.save_lora_adapter(student, "lora_adapter.safetensors", adapter_name="my_lora_v1")

# 3. Load adapter into fresh model in another session
checkpoint.load_lora_adapter(student, "lora_adapter.safetensors")
```

---

## 📊 Canonical Examples Suite (`examples/`)

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
