# termux-train (v1.1.0)

[![PyPI Version](https://img.shields.io/pypi/v/termux-train.svg?color=blue&style=flat-square)](https://pypi.org/project/termux-train/)
[![NPM Version](https://img.shields.io/npm/v/termux-train.svg?color=red&style=flat-square)](https://www.npmjs.com/package/termux-train)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/termux-train)
[![Python Support](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg?style=flat-square)](https://pypi.org/project/termux-train/)
[![Node Support](https://img.shields.io/badge/Node.js-16%2B-brightgreen.svg?style=flat-square)](https://www.npmjs.com/package/termux-train)

> **Production-Grade Native On-Device Deep Learning & LoRA Training Framework for Android Termux & ARM64.**  
> **Dual-Engine Architecture: Native Python Autograd DAG + Node.js / TypeScript SDK.**

---

## 📌 Key Architectural Highlights

- ⚡ **Production Dual-Engine**: Seamless CLI & SDK parity across Python (`termux_train`) and Node.js/TypeScript (`termux-train`).
- 🛡️ **HuggingFace SafeTensors Hardening**: 100MB Header Bomb defense and zero-copy binary checkpointing with optimizer momentum.
- 💾 **2M-Sample Bounded MMap Dataset**: High-throughput binary `.bin` token stream loader operating within constant <50MB RAM.
- 🧬 **LoRA & RoPE Attention**: Rank-decomposition parameter-efficient fine-tuning with transactional snapshot rollback & Rotary Position Embeddings.
- 🌐 **Multilingual ByteTokenizer**: Native UTF-8 tokenizer with full Korean Hangul, CJK ideographs, Emoji, and UTF-8 BOM support.
- 🚀 **One-Touch Universal Installer**: `install.sh` for one-click setup across Android Termux, Linux, and macOS.

---

## 🚀 Installation & Quickstart

### 1. Universal One-Touch Installation (Recommended)
```bash
curl -fsSL https://raw.githubusercontent.com/uno-km/termux-train/main/install.sh | bash
```

### 2. Python Package (PyPI)
```bash
# Standard pure Python + NumPy Autograd engine
pip install termux-train

# With Vulkan GPU acceleration
pip install "termux-train[vulkan]"
```

### 3. Node.js Global CLI (NPM)
```bash
npm install -g termux-train
```

---

## 🛠️ CLI Usage Guide

```bash
# 1. Hardware & Vulkan GPU Diagnostics
termux-train doctor
# or via Python:
python3 -m termux_train.cli doctor

# 2. On-Device GEMM & Autograd Latency Benchmark
termux-train benchmark --dim 256

# 3. Train MLP / LoRA / Transformer with Checkpoints
termux-train train --model lora --dim 64 --rank 8 --epochs 5 --checkpoint ./adapter.safetensors

# 4. Stream 2,000,000+ Sample MMap Dataset
termux-train train --data ./corpus.bin --batch-size 32 --epochs 3
```

---

## 📖 Node.js & TypeScript SDK Usage

```typescript
import { TermuxTrainer, runDoctor, runBenchmark } from 'termux-train';

// 1. Diagnostics
const doc = runDoctor();
console.log(`Hardware Tier: ${doc.hardware.tier} | RAM: ${doc.hardware.totalRamMb}MB`);

// 2. Training Session
const trainer = new TermuxTrainer();
const result = await trainer.train({
  modelType: 'lora',
  dim: 64,
  loraRank: 8,
  epochs: 5,
  lr: 0.001,
  checkpointPath: './adapter.safetensors'
});

console.log(`Training complete! Final Loss: ${result.finalLoss}`);
```

---

## 📖 Official Documentation & Portal
- [Official Architecture & API Reference](https://uno-km.vercel.app/lib/train/)
- [Ecosystem Metrics & Registry Stats](https://uno-km.vercel.app/foundation/metrics)
- [AMEVA Open-Source Foundation](https://uno-km.vercel.app/foundation/index.html)

---

## 📄 License
Licensed under the Apache-2.0 License. Copyright (c) 2026 Eunho Kim ([@uno-km](https://github.com/uno-km)).
