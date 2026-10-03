# Termux-Train (v2.0.0)

[![PyPI](https://img.shields.io/pypi/v/termux-train.svg?style=flat-square&color=0369a1)](https://pypi.org/project/termux-train/)
[![Python](https://img.shields.io/pypi/pyversions/termux-train.svg?style=flat-square)](https://pypi.org/project/termux-train/)
[![npm](https://img.shields.io/npm/v/termux-train.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/termux-train)
[![npm downloads](https://img.shields.io/npm/dm/termux-train.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/termux-train)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/termux-train)

> **Unified Multimodal On-Device Deep Learning & LoRA Training Framework for Android Termux with 6-Modality Adapters (LLM, Diffusion, VLM, STT, TTS, BitNet), GPU Slicing, and 44GB Disaggregated Cluster Virtual RAM Pooling**

---

## Executive Summary & Architecture Overview

**Termux-Train v2.0.0** is an industrial-grade, zero-dependency deep learning framework engineered specifically for constrained ARM64 Bionic environments (Android Termux, mobile Linux, embedded edge devices). By combining a C-vectorized Directed Acyclic Graph (DAG) Autograd runtime with mobile hardware acceleration (Vulkan Compute, OpenCL, ARM64 NEON), Termux-Train enables full backward gradient propagation, parameter-efficient fine-tuning (LoRA / DoRA), and reinforcement learning directly on mobile devices.

```
+---------------------------------------------------------------------------------------------------+
|                                        TERMUX-TRAIN v2.0.0                                        |
|                           Unified Multimodal On-Device Deep Learning Engine                       |
+---------------------------------------------------------------------------------------------------+
                                                  |
        +-----------------------------------------+-----------------------------------------+
        |                                                                                   |
        v                                                                                   v
+-------------------------------+                                           +-------------------------------+
|     6-Modality Adapters       |                                           |     Memory & Scaling Subsys   |
|  - LLM / DoRA / TinyLM        |                                           |  - Single-Device GPU Slicing  |
|  - Image Diffusion LoRA       |                                           |    * Vocab Slicing            |
|  - Vision Multimodal (VLM)    |                                           |    * Chunked Layer Dispatch   |
|  - Whisper STT Acoustic LoRA  |                                           |    * Layer Streaming          |
|  - Speech TTS Style LoRA      |                                           |  - 44GB Cluster Virtual RAM   |
|  - BitNet 1.58-bit Ternary    |                                           |  - 300MB Guard-Band Memory    |
+-------------------------------+                                           +-------------------------------+
        |                                                                                   |
        +-----------------------------------------+-----------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                             Target-Aware Runtime Export & Compatibility                           |
|    ComfyUI / Diffusers   |   llama.cpp / GGUF   |   termux-vision   |   termux-stt   |   BitNet  |
+---------------------------------------------------------------------------------------------------+
```

### Core Architectural Pillars

1. **Zero External Heavy Dependencies**: Executes autonomously without PyTorch, LibTorch, CMake, or heavy BLAS runtimes. Pure Python DAG autograd paired with optional C-accelerator ABI (`libtermux_train_accel.so`) or Vulkan compute shaders.
2. **6-Modality Target-Aware Adapters**: Generates standard SafeTensors weights and companion `_config.json` manifests natively recognized by major desktop and mobile inference runtimes without conversion tooling.
3. **Single-Device GPU Slicing**: Bypasses mobile Unified Memory Architecture (UMA) spikes and watchdog timeouts on ARM Mali and Qualcomm Adreno GPUs through granular vocabulary chunking and layer streaming.
4. **44GB Disaggregated Cluster Virtual RAM Pooling**: Binds physical memory across heterogeneous mobile nodes (e.g., Galaxy S25, S21, S20, A53, A35) into an aggregated pipeline training session protected by a 300MB memory Guard-Band against the Android Low Memory Killer (LMK).
5. **100% Dual-Engine Parity**: Complete feature and interface symmetry across Python CLI (`termux-train`), Node.js Global CLI (`termux-train`), Python Library (`import termux_train`), and Node.js SDK (`import { ... } from 'termux-train'`).

---

## Installation

### 1. Via Python Package Index (PyPI)
```bash
pip install termux-train
```

### 2. Via Node Package Manager (npm)
```bash
npm install -g termux-train
```

### 3. Standalone Bootstrap Script (Termux One-Liner)
```bash
pkg update && pkg install -y python nodejs clang
bash <(curl -sSL https://raw.githubusercontent.com/uno-km/termux-train/main/install.sh)
```

---

## 6-Modality Training Pipelines & CLI Reference

### 1. Image Diffusion LoRA Training (`diffusion-train`)
Trains latent diffusion cross-attention LoRA layers. Integrates a deterministic DDPM noise scheduler, sinusoidal timestep MLP, and SHA-256 disk-based VAE latent caching to eliminate redundant image decoding across epochs. Outputs ComfyUI- and Diffusers-compatible SafeTensors adapters.

```bash
termux-train diffusion-train \
  --image-dir ./training_images \
  --prompt "high quality product photograph, 4k" \
  --output ./adapter_diffusion.safetensors \
  --resolution 512 \
  --epochs 5 \
  --batch-size 1 \
  --lr 0.0001 \
  --rank 8 \
  --alpha 16.0 \
  --backend vulkan
```

### 2. Vision Multimodal VLM LoRA Training (`vision-train`)
Trains multimodal projection adapters and visual cross-attention layers. Operates on image-text pairs formatted in LLaVA or Qwen2-VL visual question answering schemas. Directly bridges into the `termux-vision` mobile inference runtime.

```bash
termux-train vision-train \
  --data ./vlm_dataset.jsonl \
  --output ./adapter_vision.safetensors \
  --epochs 3 \
  --batch-size 2 \
  --lr 0.0002 \
  --rank 8 \
  --alpha 16.0 \
  --backend vulkan
```

### 3. Speech-to-Text Whisper LoRA Training (`stt-train`)
Performs acoustic adaptation on Whisper encoder-decoder attention blocks. Features automated Mel filterbank spectrogram extraction and persistent caching. Natively consumable by `termux-stt`.

```bash
termux-train stt-train \
  --data ./speech_corpus.jsonl \
  --output ./adapter_stt.safetensors \
  --epochs 4 \
  --batch-size 4 \
  --lr 0.0005 \
  --rank 4 \
  --backend vulkan
```

### 4. Text-to-Speech Style Adaptation LoRA Training (`tts-train`)
Fine-tunes speaker embedding and acoustic style parameters for neural speech synthesis. Generates low-rank voice profile adapters directly compatible with `termux-tts`.

```bash
termux-train tts-train \
  --data ./voice_samples.jsonl \
  --output ./adapter_tts.safetensors \
  --epochs 5 \
  --batch-size 2 \
  --lr 0.0003 \
  --rank 4 \
  --backend cpu
```

### 5. Large Language Model PEFT (`train` / `peft`)
Trains LoRA / DoRA weight matrices over multi-head attention and feed-forward projections with RoPE positional encodings. Supports single-device GPU slicing flags to eliminate out-of-memory crashes on mobile GPUs. Exports standard GGUF PEFT adapters for `llama.cpp`.

```bash
termux-train train \
  --model tiny-transformer \
  --data ./training_corpus.txt \
  --output ./adapter_llm.safetensors \
  --epochs 3 \
  --lr 0.0002 \
  --lora-rank 8 \
  --lora-alpha 16.0 \
  --vocab-slice 4096 \
  --chunk-layers 2 \
  --stream-layers \
  --backend vulkan
```

### 6. BitNet 1.58-bit Ternary Quantization-Aware Training
Applies dynamic activation scaling and ternary weight quantization $\{-1, 0, +1\}$ during forward and backward passes. Directly exports quantized kernels compatible with `bitnet.cpp` and `termux-bitnet`.

```bash
termux-train train \
  --model bitnet \
  --quantize-bits 1.58 \
  --data ./corpus.txt \
  --output ./bitnet_adapter.safetensors \
  --backend opencl
```

---

## Complete Parameter Reference Matrix

| Parameter Flag | Type | Default | Applicable Subcommands | Engineering Semantics & Usage Description |
| :--- | :--- | :--- | :--- | :--- |
| `--image-dir` | String | *Required* | `diffusion-train` | Absolute or relative path to directory containing source training images (`.png`, `.jpg`). |
| `--prompt` | String | `""` | `diffusion-train` | Conditioning text prompt applied across image diffusion training instances. |
| `--resolution` | Integer | `512` | `diffusion-train` | Input image dimension for latent VAE spatial encoding ($H=W$). |
| `--data` | String | *Required* | `vision-train`, `stt-train`, `tts-train`, `train`, `rl` | Path to dataset file (`.jsonl`, `.txt`, `.csv`). |
| `--output` | String | *Required* | All training subcommands | Destination filepath for exported `.safetensors` adapter file. |
| `--epochs` | Integer | `3` | All training subcommands | Total complete passes over the dataset. |
| `--batch-size` | Integer | `1` | All training subcommands | Micro-batch size per training step. |
| `--lr` | Float | `0.0001` | All training subcommands | Optimization learning rate for AdamW / SGD gradient update. |
| `--rank` / `--lora-rank` | Integer | `8` | All training subcommands | Inner rank dimension $r$ for LoRA projection matrices $A \in \mathbb{R}^{r \times d_{in}}$ and $B \in \mathbb{R}^{d_{out} \times r}$. |
| `--alpha` / `--lora-alpha`| Float | `16.0` | All training subcommands | Scaling hyperparameter $\alpha$ for LoRA weight contribution ($\Delta W = \frac{\alpha}{r} BA$). |
| `--backend` | String | `auto` | All subcommands | Compute acceleration backend: `auto`, `vulkan`, `opencl`, `cpu`. |
| `--vocab-slice` | Integer | `0` (off) | `train`, `peft` | Granular vocabulary partition size to prevent GPU out-of-memory spikes during softmax cross-entropy. |
| `--chunk-layers` | Integer | `0` (off) | `train`, `peft` | Number of sequential transformer layers evaluated per GPU dispatch to prevent OS watchdog timeout resets. |
| `--stream-layers` | Boolean | `false` | `train`, `peft` | Dynamically streams inactive layer weights between system RAM and GPU buffer memory. |
| `--virtual-ram-pool` | String | `""` | `train`, `diffusion-train`, `peft` | Cluster connection endpoint (`host:port`) binding local training session to aggregated virtual RAM pool. |
| `--guard-band` | Integer | `300` | `cluster-worker`, `train` | Reserved unallocated system RAM buffer (in MB) preventing Android Low Memory Killer (LMK) aborts. |
| `--method` | String | `grpo` | `rl` | Reinforcement learning optimization algorithm: `grpo`, `dpo`, `ppo`. |
| `--group-size` | Integer | `4` | `rl` | Generation group candidate size for Group Relative Policy Optimization (GRPO). |
| `--beta` | Float | `0.1` | `rl` | KL divergence regularization coefficient for DPO and GRPO reference policy stabilization. |
| `--host` | String | `0.0.0.0` | `cluster-worker`, `cluster-probe`| Network host address for disaggregated virtual RAM clustering daemon. |
| `--port` | Integer | `50052` | `cluster-worker`, `cluster-probe`| Network TCP port for cluster worker coordination. |

---

## Single-Device GPU Slicing Architecture

On mobile platforms, the GPU and CPU share a Unified Memory Architecture (UMA). Large batch matrix multiplications during autograd activation accumulation trigger sudden memory surges that prompt the Android system to issue SIGKILL signals. Furthermore, ARM Mali GPU driver stacks enforce a hardware watchdog timeout: any single compute command buffer executing longer than 2.0 seconds is terminated with `VK_ERROR_DEVICE_LOST`.

```
                        MOBILE UNIFIED RAM (LPDDR5)
+-------------------------------------------------------------------------+
| OS Kernel / Termux Base | Active Layer Buffer | Guard-Band Buffer (300MB)|
+-------------------------------------------------------------------------+
                                   ^
                                   |  Layer Streaming Swap
                                   v
                        VULKAN / OPENCL GPU ENGINE
+-------------------------------------------------------------------------+
| [Vocab Chunk 1] -> [Vocab Chunk 2] | [Layer N] -> [Layer N+1] (Chunked) |
+-------------------------------------------------------------------------+
```

To neutralize these constraints:
- **Vocab Slicing (`--vocab-slice 4096`)**: Partitions vocabulary projection matrices into slices of 4,096 tokens, performing sequential cross-entropy reductions with minimal temporary tensor overhead.
- **Chunked Layer Dispatch (`--chunk-layers 2`)**: Segments multi-layer transformer forward-backward passes into 2-layer dispatches, returning control to the Vulkan queue to reset watchdog timers.
- **Layer Streaming (`--stream-layers`)**: Maintains only active weight layers inside device GPU memory buffers, paging inactive weights into host RAM.

---

## 44GB Disaggregated Cluster Virtual RAM Pooling

When training architectures exceeding physical device capacity, Termux-Train binds heterogeneous smartphone clusters into a unified virtual memory pool. Idle mobile devices contribute unallocated RAM via lightweight worker daemons.

```
+-------------------------------------------------------------------------+
|              COORDINATOR / PRIMARY NODE (Galaxy S25 - 12GB)             |
|                Directs Training Pipeline & Weight Updates               |
+-------------------------------------------------------------------------+
           |                           |                           |
  gRPC/HTTP RPC               gRPC/HTTP RPC               gRPC/HTTP RPC
           v                           v                           v
+--------------------+      +--------------------+      +--------------------+
| WORKER 1 (S21)     |      | WORKER 2 (S20)     |      | WORKER 3 (A53/A35) |
| RAM: 8GB LPDDR5    |      | RAM: 12GB LPDDR5   |      | RAM: 12GB LPDDR4X  |
| Dedicated: 6.8GB   |      | Dedicated: 10.5GB  |      | Dedicated: 9.2GB   |
+--------------------+      +--------------------+      +--------------------+
           \                           |                           /
            +--------------------------+--------------------------+
                                       |
                                       v
                       TOTAL AGGREGATED VIRTUAL POOL: ~44GB
```

### Launching Cluster Workers on Fleet Nodes
On worker nodes (e.g., Galaxy S21, S20, A53):
```bash
termux-train cluster-worker --port 50052 --guard-band 300
```

### Probing and Discovering Fleet Nodes
On coordinator node:
```bash
termux-train cluster-probe --fleet 192.168.1.101:50052,192.168.1.102:50052,192.168.1.103:50052
```

### Initiating Distributed Pipeline Training
```bash
termux-train train \
  --model tiny-transformer \
  --data ./large_corpus.txt \
  --output ./cluster_adapter.safetensors \
  --virtual-ram-pool 192.168.1.101:50052,192.168.1.102:50052,192.168.1.103:50052 \
  --epochs 5 \
  --backend vulkan
```

---

## Programmatic SDK Usage

### Python SDK
```python
import termux_train as tt

# High-level training interface
trainer = tt.TermuxTrainer(
    model_type="lora",
    lora_rank=8,
    lora_alpha=16.0,
    backend="vulkan",
    vocab_slice=4096,
    chunk_layers=2
)

metrics = trainer.fit(
    data_path="./corpus.txt",
    epochs=3,
    learning_rate=0.0002,
    output_path="./model_lora.safetensors"
)
print(f"Final training loss: {metrics['final_loss']:.4f}")
```

### Node.js / TypeScript SDK
```typescript
import { trainDiffusion, trainVision, trainSTT, TermuxTrainer } from 'termux-train';

// Asynchronous image diffusion fine-tuning
const result = await trainDiffusion({
  imageDir: './training_images',
  output: './adapter_diffusion.safetensors',
  prompt: 'photorealistic macro nature photography',
  resolution: 512,
  epochs: 5,
  backend: 'vulkan'
});
console.log(`Diffusion training completed with status: ${result.status}`);
```

---

## Target Runtime Compatibility & Adapter Hub

All adapters generated by Termux-Train are zero-conversion artifacts accompanied by a standardized `_config.json` sidecar.

| Target Runtime | Adapter Format | Companion Sidecar | Validation Status |
| :--- | :--- | :--- | :--- |
| **ComfyUI / Diffusers** | `.safetensors` (UNet CrossAttention) | `adapter_diffusion_config.json` | Validated by automated checks |
| **llama.cpp / GGUF** | `.safetensors` / GGUF LoRA | `adapter_llm_config.json` | Validated by automated checks |
| **termux-vision** | `.safetensors` (Multimodal Projector) | `adapter_vision_config.json` | Validated by automated checks |
| **termux-stt** | `.safetensors` (Whisper Attention) | `adapter_stt_config.json` | Validated by automated checks |
| **termux-tts** | `.safetensors` (Speaker Style) | `adapter_tts_config.json` | Validated by automated checks |
| **bitnet.cpp** | `.safetensors` (Ternary 1.58-bit) | `bitnet_config.json` | Validated by automated checks |

---

## Official Documentation & Foundation Resources

- [Official Architecture & API Reference](https://uno-km.vercel.app/lib/train/)
- [Ecosystem Metrics & Registry Stats](https://uno-km.vercel.app/foundation/metrics)
- [AMEVA Open-Source Foundation Portal](https://uno-km.vercel.app/foundation/index.html)

---

## Open-Source Attribution & License

Licensed under the **Apache-2.0 License**. Copyright (c) 2026 Eunho Kim ([@uno-km](https://github.com/uno-km)) and the AMEVA Open-Source Foundation.
