# Termux-Train (Python)

[![PyPI](https://img.shields.io/pypi/v/termux-train.svg?style=flat-square&color=0369a1)](https://pypi.org/project/termux-train/)
[![Python](https://img.shields.io/pypi/pyversions/termux-train.svg?style=flat-square)](https://pypi.org/project/termux-train/)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/termux-train)

> Unified Multimodal On-Device Deep Learning & LoRA Training Framework for Android Termux with 6-Modality Adapters (LLM, Diffusion, VLM, STT, TTS, BitNet), GPU Slicing, and 44GB Disaggregated Cluster Virtual RAM Pooling.

## Installation

```bash
pip install termux-train
```

## Python Quickstart

```python
import termux_train as tt

# 1. Diffusion LoRA Training
tt.diffusion.train_diffusion_lora(
    image_dir="./training_images",
    output_path="./adapter_diffusion.safetensors",
    prompt="industrial technical illustration",
    resolution=512,
    epochs=5,
    lr=0.0001,
    rank=8,
    backend="vulkan"
)

# 2. Vision VLM LoRA Training
tt.vision.train_vision_vlm_lora(
    data_source="./vlm_dataset.jsonl",
    output_path="./adapter_vision.safetensors",
    epochs=3,
    backend="vulkan"
)
```

## CLI Usage

```bash
# Diffusion LoRA
termux-train diffusion-train --image-dir ./images --output ./adapter.safetensors --epochs 5

# Vision VLM LoRA
termux-train vision-train --data ./vlm.jsonl --output ./vlm_adapter.safetensors --epochs 3

# Whisper STT LoRA
termux-train stt-train --data ./audio.jsonl --output ./stt_adapter.safetensors --epochs 4

# TTS Style LoRA
termux-train tts-train --data ./speech.jsonl --output ./tts_adapter.safetensors --epochs 5

# LLM LoRA with GPU Slicing
termux-train train --model tiny-transformer --data ./corpus.txt --vocab-slice 4096 --chunk-layers 2

# Cluster Worker
termux-train cluster-worker --port 50052 --guard-band 300
```

## Documentation & Repository
- Official Documentation: https://uno-km.vercel.app/lib/train/
- GitHub Repository: https://github.com/uno-km/termux-train

## License
Apache-2.0 License. Copyright (c) 2026 Eunho Kim (@uno-km).
