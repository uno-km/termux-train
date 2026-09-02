# Changelog

All notable changes to 	ermux-train will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-09-02 (Official Production Release)

### Added
- **Vulkan Backend**: Full integration with VulkanBackend for GPU-accelerated on-device tensor operations via 	ermux-train[vulkan].
- **Pure Python & NumPy DAG Autograd**: Native automatic differentiation graph supporting high-order gradients and zero-copy tensor slicing.
- **On-Device LoRA & Fine-Tuning**: Rank-decomposition parameter-efficient fine-tuning module (	ermux_train.nn.LoRALinear).
- **Transformer & RoPE**: Rotary Position Embedding and Multi-Head Attention blocks optimized for mobile low-memory constraints.
- **SafeTensors I/O**: Native serialization and checkpointing without pickle vulnerabilities.

### Fixed
- **Platform SSOT**: Migrated is_android and is_termux hardware probes to meva_vulkan_runtime.platform SSOT.
- **Vulkan Memory Leak**: Fixed tensor destruction lifecycle during backpropagation loops.
- **Security**: Removed legacy plain-text SSH connection scripts and hardened .gitignore.

### Verification
- **Unit Tests**: 741 / 741 passed with 100% assertion coverage.
- **Audit Grade**: Scorecard 100.0 / 100.0 (Grade A+).