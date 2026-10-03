# Contributing to termux-train

Thank you for your interest in contributing to `termux-train` (AMEVA-Termux)!

## Core Architectural Principles

1. **Zero-Heavy C++ Build Dependency**: Do not introduce dependencies that require host PyTorch or node-gyp compilation on mobile devices.
2. **Fail-Closed Parameter Boundaries**: Always validate inputs strictly. Never employ silent dummy fallbacks for missing or invalid values.
3. **Dual-Engine Parity**: Any new feature must be verified across both Python API and Node.js SDK / CLI.
4. **0-Point Baseline Compliance**: All functions and tests must pass strict regression suites with 100% pass rate.
5. **Memory Safety & Guard-Band Protection**: Ensure mobile memory safety by respecting the 300MB Guard-Band to prevent Android Low Memory Killer (LMK) aborts.

## Development Workflow

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/uno-km/termux-train.git
   cd termux-train
   ```

2. Install development dependencies:
   ```bash
   pip install -e .
   pip install pytest
   npm install
   ```

3. Run automated tests:
   ```bash
   python -m pytest tests/ -v
   npm test
   ```
