#!/data/data/com.termux/files/usr/bin/bash
# ==============================================================================
# termux-train (AMEVA-Termux) - Android Termux Native Environment Setup Script
# ==============================================================================

set -e

echo "========================================================"
echo "🚀 [termux-train] Initializing Android Termux Environment"
echo "========================================================"

# 1. Update Termux Package Repositories
echo "[*] Updating apt packages..."
pkg update -y && pkg upgrade -y

# 2. Install Core Build & Runtime Dependencies
echo "[*] Installing core development packages (Python, NumPy C-Accelerated Engine, Clang, Git, OpenSSH, Make)..."
pkg install -y python python-numpy python-pip git clang make cmake openssh nano vim pkg-config

# 3. Display Python & System Versions
echo "========================================================"
echo "✅ Base Environment Ready:"
echo " - Python: $(python3 --version 2>&1)"
echo " - NumPy:  $(python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'Not installed')"
echo " - Clang:  $(clang --version | head -n 1)"
echo " - Git:    $(git --version)"
echo "========================================================"
