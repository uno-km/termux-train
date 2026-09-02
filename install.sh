#!/usr/bin/env bash
# =======================================================================
# termux-train: One-Touch Universal Automated Installer
# Supported OS: Android Termux (ARM64/x86_64), Linux, macOS
# Open-Source under Apache License 2.0
# ======================================================================

set -e

COLOR_GREEN='\033[0;32m'
COLOR_BLUE='\033[0;34m'
COLOR_YELLOW='\033[1;33m'
COLOR_RED='\033[0;31m'
COLOR_RESET='\033[0m'

echo -e "${COLOR_BLUE}================================================================${COLOR_RESET}"
echo -e "${COLOR_BLUE}    termux-train: Universal One-Touch Installation Engine       ${COLOR_RESET}"
echo -e "${COLOR_BLUE}================================================================${COLOR_RESET}"

# 1. Environment Detection
IS_TERMUX=false
if [ -n "$TERMUX_VERSION" ] || [ -d "/data/data/com.termux" ]; then
    IS_TERMUX=true
    echo -e "${COLOR_GREEN}[+] Detected Environment: Android Termux${COLOR_RESET}"
else
    echo -e "${COLOR_GREEN}[+] Detected Environment: Standard Linux / POSIX System${COLOR_RESET}"
fi

# 2. System Dependencies
if [ "$IS_TERMUX" = true ]; then
    echo -e "\n${COLOR_BLUE}[1/4] Installing Termux System Packages (Python, NumPy, Node.js, Clang)...${COLOR_RESET}"
    pkg update -y || true
    pkg install -y python python-numpy nodejs clang make binutils || {
        echo -e "${COLOR_YELLOW}[!] pkg install warning: continuing with pip/npm fallbacks...${COLOR_RESET}"
    }
else
    echo -e "\n${COLOR_BLUE}[1/4] Checking System Python & Node.js...${COLOR_RESET}"
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        echo -e "${COLOR_RED}[!] Error: Python 3 is required. Please install python3 first.${COLOR_RESET}"
        exit 1
    fi
fi

# 3. Resolve Python Command
PY_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PY_CMD="python"
fi

# 4. Python Package Installation
echo -e "\n${COLOR_BLUE}[2/4] Installing Python termux-train Engine & Dependencies...${COLOR_RESET}"
$PY_CMD -m pip install --upgrade pip setuptools wheel || true

if [ "$IS_TERMUX" = true ]; then
    $PY_CMD -m pip install -e .
else
    $PY_CMD -m pip install -e ".[all]" || $PY_CMD -m pip install -e ".[accelerated]" || $PY_CMD -m pip install -e .
fi

# 5. Node.js SDK Link
if command -v npm &> /dev/null; then
    echo -e "\n${COLOR_BLUE}[3/4] Linking Node.js CLI globally (npm link)...${COLOR_RESET}"
    npm link || npm install -g . || true
fi

# 6. Automated Doctor Diagnostic Verification
echo -e "\n${COLOR_BLUE}[4/4] Running Diagnostic Doctor Self-Verification...${COLOR_RESET}"
if command -v node &> /dev/null; then
    node bin/cli.js doctor
else
    $PY_CMD -m termux_train.cli doctor
fi

echo -e "\n${COLOR_GREEN}================================================================${COLOR_RESET}"
echo -e "${COLOR_GREEN} [+] termux-train successfully installed and ready!             ${COLOR_RESET}"
echo -e "${COLOR_GREEN}     - Node CLI   : npx termux-train --help                     ${COLOR_RESET}"
echo -e "${COLOR_GREEN}     - Python CLI : python3 -m termux_train.cli --help          ${COLOR_RESET}"
echo -e "${COLOR_GREEN}================================================================${COLOR_RESET}"