#!/data/data/com.termux/files/usr/bin/bash
# ==============================================================================
# termux-train (AMEVA-Termux) - SSH Always-On Server Daemon Script
# ==============================================================================

set -e

PORT=8022

echo "========================================================"
echo "🔒 [termux-train] Starting Termux SSH Server on Port $PORT"
echo "========================================================"

# Ensure OpenSSH is installed
if ! command -v sshd >/dev/null 2>&1; then
    echo "[*] Installing openssh..."
    pkg install -y openssh
fi

# Ensure Wake-Lock
echo "[*] Requesting Wake-Lock..."
termux-wake-lock 2>/dev/null || true

# Start sshd daemon
sshd -p $PORT

# Get user and IP address
USER_NAME=$(whoami)
IP_ADDR=$(ifconfig 2>/dev/null | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -n 1 || hostname -I | awk '{print $1}')

echo "========================================================"
echo "✅ SSH Server Active!"
echo " - User: $USER_NAME"
echo " - Port: $PORT"
echo " - Phone IP: ${IP_ADDR:-<Check with 'ip addr'>}"
echo ""
echo "💻 Connect from Windows/PC via:"
echo "   ssh -p $PORT $USER_NAME@$IP_ADDR"
echo "========================================================"
