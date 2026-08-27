# Android Termux Remote SSH Development & Testing Guide

> **Scope:** Connecting to physical Android devices over SSH for on-device ML verification.

---

## 1. Setup OpenSSH on Termux
```bash
pkg update -y && pkg install -y openssh
passwd
sshd
whoami
```

## 2. Connect from Host Machine
```bash
ssh -p 8022 u0_aXXX@<DEVICE_IP>
```
