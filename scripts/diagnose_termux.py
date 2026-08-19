#!/usr/bin/env python3
"""
scripts/diagnose_termux.py
==========================
Run system diagnostics and generate reports/termux_environment_report.md
"""

import os
import sys
import json

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from termux_train.utils.termux_env import get_device_info

def main():
    print("🔍 Running termux-train Environment Diagnostics...")
    info = get_device_info()
    
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "termux_environment_report.md")
    
    md_content = f"""# 📱 Android Termux Environment Diagnostic Report

> **Generated on**: {os.popen('date /t' if os.name == 'nt' else 'date').read().strip()}
> **Framework**: `termux-train` v0.1.0-alpha

---

## 🖥️ System & Runtime Summary

| Metric | Detected Value | Status / Notes |
| :--- | :--- | :--- |
| **Operating System** | `{info['platform']}` | {'✅ Android Native' if info['is_android'] else '💻 Desktop / Emulated'} |
| **Termux Environment** | `{info['is_termux']}` | {'✅ Termux Native Bionic' if info['is_termux'] else '⚠️ Standard Host / Non-Termux'} |
| **Python Version** | `{info['python_version']}` | ✅ Python 3.x Ready |
| **CPU Architecture** | `{info['cpu']['architecture']}` | {'✅ ARM64 / aarch64 (Android Native)' if info['cpu']['is_arm64'] else 'ℹ️ Host Architecture'} |
| **Logical CPU Cores** | `{info['cpu']['cores_logical']}` | Octa-Core Multithreading Available |
| **Total System RAM** | `{info['memory'].get('total_mb', 'N/A')} MB` | {'Available: ' + str(info['memory'].get('available_mb', 'N/A')) + ' MB'} |
| **Disk Space** | `{info['storage'].get('free_gb', 'N/A')} GB Free / {info['storage'].get('total_gb', 'N/A')} GB Total` | Storage Guard Operational |
| **Battery Level** | `{info['battery'].get('percentage', 'N/A')}%` | Status: `{info['battery'].get('status', 'N/A')}` |
| **Max Temperature** | `{info['thermal'].get('max_temp_c', 'N/A')} °C` | Thermal Guard Limit: 43.0°C |
| **Hardware OpenCL** | `{info['opencl_available']}` | Adreno/Mali GPU Discovery |

---

## 📊 Detailed Raw Diagnostics (JSON)

```json
{json.dumps(info, indent=2, ensure_ascii=False)}
```
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"✅ Diagnostic report saved successfully to: {report_path}")
    print("\n--- Summary ---")
    print(f" - Android Detected: {info['is_android']}")
    print(f" - Termux Detected:  {info['is_termux']}")
    print(f" - CPU Architecture: {info['cpu']['architecture']} ({info['cpu']['cores_logical']} cores)")
    print(f" - Python Version:   {info['python_version']}")

if __name__ == "__main__":
    main()
