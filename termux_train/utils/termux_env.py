"""
termux_train.utils.termux_env
=============================
Platform and Operating Environment Identification for Android Termux.
"""

import os
import sys
import platform
import shutil
from typing import Dict, Any

def is_android() -> bool:
    """Check if the current runtime is running on Android."""
    if os.path.exists("/system/build.prop"):
        return True
    if "ANDROID_ROOT" in os.environ or "ANDROID_DATA" in os.environ:
        return True
    return False

def is_termux() -> bool:
    """Check if running specifically inside the Termux native environment."""
    prefix = os.environ.get("PREFIX", "")
    if "com.termux" in prefix:
        return True
    if os.path.exists("/data/data/com.termux"):
        return True
    return False

def get_cpu_info() -> Dict[str, Any]:
    """Inspect CPU architecture, core count, and processor model."""
    info = {
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cores_logical": os.cpu_count() or 1,
        "is_arm64": platform.machine().lower() in ("aarch64", "arm64"),
    }
    
    # Try reading /proc/cpuinfo on Linux/Android
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                info["cpuinfo_snippet"] = content[:300].strip()
        except Exception:
            pass
    return info

def get_storage_info() -> Dict[str, Any]:
    """Inspect disk space in current working directory."""
    try:
        usage = shutil.disk_usage(os.getcwd())
        return {
            "total_gb": round(usage.total / (1024**3), 2),
            "used_gb": round(usage.used / (1024**3), 2),
            "free_gb": round(usage.free / (1024**3), 2),
        }
    except Exception:
        return {"total_gb": None, "free_gb": None}

def get_device_info() -> Dict[str, Any]:
    """Platform and OS environment metadata dictionary."""
    return {
        "is_android": is_android(),
        "is_termux": is_termux(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_compiler": platform.python_compiler(),
        "cpu": get_cpu_info(),
        "storage": get_storage_info(),
        "opencl_available": os.path.exists("/system/vendor/lib64/libOpenCL.so") or os.path.exists("/vendor/lib64/libOpenCL.so"),
    }
