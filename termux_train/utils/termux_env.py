"""
termux_train.utils.termux_env
=============================
Hardware and Operating Environment Diagnostics for Android Termux.
"""

import os
import sys
import platform
import subprocess
import shutil
import json
from typing import Dict, Any, Optional

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

def get_memory_info() -> Dict[str, Any]:
    """Inspect system RAM availability."""
    info: Dict[str, Any] = {"total_mb": None, "available_mb": None, "free_mb": None}
    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val_str = parts[1].strip().split()[0]
                        if val_str.isdigit():
                            val_mb = int(val_str) // 1024
                            if key == "MemTotal":
                                info["total_mb"] = val_mb
                            elif key == "MemAvailable":
                                info["available_mb"] = val_mb
                            elif key == "MemFree":
                                info["free_mb"] = val_mb
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

def get_battery_info() -> Dict[str, Any]:
    """Inspect battery percentage and status via termux-battery-status or sysfs."""
    info: Dict[str, Any] = {"percentage": None, "status": "unknown", "temperature": None, "plugged": None}
    
    # 1. Try termux-battery-status CLI
    if shutil.which("termux-battery-status"):
        try:
            out = subprocess.check_output(["termux-battery-status"], timeout=2)
            data = json.loads(out.decode("utf-8"))
            info["percentage"] = data.get("percentage")
            info["status"] = data.get("status")
            info["temperature"] = data.get("temperature")
            info["plugged"] = data.get("plugged")
            return info
        except Exception:
            pass

    # 2. Try sysfs power supply fallback
    power_path = "/sys/class/power_supply/battery"
    if os.path.exists(power_path):
        try:
            cap_file = os.path.join(power_path, "capacity")
            if os.path.exists(cap_file):
                with open(cap_file, "r") as f:
                    info["percentage"] = int(f.read().strip())
            stat_file = os.path.join(power_path, "status")
            if os.path.exists(stat_file):
                with open(stat_file, "r") as f:
                    info["status"] = f.read().strip()
        except Exception:
            pass
            
    return info

def get_thermal_info() -> Dict[str, Any]:
    """Inspect device temperature sensors."""
    info: Dict[str, Any] = {"max_temp_c": None, "zones": {}}
    thermal_dir = "/sys/class/thermal"
    if os.path.exists(thermal_dir):
        try:
            max_t = 0.0
            for name in os.listdir(thermal_dir):
                if name.startswith("thermal_zone"):
                    temp_file = os.path.join(thermal_dir, name, "temp")
                    type_file = os.path.join(thermal_dir, name, "type")
                    if os.path.exists(temp_file):
                        with open(temp_file, "r") as f:
                            raw = f.read().strip()
                            if raw.lstrip("-").isdigit():
                                t_c = float(raw) / 1000.0 if float(raw) > 1000 else float(raw)
                                z_type = name
                                if os.path.exists(type_file):
                                    with open(type_file, "r") as tf:
                                        z_type = tf.read().strip()
                                info["zones"][z_type] = round(t_c, 1)
                                if t_c > max_t:
                                    max_t = t_c
            if max_t > 0:
                info["max_temp_c"] = round(max_t, 1)
        except Exception:
            pass
    return info

def get_device_info() -> Dict[str, Any]:
    """Comprehensive system diagnosis report dictionary."""
    return {
        "is_android": is_android(),
        "is_termux": is_termux(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_compiler": platform.python_compiler(),
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "storage": get_storage_info(),
        "battery": get_battery_info(),
        "thermal": get_thermal_info(),
        "opencl_available": os.path.exists("/system/vendor/lib64/libOpenCL.so") or os.path.exists("/vendor/lib64/libOpenCL.so"),
    }
