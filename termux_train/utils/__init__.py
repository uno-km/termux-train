"""
termux-train utilities module
=============================
"""

from .termux_env import is_termux, is_android, get_device_info, get_battery_info, get_thermal_info
from .gradcheck import gradcheck

__all__ = [
    "is_termux",
    "is_android",
    "get_device_info",
    "get_battery_info",
    "get_thermal_info",
    "gradcheck",
]
