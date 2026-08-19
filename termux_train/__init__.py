"""
termux-train (AMEVA-Termux)
===========================
Native On-Device Deep Learning & Autograd Training Framework for Android Termux.
"""

__version__ = "0.1.0-alpha"
__author__ = "AMEVA Team"

from .backend import get_backend, set_backend, available_backends
from .tensor import Tensor, tensor, zeros, ones, zeros_like, ones_like, randn
from . import nn
from . import optim
from .utils.termux_env import is_termux, is_android, get_device_info, get_battery_info, get_thermal_info

__all__ = [
    # Core
    "Tensor",
    "tensor",
    "zeros",
    "ones",
    "zeros_like",
    "ones_like",
    "randn",
    
    # NN & Optim Submodules
    "nn",
    "optim",
    
    # Backend
    "get_backend",
    "set_backend",
    "available_backends",
    
    # Environment & Diagnostics
    "is_termux",
    "is_android",
    "get_device_info",
    "get_battery_info",
    "get_thermal_info",
    
    "__version__",
]
