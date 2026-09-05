"""
termux-train (AMEVA-Termux)
===========================
Native On-Device Deep Learning & Autograd Training Framework for Android Termux.
"""

__version__ = "1.1.4"
__author__ = "AMEVA Team"

from .backend import get_backend, set_backend, available_backends
from .tensor import Tensor, tensor, zeros, ones, zeros_like, ones_like, randn, no_grad
from . import nn
from . import optim
from . import runtime
from . import tokenization
from . import checkpoint
from . import data
from .utils.termux_env import is_termux, is_android, get_device_info

__all__ = [
    # Core
    "Tensor",
    "tensor",
    "zeros",
    "ones",
    "zeros_like",
    "ones_like",
    "randn",
    "no_grad",
    
    # Submodules
    "nn",
    "optim",
    "runtime",
    "tokenization",
    "checkpoint",
    "data",
    
    # Backend
    "get_backend",
    "set_backend",
    "available_backends",
    
    # Environment & Diagnostics
    "is_termux",
    "is_android",
    "get_device_info",
    
    "__version__",
]
