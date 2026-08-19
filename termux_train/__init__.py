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
from . import runtime
from . import tokenization
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
    
    # NN, Optim, Runtime & Tokenization Submodules
    "nn",
    "optim",
    "runtime",
    "tokenization",
    
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
