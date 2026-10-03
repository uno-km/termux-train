"""
termux-train (AMEVA-Termux)
===========================
Native On-Device Deep Learning & Autograd Training Framework for Android Termux.
"""

__version__ = "1.1.8"
__author__ = "AMEVA Team"

from .backend import get_backend, set_backend, available_backends
from .tensor import Tensor, tensor, zeros, ones, zeros_like, ones_like, randn, no_grad
from . import nn
from . import optim
from . import runtime
from . import tokenization
from . import checkpoint
from . import data
from . import cluster
from . import rl
from . import adapters
from . import diffusion
from .exceptions import ClusterConnectionError, ClusterConfigurationError
from .cluster import (
    parse_cluster_rpc_spec,
    verify_rpc_cluster_nodes,
    verify_rpc_cluster_health,
    VirtualNodeInfo,
    VirtualRAMPool,
)
from .cluster_trainer import ClusterWorkerServer, ClusterPipelineSession
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
    "cluster",
    "rl",
    "adapters",
    "diffusion",
    
    # Backend
    "get_backend",
    "set_backend",
    "available_backends",
    
    # Environment & Diagnostics
    "is_termux",
    "is_android",
    "get_device_info",

    # Cluster Virtual RAM Pooling
    "VirtualNodeInfo",
    "VirtualRAMPool",
    "ClusterWorkerServer",
    "ClusterPipelineSession",
    
    "__version__",
]
