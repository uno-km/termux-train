"""
termux_train.nn
===============
Neural Network layers, containers, modules, activations, losses, and LoRA adapters.
"""

from .module import Module
from .parameter import Parameter
from .linear import Linear
from .sequential import Sequential
from .activations import ReLU, Sigmoid, Tanh
from .loss import mse_loss, MSELoss, bce_loss, BCELoss
from .lora import LoRALinear, adapter_parameters, named_adapter_parameters

__all__ = [
    "Module",
    "Parameter",
    "Linear",
    "Sequential",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "mse_loss",
    "MSELoss",
    "bce_loss",
    "BCELoss",
    "LoRALinear",
    "adapter_parameters",
    "named_adapter_parameters",
]
