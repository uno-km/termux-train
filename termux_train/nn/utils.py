"""
termux_train.nn.utils
=====================
Neural Network Utilities: Global Gradient Norm Clipping, Parameter Flat Buffer Packing.
"""

import math
from typing import Union, Iterable, Optional
from .parameter import Parameter
from ..tensor import Tensor


def clip_grad_norm_(
    parameters: Union[Parameter, Iterable[Parameter]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False
) -> float:
    """
    Clips gradient norm of an iterable of parameters in-place.
    Matches PyTorch torch.nn.utils.clip_grad_norm_ contract.

    Args:
        parameters: an iterable of Parameters or a single Parameter with gradients
        max_norm: max norm of the gradients
        norm_type: type of the used p-norm. Can be 2.0 or float('inf').
        error_if_nonfinite: if True, raises an error if total_norm is NaN or Inf.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, Parameter):
        parameters = [parameters]
    else:
        parameters = [p for p in parameters if isinstance(p, Parameter)]

    params_with_grad = [p for p in parameters if p.grad is not None]
    if len(params_with_grad) == 0:
        return 0.0

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if norm_type == float("inf"):
        total_norm = max(
            max(abs(float(x)) for x in p.grad.backend.to_flat_list(p.grad._data))
            for p in params_with_grad
        )
    else:
        total_sum = 0.0
        for p in params_with_grad:
            flat_g = p.grad.backend.to_flat_list(p.grad._data)
            for val in flat_g:
                total_sum += abs(float(val)) ** norm_type
        total_norm = total_sum ** (1.0 / norm_type)

    if error_if_nonfinite and (math.isnan(total_norm) or math.isinf(total_norm)):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            f"`parameters` is non-finite, so it cannot be clipped. Value: {total_norm}"
        )

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params_with_grad:
            backend = p.grad.backend
            flat_g = backend.to_flat_list(p.grad._data)
            scaled_flat = [float(v) * clip_coef for v in flat_g]
            scaled_data = backend.from_data(backend.reshape(scaled_flat, p.grad.shape), dtype="float32")
            p.grad._replace_data(scaled_data, bump_version=False)

    return total_norm
