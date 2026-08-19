"""
termux_train.nn.loss
====================
Loss functions for model training and optimization.
"""

from typing import Optional
from .module import Module
from ..tensor import Tensor

def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """
    [Status: Stable ✅]
    Measures the mean squared error (squared L2 norm) between each element in the input and target.
    
    Args:
        input: Predicted output Tensor.
        target: Ground truth target Tensor.
        reduction: 'mean' (default), 'sum', or 'none'.
    """
    diff = input - target
    sq = diff ** 2
    
    if reduction == "mean":
        return sq.mean()
    elif reduction == "sum":
        return sq.sum()
    elif reduction == "none":
        return sq
    else:
        raise ValueError(f"Unsupported reduction mode: '{reduction}'. Choose 'mean', 'sum', or 'none'.")


class MSELoss(Module):
    """
    [Status: Stable ✅]
    Creates a criterion that measures the mean squared error between input and target.
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return mse_loss(input, target, reduction=self.reduction)

    def __repr__(self) -> str:
        return f"MSELoss(reduction='{self.reduction}')"


def bce_loss(input: Tensor, target: Tensor, reduction: str = "mean", eps: float = 1e-7) -> Tensor:
    """
    [Status: Experimental ⚠️]
    Binary Cross Entropy Loss with numerical stability clamp:
      loss = - [target * ln(clamp(p, eps, 1-eps)) + (1 - target) * ln(clamp(1-p, eps, 1-eps))]
    
    Args:
        input: Predicted probability Tensor (values in [0, 1]).
        target: Ground truth binary target Tensor (0 or 1).
        reduction: 'mean' (default), 'sum', or 'none'.
        eps: Small epsilon clamp to prevent log(0) numerical divergence.
    """
    # 1. Clamp input probability to [eps, 1 - eps] to strictly avoid log(0) NaN
    p_clamped = input.clamp(min_val=eps, max_val=1.0 - eps)
    
    # 2. term1 = target * ln(p)
    term1 = target * p_clamped.log()
    
    # 3. term2 = (1 - target) * ln(1 - p)
    one_minus_target = 1.0 - target
    one_minus_p = (1.0 - p_clamped).clamp(min_val=eps, max_val=1.0 - eps)
    term2 = one_minus_target * one_minus_p.log()
    
    # 4. loss = - (term1 + term2)
    loss = - (term1 + term2)
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unsupported reduction mode: '{reduction}'. Choose 'mean', 'sum', or 'none'.")


class BCELoss(Module):
    """
    [Status: Experimental ⚠️]
    Binary Cross Entropy Loss module.
    """
    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return bce_loss(input, target, reduction=self.reduction, eps=self.eps)

    def __repr__(self) -> str:
        return f"BCELoss(reduction='{self.reduction}', eps={self.eps})"
