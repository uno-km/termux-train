"""
termux_train.nn.loss
====================
Numerically Stable Loss Functions for Deep Learning on Mobile Constrained Devices.
Includes MSELoss, BCELoss, BCEWithLogitsLoss, and Fused CrossEntropyLoss.
"""

import math
from typing import Optional
from .module import Module
from ..tensor import Tensor, _attach_grad_fn


def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """
    Measures the mean squared error (squared L2 norm) between each element in the input and target.

    Args:
        input: Predicted output Tensor.
        target: Ground truth target Tensor.
        reduction: 'mean' (default), 'sum', or 'none'.
    """
    target = input._ensure_tensor_on_self_backend(target)
    if input.shape != target.shape:
        raise ValueError(f"MSELoss requires identical input and target shapes, got {input.shape} and {target.shape}")

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
    Binary Cross Entropy Loss with probability input in [0, 1].
    """
    target = input._ensure_tensor_on_self_backend(target)
    if input.shape != target.shape:
        raise ValueError(f"BCELoss requires identical input and target shapes, got {input.shape} and {target.shape}")

    p_clamped = input.clamp(min_val=eps, max_val=1.0 - eps)
    term1 = target * p_clamped.log()
    one_minus_target = 1.0 - target
    one_minus_p = (1.0 - p_clamped).clamp(min_val=eps, max_val=1.0 - eps)
    term2 = one_minus_target * one_minus_p.log()
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


def binary_cross_entropy_with_logits(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean"
) -> Tensor:
    """
    Numerically stable Binary Cross Entropy with Logits:
      loss = max(x, 0) - x * y + log(1 + exp(-abs(x)))
    Avoids gradient vanishing caused by premature probability clamping.
    """
    target = input._ensure_tensor_on_self_backend(target)
    if input.shape != target.shape:
        raise ValueError(
            f"BCEWithLogitsLoss requires identical input and target shapes, "
            f"got {input.shape} and {target.shape}"
        )

    backend = input.backend
    flat_x = backend.to_flat_list(input._data)
    flat_y = backend.to_flat_list(target._data)

    losses = []
    for x_val, y_val in zip(flat_x, flat_y):
        # max(x, 0) - x * y + log1p(exp(-abs(x)))
        max_val = max(x_val, 0.0)
        neg_abs = -abs(x_val)
        log1p = math.log1p(math.exp(neg_abs))
        l = max_val - x_val * y_val + log1p
        losses.append(l)

    if reduction == "mean":
        res_val = sum(losses) / max(1, len(losses))
        out = Tensor(res_val, dtype="float32", requires_grad=input.requires_grad or target.requires_grad, _prev=(input, target), _op="bce_with_logits", backend=backend)
    elif reduction == "sum":
        res_val = sum(losses)
        out = Tensor(res_val, dtype="float32", requires_grad=input.requires_grad or target.requires_grad, _prev=(input, target), _op="bce_with_logits", backend=backend)
    elif reduction == "none":
        res_data = backend.from_data(backend.reshape(losses, input.shape), dtype="float32")
        out = Tensor(res_data, dtype="float32", requires_grad=input.requires_grad or target.requires_grad, _prev=(input, target), _op="bce_with_logits", backend=backend)
    else:
        raise ValueError(f"Unsupported reduction mode: '{reduction}'. Choose 'mean', 'sum', or 'none'.")

    if out.requires_grad:
        s_input = input.shape
        count = len(losses) if reduction == "mean" else 1.0

        def _backward():
            if out.grad is not None and input.requires_grad:
                # dL/dx = (sigmoid(x) - y) * grad_output
                sig_x = [1.0 / (1.0 + math.exp(-x_val)) for x_val in flat_x]
                flat_g = backend.to_flat_list(out.grad._data) if reduction == "none" else [out.grad.item()] * len(flat_x)
                d_x = [((s - y) / count) * g for s, y, g in zip(sig_x, flat_y, flat_g)]
                d_input = backend.from_data(backend.reshape(d_x, s_input), dtype="float32")
                input._accumulate_grad_data(d_input)

        _attach_grad_fn(out, (input, target), _backward)
    return out


bce_with_logits_loss = binary_cross_entropy_with_logits


class BCEWithLogitsLoss(Module):
    """
    Binary Cross Entropy with Logits module.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return binary_cross_entropy_with_logits(input, target, reduction=self.reduction)

    def __repr__(self) -> str:
        return f"BCEWithLogitsLoss(reduction='{self.reduction}')"


def cross_entropy_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    ignore_index: int = -100
) -> Tensor:
    """
    Fused Cross-Entropy Loss directly accepting (..., C) logits and (...) int64 target indices.
    Eliminates O(V) One-Hot vector memory inflation, preventing mobile LMK kills.
    """
    target = input._ensure_tensor_on_self_backend(target)
    if target.dtype not in ("int64", "int32"):
        raise TypeError(
            f"CrossEntropyLoss target must have integer dtype ('int64' or 'int32'), "
            f"got dtype='{target.dtype}'. For probability targets, use soft cross-entropy."
        )
    if input.ndim < 1:
        raise ValueError(f"CrossEntropyLoss requires input with at least 1 dimension, got shape {input.shape}")
    c_dim = input.shape[-1]
    if c_dim == 0:
        raise ValueError(f"CrossEntropyLoss vocab/class dimension must be > 0, got shape {input.shape}")

    expected_target_shape = input.shape[:-1]
    if target.shape != expected_target_shape:
        raise ValueError(
            f"CrossEntropyLoss expected target shape {expected_target_shape}, got {target.shape}"
        )

    backend = input.backend

    # Compute log_softmax over last dimension
    log_probs = input.log_softmax(axis=-1)
    flat_lp = backend.to_flat_list(log_probs._data)
    flat_targets = backend.to_flat_list(target._data)

    num_samples = len(flat_targets)
    losses = []
    valid_indices = []

    for i, t_val in enumerate(flat_targets):
        t_idx = int(t_val)
        if t_idx == ignore_index:
            losses.append(0.0)
            continue
        if not (0 <= t_idx < c_dim):
            raise IndexError(f"Target index {t_idx} is out of bounds for vocab/class size {c_dim}")

        lp = flat_lp[i * c_dim + t_idx]
        losses.append(-lp)
        valid_indices.append((i, t_idx))

    valid_count = len(valid_indices)
    if valid_count == 0:
        out = Tensor(0.0, dtype="float32", requires_grad=input.requires_grad, _prev=(input,), _op="cross_entropy", backend=backend)
        if out.requires_grad:
            def _backward():
                if out.grad is not None and input.requires_grad:
                    d_input = backend.zeros(input.shape, dtype="float32")
                    input._accumulate_grad_data(d_input)
            _attach_grad_fn(out, (input,), _backward)
        return out

    if reduction == "mean":
        res_val = sum(losses) / float(valid_count)
        out = Tensor(res_val, dtype="float32", requires_grad=input.requires_grad, _prev=(input,), _op="cross_entropy", backend=backend)
    elif reduction == "sum":
        res_val = sum(losses)
        out = Tensor(res_val, dtype="float32", requires_grad=input.requires_grad, _prev=(input,), _op="cross_entropy", backend=backend)
    elif reduction == "none":
        res_data = backend.from_data(backend.reshape(losses, target.shape), dtype="float32")
        out = Tensor(res_data, dtype="float32", requires_grad=input.requires_grad, _prev=(input,), _op="cross_entropy", backend=backend)
    else:
        raise ValueError(f"Unsupported reduction mode: '{reduction}'. Choose 'mean', 'sum', or 'none'.")

    if out.requires_grad:
        s_input = input.shape
        softmax_probs = input.softmax(axis=-1)
        flat_probs = backend.to_flat_list(softmax_probs._data)

        def _backward():
            if out.grad is not None and input.requires_grad:
                # dL/dx = (softmax(x) - 1(x=target)) * grad_scale
                scale = (1.0 / float(valid_count)) if reduction == "mean" else 1.0
                grad_val = out.grad.item() if reduction in ("mean", "sum") else None
                flat_out_g = backend.to_flat_list(out.grad._data) if reduction == "none" else None

                d_flat = [0.0] * (num_samples * c_dim)

                # Active elements
                target_map = {i: t for i, t in valid_indices}
                for i in range(num_samples):
                    if i not in target_map:
                        continue
                    t_idx = target_map[i]
                    g_factor = grad_val if grad_val is not None else flat_out_g[i]

                    for c in range(c_dim):
                        idx = i * c_dim + c
                        p = flat_probs[idx]
                        grad_elem = (p - (1.0 if c == t_idx else 0.0)) * scale * g_factor
                        d_flat[idx] = grad_elem

                d_input = backend.from_data(backend.reshape(d_flat, s_input), dtype="float32")
                input._accumulate_grad_data(d_input)

        _attach_grad_fn(out, (input,), _backward)
    return out


class CrossEntropyLoss(Module):
    """
    Fused Cross-Entropy Loss module.
    """
    def __init__(self, reduction: str = "mean", ignore_index: int = -100):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return cross_entropy_loss(input, target, reduction=self.reduction, ignore_index=self.ignore_index)

    def __repr__(self) -> str:
        return f"CrossEntropyLoss(reduction='{self.reduction}', ignore_index={self.ignore_index})"
