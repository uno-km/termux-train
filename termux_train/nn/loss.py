"""
termux_train.nn.loss
====================
Loss functions with Fused CrossEntropy, BCEWithLogits, and MSE Loss.
Includes NumPy C-level vectorization and NaN/Inf defenses.
"""

import math
from typing import Optional, Any
from .module import Module
from ..tensor import Tensor, _attach_grad_fn

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Mean Squared Error (MSE) loss: L = (input - target)^2."""
    target = input._ensure_tensor_on_self_backend(target)
    diff = input - target
    sq_diff = diff * diff
    if reduction == "mean":
        return sq_diff.mean()
    elif reduction == "sum":
        return sq_diff.sum()
    elif reduction == "none":
        return sq_diff
    else:
        raise ValueError(f"Unsupported reduction: '{reduction}'. Must be 'mean', 'sum', or 'none'.")


class MSELoss(Module):
    """Mean Squared Error loss module."""
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return mse_loss(input, target, reduction=self.reduction)

    def __repr__(self) -> str:
        return f"MSELoss(reduction='{self.reduction}')"


def bce_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    eps: float = 1e-12
) -> Tensor:
    """Binary Cross Entropy (BCE) loss: L = - (y * log(p) + (1-y) * log(1-p))."""
    target = input._ensure_tensor_on_self_backend(target)
    if input.shape != target.shape:
        raise ValueError(f"BCELoss requires identical input and target shapes, got {input.shape} vs {target.shape}")
    backend = input.backend
    p_clamped = input.clamp(min_val=eps, max_val=1.0 - eps)
    flat_p = backend.to_flat_list(p_clamped._data)
    flat_t = backend.to_flat_list(target._data)
    losses = []
    for p_val, t_val in zip(flat_p, flat_t):
        if not (0.0 <= t_val <= 1.0):
            raise ValueError(f"BCELoss target values must be in range [0, 1], got {t_val}")
        p_safe = max(eps, min(1.0 - eps, p_val))
        loss_val = -(t_val * math.log(p_safe) + (1.0 - t_val) * math.log(max(eps, 1.0 - p_safe)))
        losses.append(loss_val)

    if reduction == "mean":
        res_val = sum(losses) / max(1, len(losses))
        out = Tensor(res_val, dtype="float32", requires_grad=input.requires_grad or target.requires_grad, _prev=(input, target), _op="bce", backend=backend)
    elif reduction == "sum":
        res_val = sum(losses)
        out = Tensor(res_val, dtype="float32", requires_grad=input.requires_grad or target.requires_grad, _prev=(input, target), _op="bce", backend=backend)
    elif reduction == "none":
        res_data = backend.from_data(backend.reshape(losses, input.shape), dtype="float32")
        out = Tensor(res_data, dtype="float32", requires_grad=input.requires_grad or target.requires_grad, _prev=(input, target), _op="bce", backend=backend)
    else:
        raise ValueError(f"Unsupported reduction mode: '{reduction}'. Choose 'mean', 'sum', or 'none'.")

    if out.requires_grad:
        s_input = input.shape
        count = len(losses) if reduction == "mean" else 1.0
        def _backward():
            if out.grad is not None:
                flat_g = backend.to_flat_list(out.grad._data) if reduction == "none" else [out.grad.item()] * len(losses)
                if input.requires_grad:
                    d_x = [(((p - t) / (p * (1.0 - p) + 1e-15)) / count) * g for p, t, g in zip(flat_p, flat_t, flat_g)]
                    d_input = backend.from_data(backend.reshape(d_x, s_input), dtype="float32")
                    input._accumulate_grad_data(d_input)
                if target.requires_grad:
                    d_y = [((math.log(1.0 - p) - math.log(p)) / count) * g for p, g in zip(flat_p, flat_g)]
                    d_target = backend.from_data(backend.reshape(d_y, s_input), dtype="float32")
                    target._accumulate_grad_data(d_target)
        _attach_grad_fn(out, (input, target), _backward)
    return out


class BCELoss(Module):
    """Binary Cross Entropy loss module."""
    def __init__(self, reduction: str = "mean", eps: float = 1e-12):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return bce_loss(input, target, reduction=self.reduction, eps=self.eps)

    def __repr__(self) -> str:
        return f"BCELoss(reduction='{self.reduction}')"


def binary_cross_entropy_with_logits(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean"
) -> Tensor:
    """Numerically stable BCE with Logits: max(x, 0) - x*z + log(1 + exp(-|x|))."""
    target = input._ensure_tensor_on_self_backend(target)
    if input.shape != target.shape:
        raise ValueError(
            f"BCEWithLogitsLoss requires identical input and target shapes, got {input.shape} vs {target.shape}"
        )
    backend = input.backend
    flat_x = backend.to_flat_list(input._data)
    flat_y = backend.to_flat_list(target._data)

    losses = []
    for x_val, y_val in zip(flat_x, flat_y):
        if not (0.0 <= y_val <= 1.0):
            raise ValueError(f"BCEWithLogitsLoss target values must be in range [0, 1], got {y_val}")
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
            if out.grad is not None:
                sig_x = [1.0 / (1.0 + math.exp(-x_val)) for x_val in flat_x]
                flat_g = backend.to_flat_list(out.grad._data) if reduction == "none" else [out.grad.item()] * len(flat_x)
                if input.requires_grad:
                    d_x = [((s - y) / count) * g for s, y, g in zip(sig_x, flat_y, flat_g)]
                    d_input = backend.from_data(backend.reshape(d_x, s_input), dtype="float32")
                    input._accumulate_grad_data(d_input)
                if target.requires_grad:
                    d_y = [(-x_val / count) * g for x_val, g in zip(flat_x, flat_g)]
                    d_target = backend.from_data(backend.reshape(d_y, s_input), dtype="float32")
                    target._accumulate_grad_data(d_target)

        _attach_grad_fn(out, (input, target), _backward)
    return out


bce_with_logits_loss = binary_cross_entropy_with_logits


class BCEWithLogitsLoss(Module):
    """Binary Cross Entropy with Logits module."""
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
    Vectorized for C-speed on NumPy with Pure Python fallback.
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
    s_input = input.shape
    log_probs = input.log_softmax(axis=-1)

    # NumPy Fast Path
    if HAS_NUMPY and isinstance(input._data, np.ndarray) and getattr(backend, "name", "").lower() == "numpy":
        flat_lp = log_probs._data.reshape(-1, c_dim)
        flat_tgts = target._data.flatten()
        num_samples = len(flat_tgts)

        valid_mask = (flat_tgts != ignore_index)
        valid_tgts = flat_tgts[valid_mask]

        if len(valid_tgts) > 0:
            min_t = int(np.min(valid_tgts))
            max_t = int(np.max(valid_tgts))
            if min_t < 0 or max_t >= c_dim:
                bad_idx = min_t if min_t < 0 else max_t
                raise IndexError(f"Target index {bad_idx} is out of bounds for vocab/class size {c_dim}")

        valid_count = int(np.sum(valid_mask))
        if valid_count == 0:
            out = Tensor(0.0, dtype="float32", requires_grad=input.requires_grad, _prev=(input,), _op="cross_entropy", backend=backend)
            if out.requires_grad:
                def _backward_zero():
                    if out.grad is not None and input.requires_grad:
                        input._accumulate_grad_data(np.zeros(s_input, dtype=np.float32))
                _attach_grad_fn(out, (input,), _backward_zero)
            return out

        row_idx = np.arange(num_samples)[valid_mask]
        col_idx = valid_tgts
        selected_lp = flat_lp[row_idx, col_idx]
        sample_losses = np.zeros(num_samples, dtype=np.float32)
        sample_losses[valid_mask] = -selected_lp

        if reduction == "mean":
            res_val = float(np.sum(sample_losses) / float(valid_count))
            out = Tensor(res_val, dtype="float32", requires_grad=input.requires_grad, _prev=(input,), _op="cross_entropy", backend=backend)
        elif reduction == "sum":
            res_val = float(np.sum(sample_losses))
            out = Tensor(res_val, dtype="float32", requires_grad=input.requires_grad, _prev=(input,), _op="cross_entropy", backend=backend)
        elif reduction == "none":
            res_data = sample_losses.reshape(target.shape)
            out = Tensor(res_data, dtype="float32", requires_grad=input.requires_grad, _prev=(input,), _op="cross_entropy", backend=backend)
        else:
            raise ValueError(f"Unsupported reduction mode: '{reduction}'. Choose 'mean', 'sum', or 'none'.")

        if out.requires_grad:
            softmax_probs = input.softmax(axis=-1)
            def _backward_np():
                if out.grad is not None and input.requires_grad:
                    scale = (1.0 / float(valid_count)) if reduction == "mean" else 1.0
                    g_factor = out.grad._data if reduction == "none" else out.grad.item()
                    d_lp = softmax_probs._data.reshape(-1, c_dim).copy()
                    d_lp[row_idx, col_idx] -= 1.0

                    if reduction == "none":
                        d_lp *= (g_factor.reshape(-1, 1) * scale)
                    else:
                        d_lp *= (g_factor * scale)

                    d_lp[~valid_mask] = 0.0
                    input._accumulate_grad_data(d_lp.reshape(s_input))

            _attach_grad_fn(out, (input,), _backward_np)
        return out

    # Pure Python Fallback
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
        softmax_probs = input.softmax(axis=-1)
        flat_probs = backend.to_flat_list(softmax_probs._data)

        def _backward_py():
            if out.grad is not None and input.requires_grad:
                scale = (1.0 / float(valid_count)) if reduction == "mean" else 1.0
                grad_val = out.grad.item() if reduction in ("mean", "sum") else None
                flat_out_g = backend.to_flat_list(out.grad._data) if reduction == "none" else None

                d_flat = [0.0] * (num_samples * c_dim)
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

        _attach_grad_fn(out, (input,), _backward_py)
    return out


class CrossEntropyLoss(Module):
    """Fused Cross-Entropy Loss module."""
    def __init__(self, reduction: str = "mean", ignore_index: int = -100):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return cross_entropy_loss(input, target, reduction=self.reduction, ignore_index=self.ignore_index)

    def __repr__(self) -> str:
        return f"CrossEntropyLoss(reduction='{self.reduction}', ignore_index={self.ignore_index})"
