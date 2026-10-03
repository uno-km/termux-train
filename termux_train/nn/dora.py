"""
termux_train.nn.dora
====================
Weight-Decomposed Low-Rank Adaptation (DoRA) Layer for High-Accuracy On-Device Fine-Tuning.
Decomposes affine linear weights into magnitude (m) and directional component (V = W_0 + (alpha/r)*A*B).
Normalized Direction: V_norm = V / ||V||_c
Effective Weight: W = m * V_norm
Provides enhanced learning capacity and numerical stability matching full fine-tuning.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from .module import Module
from .parameter import Parameter
from .linear import Linear
from ..tensor import Tensor, zeros, ones
from ..backend import get_backend, BaseBackend


class DoRALinear(Module):
    """
    Applies Weight-Decomposed Low-Rank Adaptation (DoRA) to an affine linear layer:
      V = W_0 + (alpha / rank) * (A @ B)
      V_norm = V / ||V||_c (column-wise or feature-wise L2 norm)
      W = m * V_norm
      y = x @ W + bias

    Args:
        in_features: Size of input features (int >= 1).
        out_features: Size of output features (int >= 1).
        rank: Rank of decomposed matrices (1 <= rank <= min(in_features, out_features)).
        alpha: Scaling numerator (finite positive float or int).
        bias: Whether to learn an additive bias in base Linear.
        backend: Backend instance to use for tensor storage and compute.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        bias: bool = True,
        backend: Optional[BaseBackend] = None,
    ) -> None:
        super().__init__()

        if isinstance(in_features, bool) or not isinstance(in_features, int) or in_features < 1:
            raise ValueError(f"in_features must be an integer >= 1, got {in_features}")
        if isinstance(out_features, bool) or not isinstance(out_features, int) or out_features < 1:
            raise ValueError(f"out_features must be an integer >= 1, got {out_features}")

        max_rank = min(in_features, out_features)
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError(f"rank must be an integer >= 1, got {rank}")
        if rank > max_rank:
            raise ValueError(f"rank must be <= min(in_features, out_features) ({max_rank}), got {rank}")

        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError(f"alpha must be a finite positive number, got {alpha}")

        b = backend or get_backend()
        self._alpha = float(alpha)
        self._rank = rank
        self._scaling = float(alpha) / float(rank)
        self._merged: bool = False
        self._base_weight_snapshot: Optional[Any] = None

        # 1. Base Linear Layer (Frozen base weight)
        self.base = Linear(in_features=in_features, out_features=out_features, bias=bias, backend=b)
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # 2. LoRA Factor A: (in_features, rank) initialized with Uniform(-bound, bound)
        bound = 1.0 / math.sqrt(in_features) if in_features > 0 else 1.0
        lora_A_data = [
            [random.uniform(-bound, bound) for _ in range(rank)]
            for _ in range(in_features)
        ]
        self.lora_A = Parameter(lora_A_data, requires_grad=True, backend=b)

        # 3. LoRA Factor B: (rank, out_features) initialized to exact zeros
        lora_B_data = [
            [0.0 for _ in range(out_features)]
            for _ in range(rank)
        ]
        self.lora_B = Parameter(lora_B_data, requires_grad=True, backend=b)

        # 4. Magnitude Vector m: (1, out_features), initialized to column norms of W_0
        # W_0 shape: (in_features, out_features)
        w0_flat = self.base.weight.backend.to_flat_list(self.base.weight._data)
        col_norms = []
        for c in range(out_features):
            col_sq = sum(w0_flat[r * out_features + c] ** 2 for r in range(in_features))
            col_norms.append(math.sqrt(max(col_sq, 1e-12)))
        self.magnitude = Parameter([col_norms], requires_grad=True, backend=b)

    @classmethod
    def from_linear(
        cls,
        linear: Linear,
        rank: int = 4,
        alpha: float = 1.0,
    ) -> "DoRALinear":
        """Wraps an existing pre-trained Linear layer with DoRA adapters."""
        if not isinstance(linear, Linear):
            raise TypeError(f"linear must be a Linear instance, got {type(linear).__name__}")

        dora = cls.__new__(cls)
        Module.__init__(dora)

        in_features = linear.in_features
        out_features = linear.out_features

        max_rank = min(in_features, out_features)
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError(f"rank must be an integer >= 1, got {rank}")
        if rank > max_rank:
            raise ValueError(f"rank must be <= min(in_features, out_features) ({max_rank}), got {rank}")

        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError(f"alpha must be a finite positive number, got {alpha}")

        dora._alpha = float(alpha)
        dora._rank = rank
        dora._scaling = float(alpha) / float(rank)
        dora._merged = False
        dora._base_weight_snapshot = None

        dora.base = linear
        linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False

        b = linear.weight.backend
        bound = 1.0 / math.sqrt(in_features) if in_features > 0 else 1.0
        lora_A_data = [
            [random.uniform(-bound, bound) for _ in range(rank)]
            for _ in range(in_features)
        ]
        dora.lora_A = Parameter(lora_A_data, requires_grad=True, backend=b)

        lora_B_data = [
            [0.0 for _ in range(out_features)]
            for _ in range(rank)
        ]
        dora.lora_B = Parameter(lora_B_data, requires_grad=True, backend=b)

        w0_flat = b.to_flat_list(linear.weight._data)
        col_norms = []
        for c in range(out_features):
            col_sq = sum(w0_flat[r * out_features + c] ** 2 for r in range(in_features))
            col_norms.append(math.sqrt(max(col_sq, 1e-12)))
        dora.magnitude = Parameter([col_norms], requires_grad=True, backend=b)

        return dora

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def scaling(self) -> float:
        return self._scaling

    @property
    def merged(self) -> bool:
        return self._merged

    def adapter_parameters(self) -> List[Parameter]:
        """Returns learnable DoRA parameters [lora_A, lora_B, magnitude]."""
        return [self.lora_A, self.lora_B, self.magnitude]

    def named_adapter_parameters(self) -> List[Tuple[str, Parameter]]:
        """Returns named DoRA parameters."""
        return [
            ("lora_A", self.lora_A),
            ("lora_B", self.lora_B),
            ("magnitude", self.magnitude),
        ]

    def compute_effective_weight(self) -> Tensor:
        """Computes effective decomposed weight W = m * (V / ||V||_c)."""
        delta = (self.lora_A @ self.lora_B) * self._scaling
        v = self.base.weight + delta  # Shape: (in_features, out_features)
        
        # Column norm: sum(v**2, axis=0, keepdims=True).sqrt() -> (1, out_features)
        v_sq = v ** 2
        col_norm = v_sq.sum(axis=0, keepdims=True).sqrt()
        # Safe eps to prevent divide by zero
        eps = 1e-8
        v_norm = v / (col_norm + eps)
        # Scale by magnitude: (1, out_features) broadcasts across (in_features, out_features)
        w = v_norm * self.magnitude
        return w

    def forward(self, x: Tensor) -> Tensor:
        if self._merged:
            out = x @ self.base.weight
            if self.base.bias is not None:
                out = out + self.base.bias
            return out

        w = self.compute_effective_weight()
        out = x @ w
        if self.base.bias is not None:
            out = out + self.base.bias
        return out

    def merge(self) -> None:
        """Merges DoRA weight permanently into base weight."""
        if self._merged:
            raise RuntimeError("DoRALinear is already merged")

        import copy
        b = self.base.weight.backend
        if hasattr(self.base.weight._data, "copy"):
            self._base_weight_snapshot = self.base.weight._data.copy()
        elif isinstance(self.base.weight._data, list):
            self._base_weight_snapshot = copy.deepcopy(self.base.weight._data)
        else:
            self._base_weight_snapshot = b.from_data(copy.deepcopy(self.base.weight.tolist()))

        effective_w = self.compute_effective_weight()
        self.base.weight._replace_data(effective_w._data)
        self._merged = True

    def unmerge(self) -> None:
        """Restores original pre-trained base weight from snapshot."""
        if not self._merged:
            raise RuntimeError("DoRALinear is not merged")
        if self._base_weight_snapshot is None:
            raise RuntimeError("Base weight snapshot missing; cannot unmerge")

        import copy
        if hasattr(self._base_weight_snapshot, "copy"):
            restored = self._base_weight_snapshot.copy()
        elif isinstance(self._base_weight_snapshot, list):
            restored = copy.deepcopy(self._base_weight_snapshot)
        else:
            restored = self._base_weight_snapshot

        self.base.weight._replace_data(restored)
        self._base_weight_snapshot = None
        self._merged = False

    def adapter_state_dict(self) -> Dict[str, Any]:
        """Returns DoRA adapter state dict with A, B, and magnitude."""
        b = self.lora_A.backend
        return {
            "type": "dora",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "rank": self.rank,
            "alpha": self.alpha,
            "scaling": self.scaling,
            "lora_A": b.to_flat_list(self.lora_A._data),
            "lora_B": b.to_flat_list(self.lora_B._data),
            "magnitude": b.to_flat_list(self.magnitude._data),
            "merged": self._merged,
        }

    def __repr__(self) -> str:
        return (
            f"DoRALinear(in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, merged={self._merged})"
        )
