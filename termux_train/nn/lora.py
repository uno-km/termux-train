"""
termux_train.nn.lora
====================
Low-Rank Adaptation (LoRA) Layer for Parameter-Efficient On-Device Fine-Tuning.
Freezes pre-trained base Linear weights and learns decomposed low-rank matrices
lora_A (in_features, rank) and lora_B (rank, out_features) with scaling factor alpha / rank.
"""

import math
import random
from typing import Optional, List, Tuple
from .module import Module
from .parameter import Parameter
from .linear import Linear
from ..tensor import Tensor
from ..backend import get_backend, BaseBackend


class LoRALinear(Module):
    """
    Applies Low-Rank Adaptation (LoRA) to an affine linear layer:
      y = base(x) + ((x @ lora_A) @ lora_B) * (alpha / rank)

    Args:
        in_features: Size of each input sample (int >= 1).
        out_features: Size of each output sample (int >= 1).
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
    ):
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

        if not isinstance(bias, bool):
            raise TypeError(f"bias must be a boolean, got {type(bias).__name__}")

        if backend is not None and not isinstance(backend, BaseBackend):
            raise TypeError(f"backend must be a BaseBackend instance, got {type(backend).__name__}")

        b = backend or get_backend()
        self._alpha = float(alpha)
        self._scaling = float(alpha) / float(rank)
        if not math.isfinite(self._scaling) or self._scaling <= 0.0:
            raise ValueError("scaling factor must be finite and positive")

        self._merged = False

        # 1. Base Linear Layer (Frozen)
        self.base = Linear(in_features=in_features, out_features=out_features, bias=bias, backend=b)
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # 2. LoRA Factor A: (in_features, rank) initialized with small random values
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

    @classmethod
    def from_linear(
        cls,
        linear: Linear,
        rank: int = 4,
        alpha: float = 1.0,
    ) -> "LoRALinear":
        """
        Wraps an existing pre-trained Linear layer with LoRA adapters while preserving
        the original Linear submodule and parameter identities.
        """
        if not isinstance(linear, Linear):
            raise TypeError(f"linear must be a Linear instance, got {type(linear).__name__}")

        lora = cls.__new__(cls)
        Module.__init__(lora)

        in_features = linear.in_features
        out_features = linear.out_features

        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError(f"rank must be an integer >= 1, got {rank}")
        max_rank = min(in_features, out_features)
        if rank > max_rank:
            raise ValueError(f"rank must be <= min(in_features, out_features) ({max_rank}), got {rank}")

        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError(f"alpha must be a finite positive number, got {alpha}")

        lora._alpha = float(alpha)
        lora._scaling = float(alpha) / float(rank)
        if not math.isfinite(lora._scaling) or lora._scaling <= 0.0:
            raise ValueError("scaling factor must be finite and positive")

        lora._merged = False

        # Preserve original linear object identity and freeze parameters
        lora.base = linear
        linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False

        b = linear.weight.backend
        bound = 1.0 / math.sqrt(in_features) if in_features > 0 else 1.0
        lora_A_data = [
            [random.uniform(-bound, bound) for _ in range(rank)]
            for _ in range(in_features)
        ]
        lora.lora_A = Parameter(lora_A_data, requires_grad=True, backend=b)

        lora_B_data = [
            [0.0 for _ in range(out_features)]
            for _ in range(rank)
        ]
        lora.lora_B = Parameter(lora_B_data, requires_grad=True, backend=b)

        return lora

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def rank(self) -> int:
        return self.lora_A.shape[1]

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
        """Returns the learnable adapter parameters [lora_A, lora_B]. Base parameters are excluded."""
        return [self.lora_A, self.lora_B]

    def named_adapter_parameters(self) -> List[Tuple[str, Parameter]]:
        """Returns the named adapter parameters [('lora_A', lora_A), ('lora_B', lora_B)]."""
        return [("lora_A", self.lora_A), ("lora_B", self.lora_B)]

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward computation:
          y = base(x) + ((x @ lora_A) @ lora_B) * scaling
        Supported inputs:
          1D: (in_features,) -> (out_features,)
          2D: (batch_size, in_features) -> (batch_size, out_features)
          3D: (batch_size, sequence_length, in_features) -> (batch_size, sequence_length, out_features)
        """
        if not isinstance(x, Tensor):
            raise TypeError(f"LoRALinear expects a Tensor input, got {type(x).__name__}")
        if x.ndim not in (1, 2, 3):
            raise ValueError(
                f"LoRALinear expects a 1D, 2D, or 3D input, but received shape {x.shape}"
            )
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"LoRALinear expected input.shape[-1] == {self.in_features}, but received shape {x.shape}"
            )

        base_output = self.base(x)
        if self._merged:
            return base_output

        # Low-rank factor projection path
        adapter_hidden = x @ self.lora_A
        adapter_output = adapter_hidden @ self.lora_B
        return base_output + adapter_output * self._scaling

    def __repr__(self) -> str:
        return (
            f"LoRALinear(in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}, "
            f"bias={self.base.bias is not None}, merged={self.merged})"
        )


def adapter_parameters(module: Module) -> List[Parameter]:
    """
    Recursively collects all learnable LoRA adapter parameters (lora_A and lora_B)
    from all LoRALinear layers within the given module hierarchy.
    Base weights and biases are strictly excluded.
    Returns parameters in deterministic order without duplicates.
    """
    params = []
    visited_ids = set()
    for _, param in named_adapter_parameters(module):
        param_id = id(param)
        if param_id not in visited_ids:
            visited_ids.add(param_id)
            params.append(param)
    return params


def named_adapter_parameters(module: Module, prefix: str = "") -> List[Tuple[str, Parameter]]:
    """
    Recursively collects all named LoRA adapter parameters from all LoRALinear layers.
    Yields (name, parameter) tuples such as:
      ('0.lora_A', lora_A), ('0.lora_B', lora_B), ...
    """
    named_params: List[Tuple[str, Parameter]] = []
    visited_modules = set()

    def _traverse(m: Module, curr_prefix: str):
        if id(m) in visited_modules:
            return
        visited_modules.add(id(m))

        if isinstance(m, LoRALinear):
            name_a = f"{curr_prefix}.lora_A" if curr_prefix else "lora_A"
            name_b = f"{curr_prefix}.lora_B" if curr_prefix else "lora_B"
            named_params.append((name_a, m.lora_A))
            named_params.append((name_b, m.lora_B))
            return

        for sub_name, sub_m in m._modules.items():
            if sub_m is not None:
                sub_prefix = f"{curr_prefix}.{sub_name}" if curr_prefix else sub_name
                _traverse(sub_m, sub_prefix)

    _traverse(module, prefix)
    return named_params
