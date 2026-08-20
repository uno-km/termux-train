"""
termux_train.nn.lora
====================
Low-Rank Adaptation (LoRA) Layer for Parameter-Efficient On-Device Fine-Tuning.
Freezes pre-trained base Linear weights and learns decomposed low-rank matrices
lora_A (in_features, rank) and lora_B (rank, out_features) with scaling factor alpha / rank.
Supports atomic, crash-resilient adapter serialization and transactional merge/unmerge lifecycles.
"""

import copy
import math
import random
from typing import Optional, List, Tuple, Dict, Any, Union
from .module import Module
from .parameter import Parameter
from .linear import Linear
from ..tensor import Tensor
from ..backend import get_backend, BaseBackend


def _validate_positive_int_metadata(value: Any, expected: int, name: str) -> None:
    """Validates that a metadata value is a strictly positive integer (not bool) matching expected value."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"'{name}' must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"'{name}' must be >= 1, got {value}")
    if value != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {value}")


def _validate_alpha_metadata(value: Any, expected: float) -> None:
    """Validates that alpha is a finite positive numeric scalar (not bool) matching expected value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"'alpha' must be a finite number, got {type(value).__name__}")
    f_val = float(value)
    if not math.isfinite(f_val) or f_val <= 0.0:
        raise ValueError(f"'alpha' must be finite and positive, got {value}")
    if f_val != expected:
        raise ValueError(f"alpha mismatch: expected {expected}, got {value}")


def _validate_string_keys(d: Dict[Any, Any], context_name: str) -> None:
    """Validates that all keys in dictionary are strings."""
    non_str = [k for k in d if not isinstance(k, str)]
    if non_str:
        raise TypeError(f"{context_name} keys must be strings, found non-string keys: {non_str!r}")


def _validate_2d_matrix_data(data: Any, expected_shape: Tuple[int, int], name: str) -> None:
    """Validates that data is a regular 2D list of finite numeric values matching expected_shape."""
    if not isinstance(data, list):
        raise TypeError(f"'{name}' must be a 2D list, got {type(data).__name__}")
    if len(data) != expected_shape[0]:
        raise ValueError(f"'{name}' row count mismatch: expected {expected_shape[0]}, got {len(data)}")
    for r_idx, row in enumerate(data):
        if not isinstance(row, list):
            raise TypeError(f"'{name}' row {r_idx} must be a list, got {type(row).__name__}")
        if len(row) != expected_shape[1]:
            raise ValueError(f"'{name}' row {r_idx} column count mismatch: expected {expected_shape[1]}, got {len(row)}")
        for c_idx, val in enumerate(row):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"'{name}'[{r_idx}][{c_idx}] must be a float or int, got {val!r}")
            if not math.isfinite(val):
                raise ValueError(f"'{name}'[{r_idx}][{c_idx}] must be finite, got {val}")


def _assert_param_finite(param: Parameter, name: str) -> None:
    """Asserts that all elements in param._data are finite numeric values."""
    flat = param.backend.to_flat_list(param._data)
    for v in flat:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"Non-finite value found in parameter '{name}': {v}")


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

        self._merged: bool = False
        self._base_weight_snapshot: Optional[Any] = None

        # 1. Base Linear Layer (Frozen)
        self.base = Linear(in_features=in_features, out_features=out_features, bias=bias, backend=b)
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # 2. LoRA Factor A: (in_features, rank) initialized with small random values from Uniform(-1/sqrt(in), 1/sqrt(in))
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
        lora._base_weight_snapshot = None

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

    def merge(self) -> None:
        """
        adapter delta를 base weight에 반영합니다 (W_base <- W_base + (A @ B) * scaling).
        merge 직전의 base weight backend-native deep snapshot을 내부에 보관하여 unmerge 시 정확한 복원을 지원합니다.
        Base Parameter 객체 식별자(id)와 requires_grad=False는 온전히 유지됩니다.
        이미 merged 상태이거나 unmerged 상태에서 snapshot이 존재하는 경우 RuntimeError를 발생시킵니다.
        """
        if self._merged:
            raise RuntimeError("LoRALinear is already merged")
        if self._base_weight_snapshot is not None:
            raise RuntimeError("LoRALinear has an unexpected base weight snapshot while unmerged")

        if (
            isinstance(self.scaling, bool)
            or not isinstance(self.scaling, (int, float))
            or not math.isfinite(float(self.scaling))
            or float(self.scaling) <= 0.0
        ):
            raise ValueError(f"scaling factor must be a finite positive number, got {self.scaling}")

        b = self.base.weight.backend
        if self.lora_A.backend.name != b.name or self.lora_B.backend.name != b.name:
            raise RuntimeError("Cross-backend LoRALinear parameters are not supported for merge")

        # 1. Preflight validations
        _assert_param_finite(self.base.weight, "base.weight")
        _assert_param_finite(self.lora_A, "lora_A")
        _assert_param_finite(self.lora_B, "lora_B")

        if self.lora_A.shape != (self.in_features, self.rank):
            raise ValueError(f"lora_A shape mismatch: expected {(self.in_features, self.rank)}, got {self.lora_A.shape}")
        if self.lora_B.shape != (self.rank, self.out_features):
            raise ValueError(f"lora_B shape mismatch: expected {(self.rank, self.out_features)}, got {self.lora_B.shape}")
        if self.base.weight.shape != (self.in_features, self.out_features):
            raise ValueError(f"base.weight shape mismatch: expected {(self.in_features, self.out_features)}, got {self.base.weight.shape}")

        expected_shape = self.base.weight.shape

        # 2. Native calculation
        delta = b.matmul(self.lora_A._data, self.lora_B._data)
        if b.get_shape(delta) != expected_shape:
            raise RuntimeError(f"Computed delta shape mismatch: expected {expected_shape}, got {b.get_shape(delta)}")

        delta = b.mul(delta, float(self.scaling))
        merged_weight = b.add(self.base.weight._data, delta)
        if b.get_shape(merged_weight) != expected_shape:
            raise RuntimeError(f"Computed merged_weight shape mismatch: expected {expected_shape}, got {b.get_shape(merged_weight)}")

        # Validate merged weight
        flat_merged = b.to_flat_list(merged_weight)
        for v in flat_merged:
            if not math.isfinite(v):
                raise ValueError(f"Non-finite value encountered in merged weight: {v}")

        # 3. Create independent deep snapshot of original base weight
        snapshot = b.from_data(copy.deepcopy(self.base.weight.tolist()))

        # 4. Transactional commit with full rollback
        old_base_data = self.base.weight._data
        old_snapshot = self._base_weight_snapshot
        old_merged = self._merged

        try:
            self.base.weight._replace_data(merged_weight, bump_version=True)
            self._base_weight_snapshot = snapshot
            self._merged = True
        except Exception as commit_err:
            rollback_errors = []
            try:
                self.base.weight._replace_data(old_base_data, bump_version=True)
            except Exception as r_err:
                rollback_errors.append(f"base.weight rollback failed: {r_err}")
            try:
                self._base_weight_snapshot = old_snapshot
            except Exception as r_err:
                rollback_errors.append(f"_base_weight_snapshot rollback failed: {r_err}")
            try:
                self._merged = old_merged
            except Exception as r_err:
                rollback_errors.append(f"_merged rollback failed: {r_err}")

            if rollback_errors:
                raise RuntimeError(
                    f"LoRA merge failed ({commit_err}) AND rollback failed: {'; '.join(rollback_errors)}"
                ) from commit_err
            raise commit_err

    def unmerge(self) -> None:
        """
        merge 시점의 backend-native deep snapshot을 복원하여 base weight를 merge 이전 상태로 되돌립니다.
        현재 adapter delta를 빼는 방식이 아니므로, merge 이후 adapter가 변경되었더라도 원본 base weight를 정확히 복원합니다.
        Base Parameter 객체 식별자(id)와 requires_grad=False는 온전히 유지됩니다.
        unmerged 상태이거나 snapshot이 누락/손상된 경우 RuntimeError를 발생시킵니다.
        """
        if not self._merged:
            raise RuntimeError("LoRALinear is not merged")
        if self._base_weight_snapshot is None:
            raise RuntimeError("Missing base weight snapshot for unmerge")

        b = self.base.weight.backend
        snap_shape = b.get_shape(self._base_weight_snapshot)
        if snap_shape != self.base.weight.shape:
            raise RuntimeError(
                f"Base weight snapshot shape mismatch: expected {self.base.weight.shape}, got {snap_shape}"
            )

        flat_snap = b.to_flat_list(self._base_weight_snapshot)
        for v in flat_snap:
            if not math.isfinite(v):
                raise ValueError(f"Non-finite value found in base weight snapshot: {v}")

        # Transactional commit with full rollback
        old_base_data = self.base.weight._data
        old_snapshot = self._base_weight_snapshot
        old_merged = self._merged

        try:
            self.base.weight._replace_data(self._base_weight_snapshot, bump_version=True)
            self._base_weight_snapshot = None
            self._merged = False
        except Exception as commit_err:
            rollback_errors = []
            try:
                self.base.weight._replace_data(old_base_data, bump_version=True)
            except Exception as r_err:
                rollback_errors.append(f"base.weight rollback failed: {r_err}")
            try:
                self._base_weight_snapshot = old_snapshot
            except Exception as r_err:
                rollback_errors.append(f"_base_weight_snapshot rollback failed: {r_err}")
            try:
                self._merged = old_merged
            except Exception as r_err:
                rollback_errors.append(f"_merged rollback failed: {r_err}")

            if rollback_errors:
                raise RuntimeError(
                    f"LoRA unmerge failed ({commit_err}) AND rollback failed: {'; '.join(rollback_errors)}"
                ) from commit_err
            raise commit_err

    def adapter_state_dict(self) -> Dict[str, Any]:
        """
        Exports adapter-only parameters (lora_A and lora_B) as a detached, deep-copied dictionary.
        Base weights, biases, optimizer states, and merge snapshots are strictly excluded.
        """
        return {
            "format": "termux-train-lora-adapter",
            "version": "1.0",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "rank": self.rank,
            "alpha": self.alpha,
            "lora_A": copy.deepcopy(self.lora_A.tolist()),
            "lora_B": copy.deepcopy(self.lora_B.tolist()),
        }

    def load_adapter_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """
        Atomically loads adapter parameters from state_dict into this LoRALinear layer.

        Two-phase atomic commit:
          1. Pre-validates all metadata, schema, and 2D matrix buffers.
          2. Builds pending native backend buffers.
          3. Creates snapshot of existing native buffers.
          4. Commits pending buffers with automatic rollback on commit exception.
        """
        if self._merged:
            raise RuntimeError("Cannot load adapter state into a merged LoRALinear layer. Unmerge before loading.")

        if not isinstance(state_dict, dict):
            raise TypeError(f"state_dict must be a dict, got {type(state_dict).__name__}")

        _validate_string_keys(state_dict, "adapter state")

        expected_keys = {"format", "version", "in_features", "out_features", "rank", "alpha", "lora_A", "lora_B"}
        actual_keys = set(state_dict.keys())

        if strict:
            missing_keys = expected_keys - actual_keys
            if missing_keys:
                raise ValueError(f"Missing required keys in adapter_state_dict: {sorted(missing_keys)}")
            unexpected_keys = actual_keys - expected_keys
            if unexpected_keys:
                raise ValueError(f"Unexpected keys in adapter_state_dict: {sorted(unexpected_keys)}")

        if "format" in state_dict:
            if not isinstance(state_dict["format"], str) or state_dict["format"] != "termux-train-lora-adapter":
                raise ValueError(f"Unsupported adapter format: {state_dict['format']!r}")
        elif strict:
            raise ValueError("Missing required key 'format'")

        if "version" in state_dict:
            if not isinstance(state_dict["version"], str) or state_dict["version"] != "1.0":
                raise ValueError(f"Unsupported adapter version: {state_dict['version']!r}")
        elif strict:
            raise ValueError("Missing required key 'version'")

        if "in_features" in state_dict:
            _validate_positive_int_metadata(state_dict["in_features"], self.in_features, "in_features")
        elif strict:
            raise ValueError("Missing required key 'in_features'")

        if "out_features" in state_dict:
            _validate_positive_int_metadata(state_dict["out_features"], self.out_features, "out_features")
        elif strict:
            raise ValueError("Missing required key 'out_features'")

        if "rank" in state_dict:
            _validate_positive_int_metadata(state_dict["rank"], self.rank, "rank")
        elif strict:
            raise ValueError("Missing required key 'rank'")

        if "alpha" in state_dict:
            _validate_alpha_metadata(state_dict["alpha"], self.alpha)
        elif strict:
            raise ValueError("Missing required key 'alpha'")

        # Pre-validate and stage pending buffers
        pending_A_data = None
        pending_B_data = None

        if "lora_A" in state_dict:
            _validate_2d_matrix_data(state_dict["lora_A"], (self.in_features, self.rank), "lora_A")
            pending_A_data = self.lora_A.backend.from_data(state_dict["lora_A"])
        elif strict:
            raise ValueError("Missing 'lora_A' in adapter_state_dict")

        if "lora_B" in state_dict:
            _validate_2d_matrix_data(state_dict["lora_B"], (self.rank, self.out_features), "lora_B")
            pending_B_data = self.lora_B.backend.from_data(state_dict["lora_B"])
        elif strict:
            raise ValueError("Missing 'lora_B' in adapter_state_dict")

        # Snapshot existing native buffers before commit
        snapshot_A = self.lora_A.backend.from_data(copy.deepcopy(self.lora_A.tolist()))
        snapshot_B = self.lora_B.backend.from_data(copy.deepcopy(self.lora_B.tolist()))

        # Commit with rollback guarantee
        try:
            if pending_A_data is not None:
                self.lora_A._replace_data(pending_A_data, bump_version=True)
            if pending_B_data is not None:
                self.lora_B._replace_data(pending_B_data, bump_version=True)
        except Exception as commit_err:
            rollback_errors = []
            try:
                self.lora_A._replace_data(snapshot_A, bump_version=True)
            except Exception as r_err_a:
                rollback_errors.append(f"lora_A rollback failed: {r_err_a}")
            try:
                self.lora_B._replace_data(snapshot_B, bump_version=True)
            except Exception as r_err_b:
                rollback_errors.append(f"lora_B rollback failed: {r_err_b}")

            if rollback_errors:
                raise RuntimeError(
                    f"Commit failed ({commit_err}) AND atomic rollback failed: {'; '.join(rollback_errors)}"
                ) from commit_err
            raise commit_err

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward computation:
          y = base(x) + ((x @ lora_A) @ lora_B) * scaling (when unmerged)
          y = base(x) (when merged, adapter delta is fused into base weight)
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


def _collect_lora_layers(module: Module) -> List[LoRALinear]:
    """Recursively collects all LoRALinear layers within module hierarchy in deterministic order with deduplication."""
    if isinstance(module, LoRALinear):
        return [module]

    layers: List[LoRALinear] = []
    visited_ids = set()

    def _traverse(m: Module):
        if id(m) in visited_ids:
            return
        visited_ids.add(id(m))

        if isinstance(m, LoRALinear):
            layers.append(m)
            return

        for sub_m in m._modules.values():
            if sub_m is not None:
                _traverse(sub_m)

    _traverse(module)
    return layers


def merge_lora_adapters(module: Module) -> None:
    """
    모듈 계층 내의 모든 LoRALinear 레이어를 재귀적으로 탐색하여 일괄 merge를 수행합니다.
    공유 모듈(shared module)은 중복 merge되지 않도록 한 번만 처리합니다.
    어느 한 레이어라도 이미 merged 상태이거나 unmerged 상태에서 snapshot이 존재하는 경우, 또는 검증/계산에 실패하면
    어떤 레이어도 변경하지 않고 전체 호출을 거부합니다.
    커밋 도중 예외가 발생할 경우 모든 레이어를 호출 전 상태로 원자적 롤백합니다.
    LoRALinear가 없는 빈 모듈에 대해서는 안전하게 no-op으로 종료합니다.
    """
    layers = _collect_lora_layers(module)
    if not layers:
        return

    # Check for mixed / invalid lifecycle state
    for layer in layers:
        if layer.merged:
            raise RuntimeError("One or more LoRALinear layers are already merged")
        if layer._base_weight_snapshot is not None:
            raise RuntimeError("One or more LoRALinear layers have an unexpected base weight snapshot while unmerged")

    # Phase 1: Preflight & Stage for all layers
    staged: List[Tuple[LoRALinear, Any, Any, Any, Any, bool]] = []
    for layer in layers:
        if (
            isinstance(layer.scaling, bool)
            or not isinstance(layer.scaling, (int, float))
            or not math.isfinite(float(layer.scaling))
            or float(layer.scaling) <= 0.0
        ):
            raise ValueError(f"scaling factor must be a finite positive number in layer, got {layer.scaling}")

        b = layer.base.weight.backend
        if layer.lora_A.backend.name != b.name or layer.lora_B.backend.name != b.name:
            raise RuntimeError("Cross-backend LoRALinear parameters are not supported for merge")

        _assert_param_finite(layer.base.weight, "base.weight")
        _assert_param_finite(layer.lora_A, "lora_A")
        _assert_param_finite(layer.lora_B, "lora_B")

        if layer.lora_A.shape != (layer.in_features, layer.rank):
            raise ValueError(f"lora_A shape mismatch: expected {(layer.in_features, layer.rank)}, got {layer.lora_A.shape}")
        if layer.lora_B.shape != (layer.rank, layer.out_features):
            raise ValueError(f"lora_B shape mismatch: expected {(layer.rank, layer.out_features)}, got {layer.lora_B.shape}")
        if layer.base.weight.shape != (layer.in_features, layer.out_features):
            raise ValueError(f"base.weight shape mismatch: expected {(layer.in_features, layer.out_features)}, got {layer.base.weight.shape}")

        expected_shape = layer.base.weight.shape

        delta = b.matmul(layer.lora_A._data, layer.lora_B._data)
        if b.get_shape(delta) != expected_shape:
            raise RuntimeError(f"Computed delta shape mismatch: expected {expected_shape}, got {b.get_shape(delta)}")

        delta = b.mul(delta, float(layer.scaling))
        merged_weight = b.add(layer.base.weight._data, delta)
        if b.get_shape(merged_weight) != expected_shape:
            raise RuntimeError(f"Computed merged_weight shape mismatch: expected {expected_shape}, got {b.get_shape(merged_weight)}")

        flat_merged = b.to_flat_list(merged_weight)
        for v in flat_merged:
            if not math.isfinite(v):
                raise ValueError(f"Non-finite value encountered in merged weight: {v}")

        snapshot = b.from_data(copy.deepcopy(layer.base.weight.tolist()))

        old_base = layer.base.weight._data
        old_snap = layer._base_weight_snapshot
        old_m = layer._merged

        staged.append((layer, merged_weight, snapshot, old_base, old_snap, old_m))

    # Phase 2: Transactional commit across all layers
    try:
        for layer, merged_weight, snapshot, _, _, _ in staged:
            layer.base.weight._replace_data(merged_weight, bump_version=True)
            layer._base_weight_snapshot = snapshot
            layer._merged = True
    except Exception as commit_err:
        rollback_errors = []
        for layer, _, _, old_base, old_snap, old_m in staged:
            try:
                layer.base.weight._replace_data(old_base, bump_version=True)
            except Exception as r_err:
                rollback_errors.append(f"{layer}.base.weight rollback failed: {r_err}")
            try:
                layer._base_weight_snapshot = old_snap
            except Exception as r_err:
                rollback_errors.append(f"{layer} snapshot rollback failed: {r_err}")
            try:
                layer._merged = old_m
            except Exception as r_err:
                rollback_errors.append(f"{layer} merged flag rollback failed: {r_err}")

        if rollback_errors:
            raise RuntimeError(
                f"Multi-layer merge failed ({commit_err}) AND rollback failed: {'; '.join(rollback_errors)}"
            ) from commit_err
        raise commit_err


def unmerge_lora_adapters(module: Module) -> None:
    """
    모듈 계층 내의 모든 LoRALinear 레이어를 재귀적으로 탐색하여 일괄 unmerge를 수행합니다.
    공유 모듈은 한 번만 처리합니다.
    어느 한 레이어라도 unmerged 상태이거나 snapshot이 누락/손상된 경우 어떤 레이어도 변경하지 않고 전체 호출을 거부합니다.
    커밋 도중 예외가 발생할 경우 모든 레이어의 merged 상태를 원자적으로 복원합니다.
    LoRALinear가 없는 빈 모듈에 대해서는 안전하게 no-op으로 종료합니다.
    """
    layers = _collect_lora_layers(module)
    if not layers:
        return

    # Check for mixed / invalid lifecycle state
    for layer in layers:
        if not layer.merged:
            raise RuntimeError("One or more LoRALinear layers are not merged")
        if layer._base_weight_snapshot is None:
            raise RuntimeError("Missing base weight snapshot for unmerge in one or more layers")

    # Phase 1: Preflight & Stage for all layers
    staged: List[Tuple[LoRALinear, Any, Any, Any, bool]] = []
    for layer in layers:
        b = layer.base.weight.backend
        snap_shape = b.get_shape(layer._base_weight_snapshot)
        if snap_shape != layer.base.weight.shape:
            raise RuntimeError(
                f"Base weight snapshot shape mismatch in layer: expected {layer.base.weight.shape}, got {snap_shape}"
            )

        flat_snap = b.to_flat_list(layer._base_weight_snapshot)
        for v in flat_snap:
            if not math.isfinite(v):
                raise ValueError(f"Non-finite value found in base weight snapshot: {v}")

        snap_to_restore = layer._base_weight_snapshot
        old_base = layer.base.weight._data
        old_snap = layer._base_weight_snapshot
        old_m = layer._merged

        staged.append((layer, snap_to_restore, old_base, old_snap, old_m))

    # Phase 2: Transactional commit across all layers
    try:
        for layer, snap_to_restore, _, _, _ in staged:
            layer.base.weight._replace_data(snap_to_restore, bump_version=True)
            layer._base_weight_snapshot = None
            layer._merged = False
    except Exception as commit_err:
        rollback_errors = []
        for layer, _, old_base, old_snap, old_m in staged:
            try:
                layer.base.weight._replace_data(old_base, bump_version=True)
            except Exception as r_err:
                rollback_errors.append(f"{layer}.base.weight rollback failed: {r_err}")
            try:
                layer._base_weight_snapshot = old_snap
            except Exception as r_err:
                rollback_errors.append(f"{layer} snapshot rollback failed: {r_err}")
            try:
                layer._merged = old_m
            except Exception as r_err:
                rollback_errors.append(f"{layer} merged flag rollback failed: {r_err}")

        if rollback_errors:
            raise RuntimeError(
                f"Multi-layer unmerge failed ({commit_err}) AND rollback failed: {'; '.join(rollback_errors)}"
            ) from commit_err
        raise commit_err


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


def adapter_state_dict(module: Module) -> Dict[str, Any]:
    """
    Recursively exports adapter states from all LoRALinear layers within the given module hierarchy.
    Returns a container mapping deterministic hierarchical prefixes to individual layer adapter state dictionaries.
    """
    if isinstance(module, LoRALinear):
        return module.adapter_state_dict()

    adapters: Dict[str, Any] = {}
    visited_modules = set()

    def _traverse(m: Module, curr_prefix: str):
        if id(m) in visited_modules:
            return
        visited_modules.add(id(m))

        if isinstance(m, LoRALinear):
            key = curr_prefix if curr_prefix else "layer"
            adapters[key] = m.adapter_state_dict()
            return

        for sub_name, sub_m in m._modules.items():
            if sub_m is not None:
                sub_prefix = f"{curr_prefix}.{sub_name}" if curr_prefix else sub_name
                _traverse(sub_m, sub_prefix)

    _traverse(module, "")

    return {
        "format": "termux-train-lora-model-adapter",
        "version": "1.0",
        "adapters": adapters,
    }


def _validate_model_adapter_container(state_dict: Dict[str, Any], strict: bool) -> Dict[str, Any]:
    """
    Validates the outer schema of a model adapter container and returns the inner adapters dict.
    Strictly checks format, version, string keys, and adapter entry types.
    """
    _validate_string_keys(state_dict, "model adapter state")

    expected_keys = {"format", "version", "adapters"}
    actual_keys = set(state_dict.keys())

    if strict:
        missing = expected_keys - actual_keys
        if missing:
            raise ValueError(f"Missing model adapter keys: {sorted(missing)}")
        unexpected = actual_keys - expected_keys
        if unexpected:
            raise ValueError(f"Unexpected model adapter keys: {sorted(unexpected)}")

    if state_dict.get("format") != "termux-train-lora-model-adapter":
        raise ValueError(f"Unsupported model adapter format: {state_dict.get('format')!r}")

    if state_dict.get("version") != "1.0":
        raise ValueError(f"Unsupported model adapter version: {state_dict.get('version')!r}")

    adapters = state_dict.get("adapters")
    if not isinstance(adapters, dict):
        raise TypeError(f"'adapters' must be a dict, got {type(adapters).__name__}")

    _validate_string_keys(adapters, "adapter path")

    for k, v in adapters.items():
        if not isinstance(v, dict):
            raise TypeError(f"Adapter state entry for '{k}' must be a dict, got {type(v).__name__}")

    return adapters


def load_adapter_state_dict(module: Module, state_dict: Dict[str, Any], strict: bool = True) -> None:
    """
    Recursively loads adapter states into all LoRALinear layers within the given module hierarchy.
    Enforces two-phase atomic validation across all layers in the model before committing,
    and guarantees 100% rollback across all layers if any commit step fails.
    """
    if not isinstance(state_dict, dict):
        raise TypeError(f"state_dict must be a dict, got {type(state_dict).__name__}")

    _validate_string_keys(state_dict, "adapter state")

    # Case 1: Single LoRALinear layer
    if isinstance(module, LoRALinear):
        if state_dict.get("format") == "termux-train-lora-model-adapter":
            adapters = _validate_model_adapter_container(state_dict, strict=strict)
            if len(adapters) != 1:
                raise ValueError(
                    f"Model adapter container has {len(adapters)} layers, expected 1 for single LoRALinear"
                )
            single_state = next(iter(adapters.values()))
            module.load_adapter_state_dict(single_state, strict=strict)
            return
        module.load_adapter_state_dict(state_dict, strict=strict)
        return

    # Case 2: Container module (e.g. Sequential or custom Module)
    layers: Dict[str, LoRALinear] = {}
    visited_modules = set()

    def _collect(m: Module, curr_prefix: str):
        if id(m) in visited_modules:
            return
        visited_modules.add(id(m))

        if isinstance(m, LoRALinear):
            key = curr_prefix if curr_prefix else "layer"
            layers[key] = m
            return

        for sub_name, sub_m in m._modules.items():
            if sub_m is not None:
                sub_prefix = f"{curr_prefix}.{sub_name}" if curr_prefix else sub_name
                _collect(sub_m, sub_prefix)

    _collect(module, "")

    # Extract adapter dictionary
    adapters_data: Dict[str, Any] = {}
    if state_dict.get("format") == "termux-train-lora-model-adapter":
        adapters_data = _validate_model_adapter_container(state_dict, strict=strict)
    elif all(isinstance(k, str) and k in layers for k in state_dict.keys()):
        adapters_data = state_dict
    else:
        # Check flat parameter dictionary: e.g. "0.lora_A", "0.lora_B"
        flat_matching = False
        reconstructed: Dict[str, Dict[str, Any]] = {}
        for l_key, l_obj in layers.items():
            key_a = f"{l_key}.lora_A"
            key_b = f"{l_key}.lora_B"
            if key_a in state_dict or key_b in state_dict:
                flat_matching = True
                layer_d = {
                    "format": "termux-train-lora-adapter",
                    "version": "1.0",
                    "in_features": l_obj.in_features,
                    "out_features": l_obj.out_features,
                    "rank": l_obj.rank,
                    "alpha": l_obj.alpha,
                }
                if key_a in state_dict:
                    layer_d["lora_A"] = state_dict[key_a]
                if key_b in state_dict:
                    layer_d["lora_B"] = state_dict[key_b]
                reconstructed[l_key] = layer_d
        if flat_matching:
            adapters_data = reconstructed
        else:
            adapters_data = state_dict.get("adapters", state_dict)
            if not isinstance(adapters_data, dict):
                raise TypeError(f"'adapters' must be a dict, got {type(adapters_data).__name__}")
            _validate_string_keys(adapters_data, "adapter path")

    if strict:
        expected_keys = set(layers.keys())
        actual_keys = set(adapters_data.keys())
        missing_keys = expected_keys - actual_keys
        if missing_keys:
            raise ValueError(f"Missing adapter keys in model state_dict: {sorted(missing_keys)}")
        unexpected_keys = actual_keys - expected_keys
        if unexpected_keys:
            raise ValueError(f"Unexpected adapter keys in model state_dict: {sorted(unexpected_keys)}")

    # Phase 1: validate and stage pending buffers for all layers
    staged: Dict[str, Tuple[Optional[Any], Optional[Any]]] = {}
    for l_key, l_obj in layers.items():
        if l_key not in adapters_data:
            if strict:
                raise ValueError(f"Missing layer '{l_key}' in state_dict")
            continue
        l_state = adapters_data[l_key]
        if not isinstance(l_state, dict):
            raise TypeError(f"State for layer '{l_key}' must be a dict, got {type(l_state).__name__}")

        _validate_string_keys(l_state, f"layer '{l_key}' adapter state")

        if l_obj.merged:
            raise RuntimeError(f"Cannot load adapter state into merged layer '{l_key}'")

        if strict:
            exp_l_keys = {"format", "version", "in_features", "out_features", "rank", "alpha", "lora_A", "lora_B"}
            act_l_keys = set(l_state.keys())
            if exp_l_keys - act_l_keys:
                raise ValueError(f"Missing keys in layer '{l_key}': {sorted(exp_l_keys - act_l_keys)}")
            if act_l_keys - exp_l_keys:
                raise ValueError(f"Unexpected keys in layer '{l_key}': {sorted(act_l_keys - exp_l_keys)}")

        if "format" in l_state:
            if not isinstance(l_state["format"], str) or l_state["format"] != "termux-train-lora-adapter":
                raise ValueError(f"Unsupported adapter format in layer '{l_key}': {l_state['format']!r}")
        elif strict:
            raise ValueError(f"Missing required key 'format' in layer '{l_key}'")

        if "version" in l_state:
            if not isinstance(l_state["version"], str) or l_state["version"] != "1.0":
                raise ValueError(f"Unsupported adapter version in layer '{l_key}': {l_state['version']!r}")
        elif strict:
            raise ValueError(f"Missing required key 'version' in layer '{l_key}'")

        if "in_features" in l_state:
            _validate_positive_int_metadata(l_state["in_features"], l_obj.in_features, f"{l_key}.in_features")
        elif strict:
            raise ValueError(f"Missing required key 'in_features' in layer '{l_key}'")

        if "out_features" in l_state:
            _validate_positive_int_metadata(l_state["out_features"], l_obj.out_features, f"{l_key}.out_features")
        elif strict:
            raise ValueError(f"Missing required key 'out_features' in layer '{l_key}'")

        if "rank" in l_state:
            _validate_positive_int_metadata(l_state["rank"], l_obj.rank, f"{l_key}.rank")
        elif strict:
            raise ValueError(f"Missing required key 'rank' in layer '{l_key}'")

        if "alpha" in l_state:
            _validate_alpha_metadata(l_state["alpha"], l_obj.alpha)
        elif strict:
            raise ValueError(f"Missing required key 'alpha' in layer '{l_key}'")

        pending_A = None
        pending_B = None
        if "lora_A" in l_state:
            _validate_2d_matrix_data(l_state["lora_A"], (l_obj.in_features, l_obj.rank), f"{l_key}.lora_A")
            pending_A = l_obj.lora_A.backend.from_data(l_state["lora_A"])
        elif strict:
            raise ValueError(f"Missing 'lora_A' in layer '{l_key}'")

        if "lora_B" in l_state:
            _validate_2d_matrix_data(l_state["lora_B"], (l_obj.rank, l_obj.out_features), f"{l_key}.lora_B")
            pending_B = l_obj.lora_B.backend.from_data(l_state["lora_B"])
        elif strict:
            raise ValueError(f"Missing 'lora_B' in layer '{l_key}'")

        staged[l_key] = (pending_A, pending_B)

    # Phase 2: Create snapshots for all staged layers before commit
    snapshots: List[Tuple[LoRALinear, Any, Any]] = []
    for l_key in staged:
        l_obj = layers[l_key]
        snap_A = l_obj.lora_A.backend.from_data(copy.deepcopy(l_obj.lora_A.tolist()))
        snap_B = l_obj.lora_B.backend.from_data(copy.deepcopy(l_obj.lora_B.tolist()))
        snapshots.append((l_obj, snap_A, snap_B))

    # Phase 3: Commit all staged buffers atomically with multi-layer rollback
    try:
        for l_key, (pending_A, pending_B) in staged.items():
            l_obj = layers[l_key]
            if pending_A is not None:
                l_obj.lora_A._replace_data(pending_A, bump_version=True)
            if pending_B is not None:
                l_obj.lora_B._replace_data(pending_B, bump_version=True)
    except Exception as commit_err:
        rollback_errors = []
        for l_obj, snap_A, snap_B in snapshots:
            try:
                l_obj.lora_A._replace_data(snap_A, bump_version=True)
            except Exception as r_err_a:
                rollback_errors.append(f"{l_obj}.lora_A rollback failed: {r_err_a}")
            try:
                l_obj.lora_B._replace_data(snap_B, bump_version=True)
            except Exception as r_err_b:
                rollback_errors.append(f"{l_obj}.lora_B rollback failed: {r_err_b}")

        if rollback_errors:
            raise RuntimeError(
                f"Multi-layer commit failed ({commit_err}) AND atomic rollback failed: {'; '.join(rollback_errors)}"
            ) from commit_err
        raise commit_err
