"""
termux_train.nn.lora
====================
Low-Rank Adaptation (LoRA) Layer for Parameter-Efficient On-Device Fine-Tuning.
Freezes pre-trained base Linear weights and learns decomposed low-rank matrices
lora_A (in_features, rank) and lora_B (rank, out_features) with scaling factor alpha / rank.
Supports atomic, crash-resilient, cross-backend adapter-only state serialization and restoration.
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
                self.lora_A._data = pending_A_data
            if pending_B_data is not None:
                self.lora_B._data = pending_B_data
        except Exception as commit_err:
            rollback_errors = []
            try:
                self.lora_A._data = snapshot_A
            except Exception as r_err_a:
                rollback_errors.append(f"lora_A rollback failed: {r_err_a}")
            try:
                self.lora_B._data = snapshot_B
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
        if state_dict.get("format") == "termux-train-lora-model-adapter" and "adapters" in state_dict:
            adapters = state_dict["adapters"]
            if not isinstance(adapters, dict):
                raise TypeError(f"'adapters' must be a dict, got {type(adapters).__name__}")
            _validate_string_keys(adapters, "adapter path")
            if len(adapters) == 1:
                single_key = next(iter(adapters))
                module.load_adapter_state_dict(adapters[single_key], strict=strict)
                return
            raise ValueError(f"Model adapter container has {len(adapters)} layers, expected 1 for single LoRALinear")
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
    if state_dict.get("format") == "termux-train-lora-model-adapter" and "adapters" in state_dict:
        if state_dict.get("version") != "1.0":
            raise ValueError(f"Unsupported model adapter version: {state_dict.get('version')!r}")
        adapters_data = state_dict["adapters"]
        if not isinstance(adapters_data, dict):
            raise TypeError(f"'adapters' must be a dict, got {type(adapters_data).__name__}")
        _validate_string_keys(adapters_data, "adapter path")
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
                l_obj.lora_A._data = pending_A
            if pending_B is not None:
                l_obj.lora_B._data = pending_B
    except Exception as commit_err:
        rollback_errors = []
        for l_obj, snap_A, snap_B in snapshots:
            try:
                l_obj.lora_A._data = snap_A
            except Exception as r_err_a:
                rollback_errors.append(f"{l_obj}.lora_A rollback failed: {r_err_a}")
            try:
                l_obj.lora_B._data = snap_B
            except Exception as r_err_b:
                rollback_errors.append(f"{l_obj}.lora_B rollback failed: {r_err_b}")

        if rollback_errors:
            raise RuntimeError(
                f"Multi-layer commit failed ({commit_err}) AND atomic rollback failed: {'; '.join(rollback_errors)}"
            ) from commit_err
        raise commit_err
