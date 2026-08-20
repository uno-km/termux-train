"""
termux_train.runtime.checkpoint
===============================
Safe, Atomic, Crash-Resilient Checkpoint Serialization and Deserialization Engine.
Guarantees integrity via SHA256 checksums, atomic file writes (.tmp -> fsync -> os.replace),
and full rollback of model and optimizer states on load failure.
Supports standard full-model checkpoints and dedicated on-device LoRA adapter checkpoints.
"""

import copy
import hashlib
import hmac
import json
import math
import os
import time
from typing import Optional, Dict, Any, List
from ..nn.module import Module
from ..nn.lora import (
    LoRALinear,
    adapter_state_dict,
    load_adapter_state_dict,
    adapter_parameters,
)
from ..optim.optimizer import Optimizer

class CheckpointError(Exception):
    """Base exception for checkpoint operations."""
    pass

class CheckpointIntegrityError(CheckpointError):
    """Raised when checkpoint file is corrupted or SHA256 checksum mismatches."""
    pass

class CheckpointSchemaError(CheckpointError):
    """Raised when checkpoint structure or version is invalid."""
    pass

class CheckpointRollbackError(CheckpointError):
    """Raised when checkpoint loading failed AND rollback of previous state also failed."""
    pass


# =============================================================================
# Shared Private Helpers for Checkpointing Engine
# =============================================================================

def _canonical_json_bytes(payload: dict) -> bytes:
    """Serializes payload to canonical, deterministic UTF-8 JSON bytes with no NaN/Inf."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _compute_payload_checksum(payload: dict) -> str:
    """Computes hexadecimal SHA-256 digest over canonical payload bytes."""
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _validate_checksum_format(checksum: Any, context: str = "checkpoint") -> None:
    """Validates that checksum is a 64-character lowercase hexadecimal string."""
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or not all(c in "0123456789abcdef" for c in checksum)
    ):
        raise CheckpointIntegrityError(f"Invalid SHA256 checksum format in {context}: {checksum!r}")


def _validate_counter_scalar(val: Any, name: str, context: str = "checkpoint") -> None:
    """Validates non-negative integer scalar counter (epoch, global_step)."""
    if isinstance(val, bool) or not isinstance(val, int) or val < 0:
        raise CheckpointSchemaError(f"Invalid {name} in {context}: expected non-negative integer, got {val!r}")


def _validate_timestamp_scalar(val: Any, context: str = "checkpoint") -> None:
    """Validates finite non-negative numeric timestamp."""
    if (
        isinstance(val, bool)
        or not isinstance(val, (int, float))
        or not math.isfinite(float(val))
        or float(val) < 0.0
    ):
        raise CheckpointSchemaError(f"Invalid timestamp in {context}: expected non-negative finite number, got {val!r}")


def _validate_recursive_json_dict(val: Any, name: str = "extra") -> None:
    """Recursively validates that dictionary contains only string keys and JSON-compatible values."""
    if not isinstance(val, dict):
        raise TypeError(f"{name} must be a dict, got {type(val).__name__}")

    def _check_node(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if not isinstance(k, str):
                    raise TypeError(f"{name} contains non-string key {k!r} at {path or 'root'}")
                _check_node(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, (list, tuple)):
            for idx, item in enumerate(node):
                _check_node(item, f"{path}[{idx}]")
        elif node is None or isinstance(node, (str, bool)):
            pass
        elif isinstance(node, (int, float)):
            if not math.isfinite(node):
                raise ValueError(f"{name} contains non-finite number {node} at {path}")
        else:
            raise TypeError(f"{name} contains non-JSON-serializable type {type(node).__name__} at {path}")

    _check_node(val, "")


def _atomic_write_checkpoint(abs_path: str, container_json: str) -> None:
    """Atomically writes container JSON via temporary file, fsync, and atomic rename with EXDEV fallback."""
    import shutil
    parent_dir = os.path.dirname(abs_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    tmp_path = f"{abs_path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(container_json)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp_path, abs_path)
        except OSError as os_err:
            if getattr(os_err, "errno", None) == 18:  # EXDEV: Invalid cross-device link on Android sdcard
                shutil.move(tmp_path, abs_path)
            else:
                raise
    except Exception as e:
        cleanup_err = None
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as c_err:
                cleanup_err = c_err
        err_msg = f"Failed to atomically save checkpoint to {abs_path}: {e}"
        if cleanup_err is not None:
            err_msg += f" (tmp cleanup failed: {cleanup_err})"
        raise CheckpointError(err_msg) from e


def _collect_all_lora_layers(module: Module) -> List[LoRALinear]:
    """Collects all unique LoRALinear layers within module hierarchy."""
    if isinstance(module, LoRALinear):
        return [module]
    layers = []
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


def _validate_model_for_lora_checkpoint(model: Module, action: str) -> None:
    """Ensures model is a Module, contains at least one LoRA layer, and all adapters are unmerged."""
    if not isinstance(model, Module):
        raise TypeError(f"model must be a Module instance, got {type(model).__name__}")
    lora_layers = _collect_all_lora_layers(model)
    if not lora_layers:
        raise ValueError(f"LoRA checkpoint requires at least one LoRALinear layer in model for {action}")
    for layer in lora_layers:
        if layer.merged or layer._base_weight_snapshot is not None:
            raise RuntimeError(f"LoRA checkpoints require all adapters to be unmerged for {action}")


def _validate_optimizer_for_lora_checkpoint(model: Module, optimizer: Optimizer, action: str) -> None:
    """Ensures optimizer exclusively tracks model adapter parameters in deterministic order."""
    if not isinstance(optimizer, Optimizer):
        raise TypeError(f"optimizer must be an Optimizer instance, got {type(optimizer).__name__}")
    expected_params = adapter_parameters(model)
    if len(optimizer.params) != len(expected_params) or any(
        act is not exp for act, exp in zip(optimizer.params, expected_params)
    ):
        raise ValueError(
            f"Optimizer parameters must exactly match model adapter parameters in deterministic order for {action}"
        )


# =============================================================================
# Generic Full-Model Checkpointing APIs
# =============================================================================

def save_checkpoint(
    path: str,
    model: Optional[Module] = None,
    optimizer: Optional[Optimizer] = None,
    epoch: int = 0,
    global_step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Atomically saves training state to a JSON-based checkpoint file.

    Guarantees:
      - Writes to temporary file (<path>.tmp) first
      - Flushes buffer to OS disk via fsync
      - Atomically renames to target <path> via os.replace
      - Preserves existing checkpoint file if save fails at any point

    Args:
        path: Target file path to write checkpoint.
        model: Optional nn.Module instance to serialize.
        optimizer: Optional Optimizer instance to serialize.
        epoch: Current training epoch (must be non-negative integer).
        global_step: Current global optimization step (must be non-negative integer).
        extra: Optional arbitrary JSON-serializable metadata dict.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"epoch must be a non-negative integer, got {epoch}")

    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise ValueError(f"global_step must be a non-negative integer, got {global_step}")

    if model is not None and not isinstance(model, Module):
        raise TypeError(f"model must be a Module instance, got {type(model).__name__}")

    if optimizer is not None and not isinstance(optimizer, Optimizer):
        raise TypeError(f"optimizer must be an Optimizer instance, got {type(optimizer).__name__}")

    model_state = model.state_dict() if model is not None else None
    optimizer_state = optimizer.state_dict() if optimizer is not None else None

    payload = {
        "format": "termux-train-checkpoint",
        "version": "1.0",
        "timestamp": time.time(),
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "extra": copy.deepcopy(extra) if extra is not None else {},
    }

    # Deterministic serialization for checksum calculation
    payload_json = json.dumps(payload, sort_keys=True, indent=2)
    checksum = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

    container = {
        "checksum": checksum,
        "payload": payload,
    }
    container_json = json.dumps(container, indent=2)

    abs_path = os.path.abspath(path)
    _atomic_write_checkpoint(abs_path, container_json)


def load_checkpoint(
    path: str,
    model: Optional[Module] = None,
    optimizer: Optional[Optimizer] = None,
) -> Dict[str, Any]:
    """
    Atomically loads and validates a checkpoint, restoring model and optimizer states.

    Guarantees:
      - Validates SHA256 integrity checksum before parsing states
      - Enforces full rollback: if model or optimizer state loading fails,
        both model and optimizer are restored to their exact pre-call state.

    Args:
        path: Path to checkpoint file.
        model: Optional nn.Module to load model_state into.
        optimizer: Optional Optimizer to load optimizer_state into.

    Returns:
        Dict containing checkpoint metadata:
          {"epoch": int, "global_step": int, "timestamp": float, "extra": dict}
    """
    if not isinstance(path, str) or not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            container = json.load(f)
    except Exception as e:
        raise CheckpointIntegrityError(f"Malformed or unreadable checkpoint JSON in {path}: {e}") from e

    if not isinstance(container, dict) or "checksum" not in container or "payload" not in container:
        raise CheckpointSchemaError(f"Invalid checkpoint container structure in {path}")

    expected_checksum = container["checksum"]
    payload = container["payload"]

    # Verify SHA256 integrity
    recomputed_payload_json = json.dumps(payload, sort_keys=True, indent=2)
    actual_checksum = hashlib.sha256(recomputed_payload_json.encode("utf-8")).hexdigest()

    if actual_checksum != expected_checksum:
        raise CheckpointIntegrityError(
            f"Checkpoint checksum mismatch in {path}: expected {expected_checksum}, calculated {actual_checksum}"
        )

    if not isinstance(payload, dict) or payload.get("format") != "termux-train-checkpoint":
        raise CheckpointSchemaError(f"Unsupported checkpoint format in {path}: {payload.get('format') if isinstance(payload, dict) else type(payload).__name__}")

    if payload.get("version") != "1.0":
        raise CheckpointSchemaError(f"Unsupported checkpoint version in {path}: {payload.get('version')}")

    epoch = payload.get("epoch", 0)
    global_step = payload.get("global_step", 0)

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise CheckpointSchemaError(f"Invalid epoch in checkpoint: {epoch}")

    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise CheckpointSchemaError(f"Invalid global_step in checkpoint: {global_step}")

    # Snapshot current states for atomic rollback guarantee
    orig_model_state = model.state_dict() if model is not None else None
    orig_optimizer_state = optimizer.state_dict() if optimizer is not None else None

    try:
        if model is not None:
            model_state = payload.get("model_state")
            if model_state is None:
                raise CheckpointSchemaError("Checkpoint does not contain 'model_state'")
            model.load_state_dict(model_state)

        if optimizer is not None:
            optimizer_state = payload.get("optimizer_state")
            if optimizer_state is None:
                raise CheckpointSchemaError("Checkpoint does not contain 'optimizer_state'")
            optimizer.load_state_dict(optimizer_state)
    except Exception as e:
        # Full rollback to preserve pre-call state
        rollback_errors = []
        if model is not None and orig_model_state is not None:
            try:
                model.load_state_dict(orig_model_state)
            except Exception as r_err:
                rollback_errors.append(f"model rollback failed: {r_err}")

        if optimizer is not None and orig_optimizer_state is not None:
            try:
                optimizer.load_state_dict(orig_optimizer_state)
            except Exception as r_err:
                rollback_errors.append(f"optimizer rollback failed: {r_err}")

        if rollback_errors:
            raise CheckpointRollbackError(
                f"Checkpoint load failed ({e}) AND atomic rollback also failed: {'; '.join(rollback_errors)}"
            ) from e

        raise e

    return {
        "epoch": epoch,
        "global_step": global_step,
        "timestamp": payload.get("timestamp"),
        "extra": payload.get("extra", {}),
    }


# =============================================================================
# Dedicated LoRA Adapter Checkpointing APIs
# =============================================================================

def save_lora_checkpoint(
    path: str,
    model: Module,
    optimizer: Optional[Optimizer] = None,
    epoch: int = 0,
    global_step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Atomically saves LoRA adapter parameters and adapter-only optimizer state to a JSON checkpoint.

    Guarantees:
      - Model is required: model cannot be None and must contain >= 1 LoRALinear.
      - Optimizer is optional: if None, optimizer_state is serialized as None.
      - Unmerged-only policy: Rejects models with merged LoRA layers or stale snapshots.
      - Base weight and bias exclusion: Only adapter factors (lora_A, lora_B) are serialized.
      - Optimizer validation: Ensures optimizer exclusively tracks adapter parameters in exact order.
      - Atomic write: Writes to <path>.tmp, performs fsync, and atomic os.replace.
      - SHA-256 integrity: Computes deterministic checksum over canonical payload JSON.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")

    if model is None:
        raise TypeError("model is required and cannot be None")

    if not isinstance(model, Module):
        raise TypeError(f"model must be a Module instance, got {type(model).__name__}")

    _validate_counter_scalar(epoch, "epoch", "save_lora_checkpoint")
    _validate_counter_scalar(global_step, "global_step", "save_lora_checkpoint")

    _validate_model_for_lora_checkpoint(model, "saving")

    if optimizer is not None:
        _validate_optimizer_for_lora_checkpoint(model, optimizer, "saving")
        optimizer_state = optimizer.state_dict()
    else:
        optimizer_state = None

    if extra is not None:
        _validate_recursive_json_dict(extra, "extra")
        extra_dict = copy.deepcopy(extra)
    else:
        extra_dict = {}

    adapter_state = adapter_state_dict(model)

    payload = {
        "format": "termux-train-lora-checkpoint",
        "version": "1.0",
        "timestamp": float(time.time()),
        "epoch": epoch,
        "global_step": global_step,
        "adapter_state": adapter_state,
        "optimizer_state": optimizer_state,
        "extra": extra_dict,
    }

    try:
        payload_bytes = _canonical_json_bytes(payload)
    except Exception as e:
        raise CheckpointError(f"Failed to serialize LoRA checkpoint payload to canonical JSON: {e}") from e

    checksum = hashlib.sha256(payload_bytes).hexdigest()

    container = {
        "checksum": checksum,
        "payload": payload,
    }

    try:
        container_json = json.dumps(container, indent=2, allow_nan=False)
    except Exception as e:
        raise CheckpointError(f"Failed to serialize LoRA checkpoint container to JSON: {e}") from e

    abs_path = os.path.abspath(path)
    _atomic_write_checkpoint(abs_path, container_json)


def load_lora_checkpoint(
    path: str,
    model: Optional[Module] = None,
    optimizer: Optional[Optimizer] = None,
) -> Dict[str, Any]:
    """
    Atomically loads and validates a LoRA adapter checkpoint, restoring adapter factors and optimizer state.

    API Combinations:
      - Combination A (model, optimizer): Restores adapter state and optimizer state atomically.
      - Combination B (model, optimizer=None): Restores adapter state only; skips optimizer state.
      - Combination C (model=None, optimizer=None): Metadata-only validation mode. Mutates no state.
      - Combination D (model=None, optimizer): Invalid. Raises ValueError.

    Guarantees:
      - SHA-256 integrity validation before touching model or optimizer.
      - Unmerged-only policy: Rejects target models with merged LoRA layers or stale snapshots.
      - Two-phase transaction: Restores both adapter parameters and optimizer atomically.
      - Full rollback on load failure: Preserves exact pre-call state if loading fails at any point.
    """
    if model is None and optimizer is not None:
        raise ValueError("A model is required when loading LoRA optimizer state")

    if not isinstance(path, str) or not os.path.exists(path):
        raise FileNotFoundError(f"LoRA checkpoint file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            container = json.load(f)
    except Exception as e:
        raise CheckpointIntegrityError(f"Malformed or unreadable LoRA checkpoint JSON in {path}: {e}") from e

    if not isinstance(container, dict):
        raise CheckpointSchemaError(f"Invalid LoRA checkpoint container structure in {path}: expected dict, got {type(container).__name__}")

    expected_container_keys = {"checksum", "payload"}
    actual_container_keys = set(container.keys())
    if actual_container_keys != expected_container_keys:
        raise CheckpointSchemaError(f"Unexpected keys in LoRA checkpoint container: {sorted(actual_container_keys)}")

    expected_checksum = container["checksum"]
    _validate_checksum_format(expected_checksum, path)

    payload = container["payload"]
    if not isinstance(payload, dict):
        raise CheckpointSchemaError(f"Invalid payload in LoRA checkpoint {path}: expected dict, got {type(payload).__name__}")

    # Canonical JSON serialization for checksum verification
    try:
        recomputed_payload_bytes = _canonical_json_bytes(payload)
    except Exception as e:
        raise CheckpointIntegrityError(f"Failed to canonicalize payload JSON for checksum in {path}: {e}") from e

    actual_checksum = hashlib.sha256(recomputed_payload_bytes).hexdigest()
    if not hmac.compare_digest(actual_checksum, expected_checksum):
        raise CheckpointIntegrityError(
            f"Checkpoint checksum mismatch in {path}: expected {expected_checksum}, calculated {actual_checksum}"
        )

    expected_payload_keys = {"format", "version", "timestamp", "epoch", "global_step", "adapter_state", "optimizer_state", "extra"}
    actual_payload_keys = set(payload.keys())
    if actual_payload_keys != expected_payload_keys:
        raise CheckpointSchemaError(f"Invalid payload keys in LoRA checkpoint {path}: {sorted(actual_payload_keys)}")

    if payload.get("format") != "termux-train-lora-checkpoint":
        raise CheckpointSchemaError(f"Unsupported LoRA checkpoint format in {path}: {payload.get('format')}")

    if payload.get("version") != "1.0":
        raise CheckpointSchemaError(f"Unsupported LoRA checkpoint version in {path}: {payload.get('version')}")

    _validate_timestamp_scalar(payload.get("timestamp"), path)
    _validate_counter_scalar(payload.get("epoch"), "epoch", path)
    _validate_counter_scalar(payload.get("global_step"), "global_step", path)

    adapter_state = payload.get("adapter_state")
    if not isinstance(adapter_state, dict):
        raise CheckpointSchemaError(f"Invalid adapter_state in LoRA checkpoint: expected dict, got {type(adapter_state).__name__}")

    optimizer_state = payload.get("optimizer_state")
    if optimizer_state is not None and not isinstance(optimizer_state, dict):
        raise CheckpointSchemaError(f"Invalid optimizer_state in LoRA checkpoint: expected dict or None, got {type(optimizer_state).__name__}")

    extra = payload.get("extra")
    if not isinstance(extra, dict):
        raise CheckpointSchemaError(f"Invalid extra in LoRA checkpoint: expected dict, got {type(extra).__name__}")

    if model is not None:
        _validate_model_for_lora_checkpoint(model, "loading")

    if optimizer is not None:
        _validate_optimizer_for_lora_checkpoint(model, optimizer, "loading")
        if optimizer_state is None:
            raise CheckpointSchemaError("LoRA checkpoint does not contain 'optimizer_state'")

    # Metadata-only mode
    if model is None and optimizer is None:
        return {
            "epoch": payload["epoch"],
            "global_step": payload["global_step"],
            "timestamp": payload["timestamp"],
            "extra": copy.deepcopy(extra),
        }

    # Snapshot current states before loading for atomic rollback
    orig_adapter_state = adapter_state_dict(model) if model is not None else None
    orig_optimizer_state = optimizer.state_dict() if optimizer is not None else None

    try:
        if model is not None:
            load_adapter_state_dict(model, adapter_state, strict=True)

        if optimizer is not None:
            optimizer.load_state_dict(optimizer_state)
    except Exception as load_err:
        rollback_errors = []
        if model is not None and orig_adapter_state is not None:
            try:
                load_adapter_state_dict(model, orig_adapter_state, strict=True)
            except Exception as r_err:
                rollback_errors.append(f"adapter rollback failed: {r_err}")
        if optimizer is not None and orig_optimizer_state is not None:
            try:
                optimizer.load_state_dict(orig_optimizer_state)
            except Exception as r_err:
                rollback_errors.append(f"optimizer rollback failed: {r_err}")

        if rollback_errors:
            raise CheckpointRollbackError(
                f"LoRA checkpoint load failed ({load_err}) AND atomic rollback also failed: {'; '.join(rollback_errors)}"
            ) from load_err
        raise load_err

    return {
        "epoch": payload["epoch"],
        "global_step": payload["global_step"],
        "timestamp": payload["timestamp"],
        "extra": copy.deepcopy(extra),
    }
