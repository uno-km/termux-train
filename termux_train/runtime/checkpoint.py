"""
termux_train.runtime.checkpoint
===============================
Safe, Atomic, Crash-Resilient Checkpoint Serialization and Deserialization Engine.
Guarantees integrity via SHA256 checksums, atomic file writes (.tmp -> fsync -> os.replace),
and full rollback of model and optimizer states on load failure.
"""

import copy
import hashlib
import json
import os
import time
from typing import Optional, Dict, Any
from ..nn.module import Module
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
    parent_dir = os.path.dirname(abs_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    tmp_path = f"{abs_path}.tmp"

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(container_json)
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename: replaces target file instantaneously
        os.replace(tmp_path, abs_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise CheckpointError(f"Failed to atomically save checkpoint to {path}: {e}") from e


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
        raise CheckpointSchemaError(f"Unsupported checkpoint format in {path}: {payload.get('format')}")

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
        if model is not None and orig_model_state is not None:
            try:
                model.load_state_dict(orig_model_state)
            except Exception:
                pass

        if optimizer is not None and orig_optimizer_state is not None:
            try:
                optimizer.load_state_dict(orig_optimizer_state)
            except Exception:
                pass

        raise e

    return {
        "epoch": epoch,
        "global_step": global_step,
        "timestamp": payload.get("timestamp"),
        "extra": payload.get("extra", {}),
    }
