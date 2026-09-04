"""
termux_train.checkpoint.lora_io
===============================
Lightweight LoRA Adapter Serialization & Deserialization Engine.
Isolates low-rank adapter weights (A, B) from base model parameters,
achieving >99% storage compression (<1MB adapter files) with SafeTensors and JSON support.
"""

import os
import json
from typing import Dict, Any, Optional
from ..nn.module import Module
from ..nn.lora import LoRALinear, adapter_state_dict, load_adapter_state_dict
from .safetensors import save_safetensors, load_safetensors
from ..tensor import Tensor


def save_lora_adapter(
    model: Module,
    filepath: str,
    adapter_name: str = "default",
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Saves only the LoRA low-rank adapter weights (A, B) and adapter configuration metadata.
    Excludes frozen base model weights, ensuring tiny checkpoint sizes (< 1MB).
    """
    if not isinstance(model, Module):
        raise TypeError(f"model must be an instance of nn.Module, got {type(model).__name__}")

    raw_state = adapter_state_dict(model)
    adapters = raw_state.get("adapters", {})
    if not adapters:
        raise ValueError("No LoRALinear layers found in the provided model to save.")

    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    adapter_meta = {
        "format": "termux-train-lora-model-adapter",
        "adapter_name": adapter_name,
        "base_model_class": type(model).__name__,
        "num_layers": len(adapters),
    }
    if metadata:
        adapter_meta.update(metadata)

    if filepath.endswith(".safetensors"):
        # Flatten tensors for SafeTensors binary packing
        tensors: Dict[str, Tensor] = {}
        layer_meta: Dict[str, Any] = {}

        for layer_name, l_dict in adapters.items():
            lora_a_data = l_dict["lora_A"]
            lora_b_data = l_dict["lora_B"]
            tensors[f"{layer_name}.lora_A"] = Tensor(lora_a_data, dtype="float32")
            tensors[f"{layer_name}.lora_B"] = Tensor(lora_b_data, dtype="float32")
            layer_meta[layer_name] = {
                "in_features": l_dict["in_features"],
                "out_features": l_dict["out_features"],
                "rank": l_dict["rank"],
                "alpha": l_dict["alpha"],
            }

        adapter_meta["layers"] = layer_meta
        str_meta = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in adapter_meta.items()}
        save_safetensors(tensors, filepath, metadata=str_meta)
    else:
        if not filepath.endswith(".json"):
            filepath = filepath + ".json"
        
        full_payload = {
            "format": "termux-train-lora-model-adapter",
            "version": "1.0",
            "metadata": adapter_meta,
            "adapters": adapters,
        }
        tmp_path = filepath + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(full_payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, filepath)
        except Exception as primary_err:
            cleanup_err: Exception | None = None
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError as _clean_exc:
                    cleanup_err = _clean_exc
            if cleanup_err is not None:
                raise IOError(
                    f"Failed writing LoRA adapter to '{filepath}': {primary_err!r}. "
                    f"Additionally, cleanup of temp file failed: {cleanup_err!r}. "
                    f"Quarantined path: {tmp_path}"
                ) from primary_err
            raise

    return filepath


def load_lora_adapter(
    model: Module,
    filepath: str,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Loads LoRA low-rank adapter weights into an existing base model with LoRALinear layers.
    Returns the loaded adapter metadata dictionary.
    """
    if not isinstance(model, Module):
        raise TypeError(f"model must be an instance of nn.Module, got {type(model).__name__}")

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"LoRA adapter file not found: '{filepath}'")

    if filepath.endswith(".safetensors"):
        tensors, raw_meta = load_safetensors(filepath)
        metadata = {}
        for k, v in raw_meta.items():
            try:
                metadata[k] = json.loads(v)
            except Exception:
                metadata[k] = v

        layer_meta = metadata.get("layers", {})
        adapters = {}
        for layer_name, l_info in layer_meta.items():
            key_a = f"{layer_name}.lora_A"
            key_b = f"{layer_name}.lora_B"
            if key_a not in tensors or key_b not in tensors:
                if strict:
                    raise KeyError(f"Missing adapter tensors for layer '{layer_name}' in {filepath}")
                continue

            adapters[layer_name] = {
                "format": "termux-train-lora-adapter",
                "version": "1.0",
                "in_features": l_info["in_features"],
                "out_features": l_info["out_features"],
                "rank": l_info["rank"],
                "alpha": l_info["alpha"],
                "lora_A": tensors[key_a].tolist(),
                "lora_B": tensors[key_b].tolist(),
            }

        state_dict = {
            "format": "termux-train-lora-model-adapter",
            "version": "1.0",
            "adapters": adapters,
        }
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            full_payload = json.load(f)
        metadata = full_payload.get("metadata", {})
        state_dict = {
            "format": full_payload.get("format", "termux-train-lora-model-adapter"),
            "version": full_payload.get("version", "1.0"),
            "adapters": full_payload.get("adapters", {}),
        }

    # Load and validate atomically into model
    load_adapter_state_dict(model, state_dict, strict=strict)
    return metadata
