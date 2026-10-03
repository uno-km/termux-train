"""
termux_train.adapters.hub
=========================
Target-Aware Cross-Component Adapter Hub & Exporter.
Serializes trained LoRA / DoRA / Fine-Tuning adapters into native, zero-conversion formats
specifically tailored for the 6 core Termux AI ecosystem runtimes:
  1. termux-llamacpp  (GGUF / SafeTensors Llama/Qwen/Mistral PEFT)
  2. termux-bitnet    (Ternary-base + FP LoRA Adapter)
  3. termux-diffusion (UNet / CLIP LoRA SafeTensors for SD-CLI & ComfyUI)
  4. termux-vision    (VLM Multimodal Projector & Detection Heads)
  5. termux-tts       (Speaker Embedding & Prosody Adaptation)
  6. termux-stt       (Whisper Cross-Attention & Acoustic CTC Adapters)
"""

from __future__ import annotations

import os
import json
from enum import Enum
from typing import Dict, Any, Optional, Union, List
from ..nn.module import Module
from ..nn.lora import adapter_state_dict
from ..checkpoint.safetensors import save_safetensors
from ..tensor import Tensor


class TargetEcosystem(str, Enum):
    LLAMACPP = "llamacpp"
    BITNET = "bitnet"
    DIFFUSION = "diffusion"
    VISION = "vision"
    TTS = "tts"
    STT = "stt"


def _format_diffusion_lora_keys(layer_name: str) -> Tuple[str, str, str]:
    """
    Converts generic layer names (e.g. 'unet.down.attn.q') into ComfyUI / Diffusers standard:
      lora_unet_<clean_name>.lora_down.weight
      lora_unet_<clean_name>.lora_up.weight
      lora_unet_<clean_name>.alpha
    """
    clean = layer_name.replace(".", "_")
    if not clean.startswith("lora_unet_") and not clean.startswith("lora_te_"):
        clean = f"lora_unet_{clean}"
    down_key = f"{clean}.lora_down.weight"
    up_key = f"{clean}.lora_up.weight"
    alpha_key = f"{clean}.alpha"
    return down_key, up_key, alpha_key


def _format_llm_peft_keys(layer_name: str) -> Tuple[str, str]:
    """
    Converts generic layer names into HuggingFace / Llama.cpp standard PEFT names:
      base_model.model.<layer_name>.lora_A.weight
      base_model.model.<layer_name>.lora_B.weight
    """
    prefix = "base_model.model."
    if layer_name.startswith("base_model."):
        clean = layer_name
    else:
        clean = f"{prefix}{layer_name}"
    return f"{clean}.lora_A.weight", f"{clean}.lora_B.weight"


def export_target_adapter(
    model_or_adapters: Union[Module, Dict[str, Any]],
    target: Union[TargetEcosystem, str],
    output_path: str,
    base_model_name: Optional[str] = None,
    custom_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Exports trained adapter weights mapped to target runtime format.

    Args:
        model_or_adapters: Module instance containing LoRA/DoRA layers or raw adapter state dict.
        target: Target ecosystem engine ('llamacpp', 'bitnet', 'diffusion', 'vision', 'tts', 'stt').
        output_path: Destination file path (typically ending in .safetensors or .json).
        base_model_name: Identifier of base model for provenance.
        custom_metadata: Additional key-value metadata to embed in checkpoint.

    Returns:
        Canonical absolute path to the generated adapter file.
    """
    if isinstance(target, str):
        target = TargetEcosystem(target.lower())

    if isinstance(model_or_adapters, Module):
        if hasattr(model_or_adapters, "lora_A") and hasattr(model_or_adapters, "lora_B"):
            adapters = {"layer": model_or_adapters.adapter_state_dict()}
        else:
            raw_state = adapter_state_dict(model_or_adapters)
            adapters = raw_state.get("adapters", {})
            if not adapters and hasattr(model_or_adapters, "adapter_state_dict"):
                single_res = model_or_adapters.adapter_state_dict()
                if "lora_A" in single_res:
                    adapters = {"layer": single_res}
                elif "adapters" in single_res:
                    adapters = single_res["adapters"]
    elif isinstance(model_or_adapters, dict):
        if "adapters" in model_or_adapters:
            adapters = model_or_adapters["adapters"]
        elif "lora_A" in model_or_adapters:
            adapters = {"layer": model_or_adapters}
        else:
            adapters = model_or_adapters
    else:
        raise TypeError(f"Expected Module or dict, got {type(model_or_adapters).__name__}")

    if not adapters:
        raise ValueError("No adapter layers found to export.")

    abs_output = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)

    tensors: Dict[str, Tensor] = {}
    meta_dict: Dict[str, Any] = {
        "ecosystem_target": target.value,
        "framework": "termux-train",
        "base_model": base_model_name or "unknown",
        "num_adapter_layers": str(len(adapters)),
    }
    if custom_metadata:
        meta_dict.update(custom_metadata)

    if target == TargetEcosystem.DIFFUSION:
        # Standard Stable Diffusion / ComfyUI LoRA format
        meta_dict["format"] = "diffusers-lora"
        for layer_name, l_info in adapters.items():
            down_k, up_k, alpha_k = _format_diffusion_lora_keys(layer_name)
            # lora_A (in_features, rank) -> lora_down (rank, in_features)
            # lora_B (rank, out_features) -> lora_up (out_features, rank)
            # or preserve 2D weight structure:
            tensors[down_k] = Tensor(l_info["lora_A"], dtype="float32")
            tensors[up_k] = Tensor(l_info["lora_B"], dtype="float32")
            meta_dict[alpha_k] = str(l_info.get("alpha", 1.0))

    elif target in (TargetEcosystem.LLAMACPP, TargetEcosystem.BITNET):
        # Llama.cpp / BitNet PEFT format
        meta_dict["format"] = "llama-peft"
        for layer_name, l_info in adapters.items():
            a_k, b_k = _format_llm_peft_keys(layer_name)
            tensors[a_k] = Tensor(l_info["lora_A"], dtype="float32")
            tensors[b_k] = Tensor(l_info["lora_B"], dtype="float32")
            meta_dict[f"{layer_name}.alpha"] = str(l_info.get("alpha", 1.0))
            meta_dict[f"{layer_name}.rank"] = str(l_info.get("rank", 4))

    elif target == TargetEcosystem.VISION:
        meta_dict["format"] = "termux-vision-adapter"
        for layer_name, l_info in adapters.items():
            tensors[f"vlm.{layer_name}.lora_A"] = Tensor(l_info["lora_A"], dtype="float32")
            tensors[f"vlm.{layer_name}.lora_B"] = Tensor(l_info["lora_B"], dtype="float32")

    elif target == TargetEcosystem.TTS:
        meta_dict["format"] = "termux-tts-voice-adapter"
        for layer_name, l_info in adapters.items():
            tensors[f"tts.speaker.{layer_name}.lora_A"] = Tensor(l_info["lora_A"], dtype="float32")
            tensors[f"tts.speaker.{layer_name}.lora_B"] = Tensor(l_info["lora_B"], dtype="float32")

    elif target == TargetEcosystem.STT:
        meta_dict["format"] = "termux-stt-acoustic-adapter"
        for layer_name, l_info in adapters.items():
            tensors[f"stt.acoustic.{layer_name}.lora_A"] = Tensor(l_info["lora_A"], dtype="float32")
            tensors[f"stt.acoustic.{layer_name}.lora_B"] = Tensor(l_info["lora_B"], dtype="float32")

    # Serialize to SafeTensors
    str_metadata = {k: str(v) if not isinstance(v, str) else v for k, v in meta_dict.items()}
    save_safetensors(tensors, abs_output, metadata=str_metadata)

    # Also emit adapter_config.json sidecar for full HuggingFace / Ecosystem compatibility
    config_path = os.path.splitext(abs_output)[0] + "_config.json"
    sidecar_data = {
        "target_ecosystem": target.value,
        "base_model_name_or_path": base_model_name or "unknown",
        "peft_type": "LORA",
        "r": int(adapters[next(iter(adapters))].get("rank", 4)),
        "lora_alpha": float(adapters[next(iter(adapters))].get("alpha", 1.0)),
        "target_modules": list(adapters.keys()),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(sidecar_data, f, indent=2)

    return abs_output
