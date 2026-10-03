"""
tests/test_adapters_hub.py
==========================
Unit and Integration Tests for Target-Aware Adapter Hub & Exporters.
Verifies adapter generation and tensor key mapping across all 6 ecosystem components:
llamacpp, bitnet, diffusion, vision, tts, stt.
"""

import os
import json
import pytest
from termux_train.nn.linear import Linear
from termux_train.nn.lora import LoRALinear
from termux_train.nn.dora import DoRALinear
from termux_train.adapters.hub import export_target_adapter, TargetEcosystem
from termux_train.checkpoint.safetensors import load_safetensors


def test_export_all_6_targets(tmp_path):
    base = Linear(16, 16)
    model = LoRALinear.from_linear(base, rank=4, alpha=8.0)

    targets = [
        (TargetEcosystem.LLAMACPP, "lora_llm.safetensors", "base_model.model.layer.lora_A.weight"),
        (TargetEcosystem.BITNET, "lora_bitnet.safetensors", "base_model.model.layer.lora_A.weight"),
        (TargetEcosystem.DIFFUSION, "lora_diffusion.safetensors", "lora_unet_layer.lora_down.weight"),
        (TargetEcosystem.VISION, "adapter_vision.safetensors", "vlm.layer.lora_A"),
        (TargetEcosystem.TTS, "adapter_tts.safetensors", "tts.speaker.layer.lora_A"),
        (TargetEcosystem.STT, "adapter_stt.safetensors", "stt.acoustic.layer.lora_A"),
    ]

    for target, filename, expected_key in targets:
        out_file = str(tmp_path / filename)
        saved_path = export_target_adapter(
            model,
            target=target,
            output_path=out_file,
            base_model_name=f"test-{target.value}"
        )
        assert os.path.exists(saved_path)

        # Verify SafeTensors load and metadata
        tensors, meta = load_safetensors(saved_path)
        assert expected_key in tensors
        assert meta["ecosystem_target"] == target.value
        assert meta["base_model"] == f"test-{target.value}"

        # Verify sidecar config.json
        cfg_file = str(tmp_path / filename).replace(".safetensors", "_config.json")
        assert os.path.exists(cfg_file)
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            assert cfg["target_ecosystem"] == target.value
            assert cfg["r"] == 4


def test_export_dora_adapter(tmp_path):
    base = Linear(8, 8)
    dora = DoRALinear.from_linear(base, rank=2, alpha=4.0)

    out_file = str(tmp_path / "dora_diff.safetensors")
    saved = export_target_adapter(
        dora,
        target=TargetEcosystem.DIFFUSION,
        output_path=out_file
    )
    assert os.path.exists(saved)
    tensors, meta = load_safetensors(saved)
    assert "lora_unet_layer.lora_down.weight" in tensors
    assert meta["ecosystem_target"] == "diffusion"
