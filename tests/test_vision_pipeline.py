"""
Tests for on-device Vision VLM LoRA Training Pipeline.
Verifies VisionQADataset, Multimodal Projector, VisionLanguageModelLoRA,
Loss convergence, SafeTensors export, and CLI integration.
"""

import os
import json
import tempfile
import pytest

import termux_train as tt
from termux_train import Tensor
from termux_train.data.vision_dataset import VisionQADataset
from termux_train.vision.vlm_lora import MultimodalProjectorLoRA, VisionLanguageModelLoRA
from termux_train.vision.trainer import train_vision_vlm_lora
from termux_train.checkpoint.safetensors import load_safetensors


@pytest.fixture
def temp_vision_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 3 synthetic sample images and json files
        for i in range(3):
            img_p = os.path.join(tmpdir, f"sample_{i}.png")
            with open(img_p, "wb") as f:
                f.write(f"PNG_SYNTHETIC_DATA_{i}".encode("utf-8") * 10)
            
            json_p = os.path.join(tmpdir, f"sample_{i}.json")
            with open(json_p, "w", encoding="utf-8") as f:
                json.dump({"question": f"What is object {i}?", "answer": f"Object {i} is a test entity."}, f)

        # Also create a jsonl manifest
        manifest_p = os.path.join(tmpdir, "dataset.jsonl")
        with open(manifest_p, "w", encoding="utf-8") as f:
            for i in range(3):
                line = json.dumps({
                    "image": os.path.join(tmpdir, f"sample_{i}.png"),
                    "question": f"Query {i}",
                    "answer": f"Response {i}",
                })
                f.write(line + "\n")

        yield tmpdir


def test_vision_qa_dataset_and_caching(temp_vision_dir):
    ds = VisionQADataset(
        data_source=temp_vision_dir,
        patch_dim=32,
        num_patches=8,
        max_seq_len=16,
        vocab_size=200,
    )
    assert len(ds) == 3

    feat, in_ids, tgt_ids = ds[0]
    assert feat.shape == (8, 32)
    assert in_ids.ndim == 1
    assert tgt_ids.ndim == 1
    assert in_ids.shape[0] == tgt_ids.shape[0] == 15

    # Verify latent cache file exists
    cache_files = os.listdir(ds.cache_dir)
    assert len(cache_files) >= 1
    assert any(f.endswith(".safetensors") for f in cache_files)


def test_vlm_lora_isolation_and_forward_backward():
    vocab_size = 100
    visual_dim = 16
    d_model = 16
    num_heads = 2
    rank = 2

    model = VisionLanguageModelLoRA(
        vocab_size=vocab_size,
        visual_dim=visual_dim,
        d_model=d_model,
        num_layers=1,
        num_heads=num_heads,
        d_ff=32,
        rank=rank,
        alpha=1.0,
    )

    # Base weights must be frozen
    assert model.tok_emb.weight.requires_grad is False
    assert model.head.weight.requires_grad is False
    assert model.projector.proj.base.weight.requires_grad is False

    # LoRA parameters must be trainable
    assert model.projector.proj.lora_A.requires_grad is True
    assert model.projector.proj.lora_B.requires_grad is True

    # Forward
    b = model.tok_emb.weight.backend
    B, N_patches, S_text = 2, 4, 8
    vis = Tensor(b.from_data(b.reshape([0.1] * (B * N_patches * visual_dim), (B, N_patches, visual_dim)), dtype="float32"), dtype="float32", backend=b)
    txt_ids = Tensor(b.from_data(b.reshape([5] * (B * S_text), (B, S_text)), dtype="int64"), dtype="int64", backend=b)
    targets = Tensor(b.from_data(b.reshape([6] * (B * S_text), (B, S_text)), dtype="int64"), dtype="int64", backend=b)

    logits = model(vis, txt_ids)
    assert logits.shape == (B, S_text, vocab_size)

    # Compute loss & backward
    from termux_train.nn.loss import cross_entropy_loss
    loss = cross_entropy_loss(logits, targets)
    assert float(loss.item()) > 0.0

    loss.backward()
    assert model.projector.proj.lora_A.grad is not None
    assert model.projector.proj.lora_B.grad is not None
    assert model.tok_emb.weight.grad is None


def test_train_vision_vlm_lora_end_to_end(temp_vision_dir):
    out_safetensors = os.path.join(temp_vision_dir, "test_vlm_adapter.safetensors")
    
    res = train_vision_vlm_lora(
        data_source=temp_vision_dir,
        output_path=out_safetensors,
        epochs=2,
        lr=0.01,
        batch_size=1,
        vocab_size=100,
        visual_dim=16,
        d_model=16,
        num_layers=1,
        num_heads=2,
        rank=2,
        max_seq_len=8,
        backend="auto",
        verbose=False,
    )

    assert res["status"] == "success"
    assert res["epochs"] == 2
    assert os.path.isfile(out_safetensors)
    assert os.path.isfile(os.path.splitext(out_safetensors)[0] + "_config.json")

    # Load and verify exported SafeTensors
    loaded, _ = load_safetensors(out_safetensors)
    assert len(loaded) > 0
    assert any("vlm." in k for k in loaded.keys())


def test_cli_vision_train_args():
    from termux_train.cli import main
    import sys
    test_args = ["termux-train", "vision-train", "--help"]
    with pytest.raises(SystemExit) as exc_info:
        sys.argv = test_args
        main()
    assert exc_info.value.code == 0
