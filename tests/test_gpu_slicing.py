"""
AMEVA Single-Device GPU Slicing Tests for termux-train.
Covers Vocab Slicing, Chunked Dispatch, and Layer Streaming on TinyTransformerLM.
Component: [TRAIN-GPU-SLICING]
"""

import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock

import termux_train as tt
from termux_train import Tensor, randn
from termux_train.nn.transformer import TinyTransformerLM
from termux_train.runtime.runner import run_session
from termux_train.cli import main


def test_tiny_transformer_slicing_attributes():
    """Verify GPU slicing attributes are correctly initialized on TinyTransformerLM."""
    model = TinyTransformerLM(
        vocab_size=256,
        d_model=32,
        num_heads=2,
        d_ff=64,
        num_layers=4,
        vocab_slice=64,
        chunk_layers=2,
        stream_layers=1,
    )
    assert model.vocab_slice == 64
    assert model.chunk_layers == 2
    assert model.stream_layers == 1


def test_tiny_transformer_vocab_slice_forward_backward():
    """Verify vocab_slice truncates LM head projection and computes valid loss & gradients."""
    vocab_size = 256
    v_slice = 48
    seq_len = 8
    batch_size = 2

    model = TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=2,
        d_ff=64,
        num_layers=2,
        vocab_slice=v_slice,
    )

    idx_data = [[i % v_slice for i in range(seq_len)] for _ in range(batch_size)]
    target_data = [[(i + 1) % v_slice for i in range(seq_len)] for _ in range(batch_size)]

    idx = Tensor(idx_data, dtype="int64")
    targets = Tensor(target_data, dtype="int64")

    logits, loss = model(idx, targets=targets)
    
    # Sliced logits shape: (B, S, v_slice)
    assert logits.shape == (batch_size, seq_len, v_slice)
    assert loss is not None
    loss_val = float(loss.item())
    assert loss_val > 0.0

    # Verify backward pass executes cleanly through sliced projection
    loss.backward()
    assert model.head.weight.grad is not None
    assert model.tok_emb.weight.grad is not None


def test_tiny_transformer_chunk_layers_dispatch():
    """Verify chunk_layers executes blocks in segmented batches without corruption."""
    model = TinyTransformerLM(
        vocab_size=128,
        d_model=32,
        num_heads=2,
        d_ff=64,
        num_layers=6,
        chunk_layers=2,
    )

    idx = Tensor([[1, 2, 3, 4]], dtype="int64")
    targets = Tensor([[2, 3, 4, 5]], dtype="int64")

    logits, loss = model(idx, targets=targets)
    assert logits.shape == (1, 4, 128)
    assert loss is not None
    loss.backward()


def test_tiny_transformer_generate_with_vocab_slice():
    """Verify autoregressive generation operates safely under vocab_slice constraints."""
    v_slice = 32
    model = TinyTransformerLM(
        vocab_size=128,
        d_model=32,
        num_heads=2,
        d_ff=64,
        num_layers=2,
        vocab_slice=v_slice,
    )

    prompt = [5, 10, 15]
    generated = model.generate(prompt, max_new_tokens=4, use_cache=True)
    assert len(generated) == len(prompt) + 4
    # All generated tokens must be within vocab_slice
    for tok in generated[len(prompt):]:
        assert 0 <= tok < v_slice


def test_runner_gpu_slicing_metrics(capsys):
    """Verify runner emits gpu slicing metadata in __METRICS__."""
    cfg = {
        "modelType": "transformer",
        "dim": 16,
        "epochs": 1,
        "lr": 0.01,
        "batchSize": 2,
        "seqLen": 8,
        "vocabSize": 128,
        "vocab_slice": 32,
        "chunk_layers": 2,
        "stream_layers": 1,
    }

    run_session(cfg)
    captured = capsys.readouterr()
    lines = captured.out.strip().split("\n")

    found_slicing = False
    for line in lines:
        if line.startswith("__METRICS__:"):
            data = json.loads(line.replace("__METRICS__:", ""))
            if data.get("event") == "step":
                assert data.get("vocab_slice") == 32
                assert data.get("chunk_layers") == 2
                assert data.get("stream_layers") == 1
                found_slicing = True
                break

    assert found_slicing, "GPU slicing metadata was not emitted in runner metrics"


def test_train_cli_gpu_slicing_args(monkeypatch):
    """Verify CLI accepts --vocab-slice, --chunk-layers, --stream-layers flags."""
    test_args = [
        "termux-train",
        "train",
        "--model",
        "transformer",
        "--epochs",
        "1",
        "--vocab-slice",
        "64",
        "--chunk-layers",
        "2",
        "--stream-layers",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with patch("termux_train.cli.cmd_train") as mock_cmd_train:
        try:
            main()
        except SystemExit as e:
            assert e.code == 0 or e.code is None

        assert mock_cmd_train.call_count == 1
        args = mock_cmd_train.call_args[0][0]
        assert args.vocab_slice == 64
        assert args.chunk_layers == 2
        assert args.stream_layers == 1
