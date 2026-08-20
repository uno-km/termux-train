"""
tests/test_advanced_production_features.py
==========================================
Comprehensive Test Suite for Big-Tech Production Features:
  1. HuggingFace SafeTensors Zero-Copy Binary Serialization & Cleanup
  2. Rotary Position Embedding (RoPE) Forward, Backward, and Extrapolation
  3. MMapTokenDataset Memory-Mapped Streaming & Resource Lifecycle (close / unlink)
  4. INT8 Dynamic Weight Quantization & Inference Precision
"""

import os
import tempfile
import pytest
from termux_train import Tensor, nn, set_backend, available_backends
from termux_train.checkpoint import save_safetensors, load_safetensors
from termux_train.data import MMapTokenDataset


@pytest.mark.parametrize("backend_name", available_backends())
def test_safetensors_save_and_load_roundtrip(backend_name):
    set_backend(backend_name)
    t1 = Tensor([[1.0, -2.5, 3.14], [0.0, 42.0, -99.9]], dtype="float32")
    t2 = Tensor([10, 20, 30, 40], dtype="int64")
    t3 = Tensor([True, False, True], dtype="bool")

    tensors_dict = {"weight": t1, "indices": t2, "mask": t3}
    metadata = {"model": "tiny_transformer", "version": "1.0"}

    with tempfile.TemporaryDirectory() as tmpdir:
        st_path = os.path.join(tmpdir, "model.safetensors")
        save_safetensors(tensors_dict, st_path, metadata=metadata)
        assert os.path.exists(st_path)

        loaded_tensors, loaded_meta = load_safetensors(st_path)
        assert loaded_meta == metadata
        assert "weight" in loaded_tensors
        assert "indices" in loaded_tensors
        assert "mask" in loaded_tensors

        assert loaded_tensors["weight"].shape == t1.shape
        assert loaded_tensors["weight"].dtype == "float32"
        assert pytest.approx(loaded_tensors["weight"].tolist()[0][2], rel=1e-4) == 3.14

        assert loaded_tensors["indices"].shape == t2.shape
        assert loaded_tensors["indices"].dtype == "int64"
        assert loaded_tensors["indices"].tolist() == [10, 20, 30, 40]

        assert loaded_tensors["mask"].shape == t3.shape
        assert loaded_tensors["mask"].dtype == "bool"
        assert loaded_tensors["mask"].tolist() == [True, False, True]


@pytest.mark.parametrize("backend_name", available_backends())
def test_rotary_position_embedding_forward_backward(backend_name):
    set_backend(backend_name)
    dim = 8
    rope = nn.RotaryEmbedding(dim=dim, max_seq_len=64)

    # (Batch=2, Seq=4, Dim=8)
    x = Tensor([[[float(i + j) for j in range(dim)] for i in range(4)] for _ in range(2)], requires_grad=True)
    out = rope(x, position_offset=0)
    assert out.shape == (2, 4, dim)

    # Position offset extrapolation
    out_offset = rope(x, position_offset=10)
    assert out_offset.shape == (2, 4, dim)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None


@pytest.mark.parametrize("backend_name", available_backends())
def test_mmap_token_dataset_streaming_and_cleanup(backend_name):
    set_backend(backend_name)
    sample_tokens = list(range(100))
    seq_len = 16

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "corpus.bin")
        ds = MMapTokenDataset.create_from_tokens(sample_tokens, bin_path, seq_len=seq_len)
        assert os.path.exists(bin_path)
        assert len(ds) == 100 - seq_len

        # Read samples
        x0, y0 = ds[0]
        assert x0.shape == (1, seq_len)
        assert y0.shape == (1, seq_len)
        assert x0.tolist()[0] == list(range(16))
        assert y0.tolist()[0] == list(range(1, 17))

        # Close and unlink cleanup
        ds.close()
        ds.unlink()
        assert not os.path.exists(bin_path)


@pytest.mark.parametrize("backend_name", available_backends())
def test_int8_quantization_inference_parity(backend_name):
    set_backend(backend_name)
    in_features, out_features = 8, 4
    linear = nn.Linear(in_features, out_features, bias=True)

    # Quantize to INT8
    qlinear = nn.quantize_linear_int8(linear)
    assert isinstance(qlinear, nn.QuantizedLinear)
    assert qlinear.scale > 0.0

    # Forward check
    x = Tensor([[1.0] * in_features])
    out_fp32 = linear(x)
    out_int8 = qlinear(x)

    assert out_int8.shape == (1, out_features)
    # Output should be close within quantization resolution
    fp32_list = out_fp32.tolist()[0]
    int8_list = out_int8.tolist()[0]
    for v1, v2 in zip(fp32_list, int8_list):
        assert abs(v1 - v2) < 0.2
