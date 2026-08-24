"""
termux-train Comprehensive Architectural Audit Verification Suite
==================================================================
Tests and verifies all 10 Level A critical compiler/autograd fixes,
numerical stability invariants, and mobile runtime safeguards.
"""

import math
import os
import tempfile
import pytest

from termux_train.tensor import Tensor, tensor, zeros, ones, randn, no_grad, _unbroadcast_to, _promote_dtype
from termux_train.nn.layernorm import LayerNorm
from termux_train.nn.embedding import Embedding
from termux_train.nn.attention import MultiHeadAttention
from termux_train.nn.rope import RotaryEmbedding
from termux_train.nn.loss import cross_entropy_loss, bce_loss, mse_loss
from termux_train.checkpoint.safetensors import save_safetensors, load_safetensors


def test_level_a_01_autograd_no_grad_isolation():
    """Verify that backward pass does not create new live autograd graph nodes."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[0.5, 1.5], [-1.0, 2.0]], requires_grad=True)
    c = a @ b
    loss = c.sum()

    # Initial state
    assert loss._grad_fn_state == "live"
    loss.backward()

    # Verify gradients computed cleanly
    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == (2, 2)
    assert b.grad.shape == (2, 2)
    # Verify graph intermediate nodes released when retain_graph=False
    assert len(c._prev) == 0
    assert c._grad_fn_state == "freed"


def test_level_a_02_logsumexp_softmax_all_masked_no_nan():
    """Verify logsumexp and softmax on all-negative-infinity rows produce exact 0.0 and NO NaN."""
    # Simulating all-masked attention row
    all_masked = Tensor([[-float('inf'), -float('inf'), -float('inf'), -float('inf')]], requires_grad=True)
    
    probs = all_masked.softmax(axis=-1)
    flat_probs = probs.tolist()[0]
    
    # Must not contain NaN and should sum to <= eps (0.0 probabilities)
    for p in flat_probs:
        assert not math.isnan(p), f"Softmax on all-masked row produced NaN: {p}"
        assert p == 0.0 or abs(p) < 1e-10


def test_level_a_03_bool_tensor_algebraic_ring_promotion():
    """Verify that adding bool tensors promotes to int64 and satisfies additive ring (1 + 1 = 2)."""
    t1 = Tensor([True, False, True], dtype="bool")
    t2 = Tensor([True, True, False], dtype="bool")
    
    res = t1 + t2
    assert res.dtype == "int64", f"Expected int64 dtype after bool addition, got {res.dtype}"
    assert res.tolist() == [2, 1, 1], f"Expected [2, 1, 1], got {res.tolist()}"


def test_level_a_04_unbroadcast_to_multidimensional():
    """Verify _unbroadcast_to reduces broadcasted dimensions without index shift."""
    g = Tensor([[[1.0, 2.0, 3.0, 4.0],
                 [1.0, 2.0, 3.0, 4.0],
                 [1.0, 2.0, 3.0, 4.0]],
                [[1.0, 2.0, 3.0, 4.0],
                 [1.0, 2.0, 3.0, 4.0],
                 [1.0, 2.0, 3.0, 4.0]]])  # shape (2, 3, 4)
    
    # Target shape (1, 4)
    reduced = _unbroadcast_to(g, (1, 4))
    assert reduced.shape == (1, 4)
    assert reduced.tolist() == [[6.0, 12.0, 18.0, 24.0]]

    # Target shape (4,)
    reduced_1d = _unbroadcast_to(g, (4,))
    assert reduced_1d.shape == (4,)
    assert reduced_1d.tolist() == [6.0, 12.0, 18.0, 24.0]


def test_level_a_05_tensor_replace_data_shape_invariance():
    """Verify that mutating tensor.data enforces shape and dtype invariance."""
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    
    # Valid shape replacement
    t._replace_data([[5.0, 6.0], [7.0, 8.0]])
    assert t.shape == (2, 2)
    
    # Invalid shape replacement must raise ValueError
    with pytest.raises(ValueError, match="Cannot replace tensor data with mismatched shape"):
        t._replace_data([1.0, 2.0, 3.0, 4.0])


def test_level_a_06_safetensors_corrupted_boundary_check():
    """Verify that safetensors loader rejects out-of-bounds offsets with ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "test.safetensors")
        import struct, json
        header = {
            "weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 999999999]  # Corrupted huge offset
            }
        }
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        header_len_b = struct.pack("<Q", len(header_bytes))
        with open(fpath, "wb") as f:
            f.write(header_len_b)
            f.write(header_bytes)
            f.write(b"12345678")

        with pytest.raises(ValueError, match="exceeds file boundary"):
            load_safetensors(fpath)


def test_level_a_07_rope_dynamic_table_expansion():
    """Verify that RoPE dynamically expands its frequency tables when context length exceeds max_seq_len."""
    rope = RotaryEmbedding(dim=32, max_seq_len=64)
    assert rope.max_seq_len == 64
    
    # Input with sequence length 128 (exceeding initial 64)
    x = randn((1, 2, 128, 32))
    out = rope(x)
    
    assert out.shape == (1, 2, 128, 32)
    assert rope.max_seq_len >= 128


def test_level_a_08_bce_loss_boundary_stability():
    """Verify BCELoss numerical stability when probabilities are near 0.0 or 1.0."""
    pred = Tensor([[0.00000001, 0.99999999]], requires_grad=True)
    target = Tensor([[0.0, 1.0]])
    
    loss = bce_loss(pred, target)
    assert not math.isnan(loss.item())
    assert not math.isinf(loss.item())
    
    loss.backward()
    assert pred.grad is not None
    for g in pred.grad.tolist()[0]:
        assert not math.isnan(g)
        assert not math.isinf(g)


def test_level_a_09_layernorm_forward_backward_numerical_accuracy():
    """Verify LayerNorm forward and backward gradient computation."""
    ln = LayerNorm(normalized_shape=4, eps=1e-5)
    x = Tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], requires_grad=True)
    
    out = ln(x)
    assert out.shape == (2, 4)
    
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None
    # For linear scale inputs, sum of LayerNorm gradient w.r.t input is close to 0
    grad_sum_row0 = sum(x.grad.tolist()[0])
    assert abs(grad_sum_row0) < 1e-4


def test_level_a_10_end_to_end_transformer_lora_training_step():
    """Verify full end-to-end forward, loss, backward, and parameter update."""
    from termux_train.nn.linear import Linear
    from termux_train.nn.lora import LoRALinear
    from termux_train.optim.adamw import AdamW
    
    layer = LoRALinear(in_features=8, out_features=8, rank=2, alpha=1.0)
    optimizer = AdamW(layer.parameters(), lr=0.01)
    
    x = randn((2, 8))
    target = randn((2, 8))
    
    # 1. Forward
    pred = layer(x)
    loss = mse_loss(pred, target)
    
    # 2. Backward
    optimizer.zero_grad()
    loss.backward()
    
    assert layer.lora_A.grad is not None
    assert layer.lora_B.grad is not None
    assert layer.base.weight.grad is None  # Base weights frozen
    
    # 3. Optimizer Step
    optimizer.step()
    assert optimizer.state[0]["step"] == 1


def test_level_b_01_cross_entropy_loss_ignore_index():
    """Verify CrossEntropyLoss with ignore_index correctly zeroes out gradients for padded tokens."""
    logits = Tensor([[[2.0, 1.0, 0.1], [0.5, 3.0, 0.2]],
                     [[0.1, 0.2, 4.0], [1.0, 1.0, 1.0]]], requires_grad=True)  # (2, 2, 3)
    targets = Tensor([[0, -100],
                      [2, 1]], dtype="int64")  # target at [0, 1] is ignored
    
    loss = cross_entropy_loss(logits, targets, ignore_index=-100)
    assert not math.isnan(loss.item())
    
    loss.backward()
    assert logits.grad is not None
    # Token at [0, 1] had ignore_index=-100 -> its gradient slice must be all zeros
    ignored_grad_slice = logits.grad.tolist()[0][1]
    for g in ignored_grad_slice:
        assert g == 0.0, f"Expected 0.0 gradient for ignored token, got {g}"


def test_level_b_02_embedding_padding_idx_zero_grad():
    """Verify that Embedding with padding_idx leaves padding embedding gradient as zero."""
    emb = Embedding(num_embeddings=10, embedding_dim=4, padding_idx=0)
    tokens = Tensor([[1, 0, 2]], dtype="int64")
    
    out = emb(tokens)
    loss = out.sum()
    loss.backward()
    
    assert emb.weight.grad is not None
    pad_grad = emb.weight.grad.tolist()[0]
    for g in pad_grad:
        assert g == 0.0, f"Expected zero gradient for padding_idx=0, got {g}"


def test_level_b_03_multihead_attention_causal_mask_and_kv_cache():
    """Verify MultiHeadAttention with causal mask and incremental KV cache consistency."""
    mha = MultiHeadAttention(d_model=16, num_heads=2, max_seq_len=32)
    
    # 1. Full Sequence Prompt Processing (S=4)
    prompt = randn((1, 4, 16))
    out_prompt, past_kv = mha(prompt, causal=True, use_cache=True)
    assert out_prompt.shape == (1, 4, 16)
    assert past_kv[0].shape == (1, 2, 4, 8)  # (B, H, S, d_k)
    
    # 2. Next Token Generation (S=1) with KV Cache
    next_token = randn((1, 1, 16))
    out_next, new_kv = mha(next_token, causal=True, past_key_value=past_kv, use_cache=True, position_offset=4)
    assert out_next.shape == (1, 1, 16)
    assert new_kv[0].shape == (1, 2, 5, 8)  # Cache extended to S=5


def test_level_b_04_inplace_mutation_autograd_version_check():
    """Verify that modifying a tensor needed for backward raises RuntimeError."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = a * 2.0
    c = (b * a).sum()  # uses 'a' in backward

    # In-place modify 'a'
    a._replace_data([[10.0, 20.0], [30.0, 40.0]])

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        c.backward()


def test_level_b_05_tensor_repr_large_tensor_truncation():
    """Verify that large tensors do not freeze repr/str output."""
    large_t = randn((100, 100))
    r = repr(large_t)
    assert "[... 10000 elements ...]" in r
    assert "shape=(100, 100)" in r


if __name__ == "__main__":
    pytest.main(["-v", __file__])
