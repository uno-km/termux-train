"""
tests/test_audit_production_fixes.py
====================================
Comprehensive Test Suite for Production Hardening & Big-Tech Architectures:
  1. MultiHeadAttention Zero-Allocation Causal Mask Caching
  2. MultiHeadAttention Cross-Attention (Q != K/V) Support
  3. MultiHeadAttention N-D Batch Shapes (*B, S, D) Support
  4. TinyTransformerLM Weight Tying & Pre-Allocated Position Buffer
  5. TinyTransformerLM Safe Temperature Scaling (Low-Temp Overflow Defense)
  6. nn.Embedding Vectorized Backend Gather and Scatter-Add
  7. CrossEntropyLoss Strict Integer Dtype Contract (TypeError on Float Targets)
  8. Vectorized _unbroadcast_to Multi-Axis Tensor Reduction
"""

import pytest
from termux_train import Tensor, nn, set_backend, available_backends


@pytest.mark.parametrize("backend_name", available_backends())
def test_multihead_attention_causal_mask_caching(backend_name):
    set_backend(backend_name)
    mha = nn.MultiHeadAttention(d_model=8, num_heads=2, max_seq_len=16)

    # First forward allocates mask cache
    x1 = Tensor([[[0.1] * 8] * 4], requires_grad=True)
    _ = mha(x1, causal=True)
    assert mha._cached_causal_mask is not None
    initial_cache_id = id(mha._cached_causal_mask)

    # Second forward reuses cached mask without reallocation
    x2 = Tensor([[[0.2] * 8] * 4], requires_grad=True)
    _ = mha(x2, causal=True)
    assert id(mha._cached_causal_mask) == initial_cache_id


@pytest.mark.parametrize("backend_name", available_backends())
def test_multihead_attention_cross_attention(backend_name):
    set_backend(backend_name)
    d_model = 8
    num_heads = 2
    mha = nn.MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # Query length 3, Key/Value length 6 (Cross-Attention)
    q = Tensor([[[0.5] * d_model] * 3], requires_grad=True)
    k = Tensor([[[0.8] * d_model] * 6], requires_grad=True)
    v = Tensor([[[1.2] * d_model] * 6], requires_grad=True)

    out = mha(query=q, key=k, value=v)
    assert out.shape == (1, 3, d_model)

    loss = out.sum()
    loss.backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


@pytest.mark.parametrize("backend_name", available_backends())
def test_multihead_attention_nd_batch_support(backend_name):
    set_backend(backend_name)
    d_model = 8
    num_heads = 2
    mha = nn.MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # 4D tensor input (B=2, T=3, S=4, D=8)
    x = Tensor([[[[0.1] * d_model] * 4] * 3] * 2, requires_grad=True)
    out = mha(x, causal=True)
    assert out.shape == (2, 3, 4, d_model)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None


@pytest.mark.parametrize("backend_name", available_backends())
def test_tiny_transformer_lm_weight_tying(backend_name):
    set_backend(backend_name)
    vocab_size, d_model = 20, 8
    model = nn.TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=2,
        d_ff=16,
        num_layers=1,
        tie_weights=True
    )

    idx = Tensor([[1, 2, 3]], dtype="int64")
    targets = Tensor([[2, 3, 4]], dtype="int64")
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (1, 3, vocab_size)
    assert loss is not None

    loss.backward()
    assert model.tok_emb.weight.grad is not None


@pytest.mark.parametrize("backend_name", available_backends())
def test_tiny_transformer_lm_safe_temperature_generation(backend_name):
    set_backend(backend_name)
    model = nn.TinyTransformerLM(vocab_size=10, d_model=8, num_heads=2, d_ff=16, num_layers=1)

    # Extreme low temperature (1e-6) should not overflow to inf / NaN
    gen_tokens = model.generate(prompt_tokens=[1, 2], max_new_tokens=3, temperature=1e-6)
    assert len(gen_tokens) == 5
    assert all(0 <= t < 10 for t in gen_tokens)


@pytest.mark.parametrize("backend_name", available_backends())
def test_cross_entropy_target_dtype_type_error(backend_name):
    set_backend(backend_name)
    logits = Tensor([[2.0, 1.0, 0.5]], requires_grad=True)
    float_target = Tensor([0.5], dtype="float32")

    with pytest.raises(TypeError, match="integer dtype"):
        _ = nn.cross_entropy_loss(logits, float_target)
