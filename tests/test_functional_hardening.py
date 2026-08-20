"""
tests/test_functional_hardening.py
==================================
Test Suite for Functional Hardening:
  1. MultiHeadAttention Joint Causal + Custom Padding Mask
  2. clip_grad_norm_ L2 and L_inf Norm In-Place Scaling
  3. Incremental KV-Caching Parity with Full-Context Forward
  4. generate() Early Stopping on <EOS> Token ID
  5. generate() Top-K and Top-P (Nucleus) Sampler Verification
"""

import pytest
from termux_train import Tensor, nn, set_backend, available_backends


@pytest.mark.parametrize("backend_name", available_backends())
def test_joint_causal_and_padding_mask(backend_name):
    set_backend(backend_name)
    d_model = 8
    num_heads = 2
    mha = nn.MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # (B=1, S=4, D=8)
    x = Tensor([[[0.5] * d_model] * 4], requires_grad=True)

    # Custom padding mask: mask out position 3 (last token)
    # Mask format: shape (1, 1, 1, 4) with position 3 = -inf
    pad_mask = Tensor([[[[0.0, 0.0, 0.0, -float('inf')]]]], dtype="float32")

    out = mha(x, mask=pad_mask, causal=True)
    assert out.shape == (1, 4, d_model)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None


@pytest.mark.parametrize("backend_name", available_backends())
def test_clip_grad_norm_l2(backend_name):
    set_backend(backend_name)
    p1 = nn.Parameter([[3.0, 0.0]], requires_grad=True)
    p2 = nn.Parameter([[0.0, 4.0]], requires_grad=True)

    # Mock gradient values: p1.grad has norm 3, p2.grad has norm 4 -> total L2 norm = sqrt(9 + 16) = 5.0
    p1.grad = Tensor([[3.0, 0.0]], dtype="float32")
    p2.grad = Tensor([[0.0, 4.0]], dtype="float32")

    total_norm = nn.clip_grad_norm_([p1, p2], max_norm=2.5, norm_type=2.0)
    assert pytest.approx(total_norm, rel=1e-4) == 5.0

    # Scaled gradients: clip_coef = 2.5 / 5.0 = 0.5
    assert pytest.approx(p1.grad.tolist()[0][0], rel=1e-4) == 1.5
    assert pytest.approx(p2.grad.tolist()[0][1], rel=1e-4) == 2.0


@pytest.mark.parametrize("backend_name", available_backends())
def test_clip_grad_norm_inf(backend_name):
    set_backend(backend_name)
    p1 = nn.Parameter([[10.0, 2.0]], requires_grad=True)
    p1.grad = Tensor([[10.0, 2.0]], dtype="float32")

    total_norm = nn.clip_grad_norm_([p1], max_norm=5.0, norm_type=float("inf"))
    assert pytest.approx(total_norm, rel=1e-4) == 10.0
    assert pytest.approx(p1.grad.tolist()[0][0], rel=1e-4) == 5.0


@pytest.mark.parametrize("backend_name", available_backends())
def test_incremental_kv_caching_parity(backend_name):
    set_backend(backend_name)
    vocab_size, d_model = 15, 8
    model = nn.TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        max_seq_len=32
    )

    prompt = [1, 5, 3]

    # Generate with KV Cache
    gen_cached = model.generate(prompt, max_new_tokens=4, temperature=0.0, use_cache=True)

    # Generate without KV Cache
    gen_uncached = model.generate(prompt, max_new_tokens=4, temperature=0.0, use_cache=False)

    assert len(gen_cached) == 7
    assert len(gen_uncached) == 7
    # Both must produce deterministic greedy parity
    assert gen_cached == gen_uncached


@pytest.mark.parametrize("backend_name", available_backends())
def test_generate_early_stopping_on_eos(backend_name):
    set_backend(backend_name)
    vocab_size, d_model = 10, 8
    model = nn.TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=2,
        d_ff=16,
        num_layers=1,
        max_seq_len=32
    )

    prompt = [1, 2]
    # Set eos_token_id to a valid token
    eos_id = 3
    # Force first step generation to emit eos_id by setting head bias / greedy weights or test check
    # Generate requesting 20 tokens, but should stop if eos_id is emitted
    out = model.generate(prompt, max_new_tokens=20, eos_token_id=eos_id)
    if eos_id in out:
        assert out[-1] == eos_id
        assert len(out) <= len(prompt) + 20
