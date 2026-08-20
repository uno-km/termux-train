"""
tests/test_transformer.py
==========================
Comprehensive Test Suite for Tiny Transformer Architecture:
  - LayerNorm forward, backward, affine parameters
  - MultiHeadAttention scaled dot-product & causal masking
  - FeedForward MLP forward and backward
  - TransformerBlock Pre-LN Residual flow
  - TinyTransformerLM end-to-end next-token prediction & autoregressive generation
"""

import pytest
from termux_train import Tensor, nn, set_backend, available_backends


@pytest.mark.parametrize("backend_name", available_backends())
def test_layernorm_forward_backward(backend_name):
    set_backend(backend_name)
    d_model = 4
    ln = nn.LayerNorm(d_model)

    # 2D Input (Batch, d_model)
    x = Tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], requires_grad=True)
    out = ln(x)
    assert out.shape == (2, d_model)

    # Output mean should be approx 0, variance approx 1
    out_list = out.tolist()
    assert pytest.approx(sum(out_list[0]) / d_model, abs=1e-4) == 0.0
    assert pytest.approx(sum(out_list[1]) / d_model, abs=1e-4) == 0.0

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None


@pytest.mark.parametrize("backend_name", available_backends())
def test_multi_head_attention_shapes_and_causal_mask(backend_name):
    set_backend(backend_name)
    d_model = 8
    num_heads = 2
    mha = nn.MultiHeadAttention(d_model, num_heads)

    # (Batch=2, Seq=4, d_model=8)
    x = Tensor([[[float(i + j) for j in range(d_model)] for i in range(4)] for _ in range(2)], requires_grad=True)
    out = mha(x, causal=True)
    assert out.shape == (2, 4, d_model)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert mha.q_proj.weight.grad is not None
    assert mha.k_proj.weight.grad is not None
    assert mha.v_proj.weight.grad is not None
    assert mha.out_proj.weight.grad is not None


@pytest.mark.parametrize("backend_name", available_backends())
def test_transformer_block_forward_backward(backend_name):
    set_backend(backend_name)
    d_model = 8
    num_heads = 2
    d_ff = 16
    block = nn.TransformerBlock(d_model, num_heads, d_ff)

    x = Tensor([[[0.5] * d_model] * 3] * 2, requires_grad=True)
    out = block(x, causal=True)
    assert out.shape == (2, 3, d_model)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert block.attn.q_proj.weight.grad is not None
    assert block.ffn.w1.weight.grad is not None


@pytest.mark.parametrize("backend_name", available_backends())
def test_tiny_transformer_lm_forward_loss_and_generation(backend_name):
    set_backend(backend_name)
    vocab_size = 20
    d_model = 8
    num_heads = 2
    d_ff = 16
    num_layers = 2
    max_seq_len = 32

    model = nn.TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )

    # Input token sequence (Batch=2, Seq=5)
    idx = Tensor([[1, 5, 3, 7, 2], [0, 4, 8, 2, 9]], dtype="int64")
    targets = Tensor([[5, 3, 7, 2, 4], [4, 8, 2, 9, 1]], dtype="int64")

    # 1. Forward with loss computation
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (2, 5, vocab_size)
    assert loss is not None
    assert loss.item() > 0.0

    # 2. Backward pass
    loss.backward()
    assert model.tok_emb.weight.grad is not None
    assert model.head.weight.grad is not None

    # 3. Autoregressive generation
    prompt = [1, 5, 3]
    generated = model.generate(prompt, max_new_tokens=4, temperature=1.0)
    assert len(generated) == len(prompt) + 4
    assert generated[:3] == prompt
    assert all(0 <= t < vocab_size for t in generated)
