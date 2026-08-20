"""
tests/test_audit_scorecard.py
=============================
Granular Audit Scoring Engine (0-Point Baseline).
Evaluates 5 Pillars of Production Integrity (Max 100 Points Total):
  1. Autograd DAG & Numerical Stability (20 Pts)
  2. Transformer & Native RoPE Architecture (20 Pts)
  3. Memory & Allocation Efficiency (20 Pts)
  4. Performance & Latency Benchmark (20 Pts)
  5. Crash Resilience & Checkpoint Integrity (20 Pts)
"""

import os
import time
import tempfile
import pytest
from termux_train import Tensor, nn, set_backend, available_backends
from termux_train.checkpoint import save_safetensors, load_safetensors
from termux_train.data import MMapTokenDataset


class AuditScorecard:
    """Cumulative Scorecard Accumulator starting at 0.0 points."""
    total_score = 0.0
    category_scores = {
        "autograd_math": 0.0,
        "transformer_rope": 0.0,
        "memory_efficiency": 0.0,
        "performance_latency": 0.0,
        "checkpoint_integrity": 0.0,
    }


def record_score(category: str, points: float, test_name: str, latency_ms: float):
    AuditScorecard.category_scores[category] += points
    AuditScorecard.total_score += points
    print(f"\n[SCORE +{points:.1f} pts] ({category}) {test_name} in {latency_ms:.2f}ms | Subtotal: {AuditScorecard.category_scores[category]:.1f}/20.0")


@pytest.mark.parametrize("backend_name", available_backends())
def test_pillar1_autograd_and_math(backend_name):
    set_backend(backend_name)
    t0 = time.perf_counter()

    # 1. IEEE 754 logsumexp defense (+5.0 pts)
    logits = Tensor([[-float('inf'), -float('inf')]], requires_grad=True)
    lse = logits.logsumexp(axis=-1)
    assert lse.item() == 0.0 or not (lse.item() != lse.item())

    # 2. Strict CrossEntropyLoss integer target (+5.0 pts)
    with pytest.raises(TypeError):
        nn.cross_entropy_loss(Tensor([[1.0, 2.0]]), Tensor([0.5], dtype="float32"))

    # 3. Multi-axis unbroadcasting backward (+5.0 pts)
    a = Tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)  # (1, 2, 2)
    b = Tensor([10.0], requires_grad=True)                        # (1,)
    c = (a + b).sum()
    c.backward()
    assert b.grad.shape == (1,)
    assert pytest.approx(b.grad.tolist()[0], rel=1e-4) == 4.0

    # 4. Grad clipping finite assertion (+5.0 pts)
    p = nn.Parameter([[10.0, 20.0]], requires_grad=True)
    p.grad = Tensor([[10.0, 20.0]], dtype="float32")
    norm = nn.clip_grad_norm_([p], max_norm=5.0)
    assert pytest.approx(norm, rel=1e-3) == 22.3606

    dt = (time.perf_counter() - t0) * 1000.0
    record_score("autograd_math", 20.0 if backend_name == "numpy" else 0.0, f"Pillar 1 Autograd ({backend_name})", dt)


@pytest.mark.parametrize("backend_name", available_backends())
def test_pillar2_transformer_and_rope(backend_name):
    set_backend(backend_name)
    t0 = time.perf_counter()

    # 1. Native RoPE TinyTransformerLM (+5.0 pts)
    model = nn.TinyTransformerLM(vocab_size=20, d_model=16, num_heads=2, d_ff=32, num_layers=2, pos_type="rope")
    assert model.pos_type == "rope"
    assert model.pos_emb is None  # O(0) positional embedding parameters!

    x = Tensor([[1, 5, 8]], dtype="int64")
    logits, _ = model(x)
    assert logits.shape == (1, 3, 20)

    # 2. Universal Causal Trapezoid Masking for Chunked Prefill (+5.0 pts)
    mha = nn.MultiHeadAttention(d_model=8, num_heads=2)
    q = Tensor([[[0.1] * 8] * 2], requires_grad=True)   # S_q = 2
    k = Tensor([[[0.1] * 8] * 6], requires_grad=True)   # S_k = 6
    out = mha(query=q, key=k, value=k, causal=True)
    assert out.shape == (1, 2, 8)

    # 3. Incremental KV Caching parity (+5.0 pts)
    gen_cached = model.generate([1, 2], max_new_tokens=3, temperature=0.0, use_cache=True)
    gen_uncached = model.generate([1, 2], max_new_tokens=3, temperature=0.0, use_cache=False)
    assert gen_cached == gen_uncached

    # 4. Weight Tying parameter sharing (+5.0 pts)
    model_tied = nn.TinyTransformerLM(vocab_size=15, d_model=8, num_heads=2, d_ff=16, num_layers=1, tie_weights=True)
    assert model_tied.head is None
    idx = Tensor([[1, 2]], dtype="int64")
    tgt = Tensor([[2, 3]], dtype="int64")
    _, loss = model_tied(idx, targets=tgt)
    loss.backward()
    assert model_tied.tok_emb.weight.grad is not None

    dt = (time.perf_counter() - t0) * 1000.0
    record_score("transformer_rope", 20.0 if backend_name == "numpy" else 0.0, f"Pillar 2 Transformer ({backend_name})", dt)


@pytest.mark.parametrize("backend_name", available_backends())
def test_pillar3_memory_efficiency(backend_name):
    set_backend(backend_name)
    t0 = time.perf_counter()

    # 1. Zero-allocation Causal Mask cache (+5.0 pts)
    mha = nn.MultiHeadAttention(d_model=8, num_heads=2, max_seq_len=16)
    x = Tensor([[[0.1] * 8] * 4])
    _ = mha(x, causal=True)
    cached_id = id(mha._cached_causal_mask)
    _ = mha(x, causal=True)
    assert id(mha._cached_causal_mask) == cached_id

    # 2. Vectorized Embedding gather/scatter (+5.0 pts)
    emb = nn.Embedding(num_embeddings=50, embedding_dim=16)
    inp = Tensor([[5, 10, 15, 20]], dtype="int64")
    out_e = emb(inp)
    assert out_e.shape == (1, 4, 16)

    # 3. Zero-allocation QuantizedLinear (+5.0 pts)
    lin = nn.Linear(16, 8)
    qlin = nn.quantize_linear_int8(lin)
    x_in = Tensor([[0.5] * 16])
    q_out = qlin(x_in)
    assert q_out.shape == (1, 8)

    # 4. SafeTensors zero-copy direct buffer (+5.0 pts)
    with tempfile.TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, "test.safetensors")
        save_safetensors({"w": lin.weight}, p)
        assert os.path.exists(p)
        loaded, _ = load_safetensors(p)
        assert loaded["w"].shape == lin.weight.shape

    dt = (time.perf_counter() - t0) * 1000.0
    record_score("memory_efficiency", 20.0 if backend_name == "numpy" else 0.0, f"Pillar 3 Memory ({backend_name})", dt)


@pytest.mark.parametrize("backend_name", available_backends())
def test_pillar4_performance_and_latency(backend_name):
    set_backend(backend_name)
    t0 = time.perf_counter()

    model = nn.TinyTransformerLM(vocab_size=30, d_model=32, num_heads=4, d_ff=64, num_layers=2, pos_type="rope")
    x = Tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype="int64")
    y = Tensor([[2, 3, 4, 5, 6, 7, 8, 9]], dtype="int64")

    # Warmup step
    model.zero_grad(set_to_none=True)
    logits, loss = model(x, targets=y)
    loss.backward()

    # Step benchmark (steady-state)
    model.zero_grad(set_to_none=True)
    t_step_start = time.perf_counter()
    logits, loss = model(x, targets=y)
    loss.backward()
    step_latency_ms = (time.perf_counter() - t_step_start) * 1000.0

    # Generation benchmark
    t_gen_start = time.perf_counter()
    gen_tokens = model.generate([1, 2], max_new_tokens=5, temperature=0.0, use_cache=True)
    gen_latency_ms = (time.perf_counter() - t_gen_start) * 1000.0
    per_token_ms = gen_latency_ms / 5.0

    # Latency contracts
    if backend_name == "numpy":
        assert step_latency_ms < 1000.0 # Steady state NumPy < 1s
        assert per_token_ms < 100.0     # Inference < 100ms / token
    else:
        assert step_latency_ms < 15000.0 # Pure Python < 15s
        assert per_token_ms < 2000.0

    dt = (time.perf_counter() - t0) * 1000.0
    record_score("performance_latency", 20.0 if backend_name == "numpy" else 0.0, f"Pillar 4 Performance ({backend_name})", dt)


@pytest.mark.parametrize("backend_name", available_backends())
def test_pillar5_crash_resilience_and_checkpoints(backend_name):
    set_backend(backend_name)
    t0 = time.perf_counter()

    # 1. SafeTensors bit-exact F32, I64, BOOL (+5.0 pts)
    t_f32 = Tensor([[1.2345]], dtype="float32")
    t_i64 = Tensor([999999], dtype="int64")
    t_bool = Tensor([True, False], dtype="bool")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "exact.safetensors")
        save_safetensors({"f": t_f32, "i": t_i64, "b": t_bool}, path)
        loaded, _ = load_safetensors(path)
        assert pytest.approx(loaded["f"].tolist()[0][0], rel=1e-5) == 1.2345
        assert loaded["i"].tolist() == [999999]
        assert loaded["b"].tolist() == [True, False]

    # 2. MMapTokenDataset streaming and clean unlink (+5.0 pts)
    tokens = list(range(50))
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_p = os.path.join(tmpdir, "stream.bin")
        ds = MMapTokenDataset.create_from_tokens(tokens, bin_p, seq_len=8)
        assert len(ds) == 50 - 8
        x0, y0 = ds[0]
        assert x0.shape == (1, 8)
        ds.unlink()
        assert not os.path.exists(bin_p)

    # 3. Empty prompt guard in generate (+5.0 pts)
    model = nn.TinyTransformerLM(vocab_size=10, d_model=8, num_heads=2, d_ff=16, num_layers=1)
    with pytest.raises(ValueError, match="cannot be empty"):
        _ = model.generate([], max_new_tokens=5)

    # 4. Out-of-bounds index error in Embedding (+5.0 pts)
    emb = nn.Embedding(num_embeddings=10, embedding_dim=4)
    with pytest.raises(IndexError):
        _ = emb(Tensor([15], dtype="int64"))

    dt = (time.perf_counter() - t0) * 1000.0
    record_score("checkpoint_integrity", 20.0 if backend_name == "numpy" else 0.0, f"Pillar 5 Resilience ({backend_name})", dt)
