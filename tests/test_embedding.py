"""
tests/test_embedding.py
=======================
Comprehensive Test Suite for nn.Embedding Layer:
  - 1D, 2D, 3D Forward Lookup Parity
  - Backward Gradient Accumulation on Duplicate Indices
  - padding_idx Masking (Zero Weights & Zero Gradients)
  - Index Out-of-Bounds & Negative Bounds Validation
  - State Dict Serialization & Load Roundtrip
  - Constructor Invariant Checks
"""

import pytest
from termux_train import Tensor, nn, set_backend, available_backends


@pytest.mark.parametrize("backend_name", available_backends())
def test_embedding_forward_shapes_and_values(backend_name):
    set_backend(backend_name)
    num_embeddings, embedding_dim = 10, 4
    # Pre-defined custom weight for exact check
    w_data = [[float(i * 10 + j) for j in range(embedding_dim)] for i in range(num_embeddings)]
    emb = nn.Embedding(num_embeddings, embedding_dim, _weight=Tensor(w_data))

    # 1. 1D input (Sequence)
    idx_1d = Tensor([0, 2, 5], dtype="int64")
    out_1d = emb(idx_1d)
    assert out_1d.shape == (3, embedding_dim)
    assert out_1d.tolist()[0] == w_data[0]
    assert out_1d.tolist()[1] == w_data[2]
    assert out_1d.tolist()[2] == w_data[5]

    # 2. 2D input (Batch, Sequence)
    idx_2d = Tensor([[1, 3], [4, 9]], dtype="int64")
    out_2d = emb(idx_2d)
    assert out_2d.shape == (2, 2, embedding_dim)
    assert out_2d.tolist()[0][0] == w_data[1]
    assert out_2d.tolist()[0][1] == w_data[3]
    assert out_2d.tolist()[1][0] == w_data[4]
    assert out_2d.tolist()[1][1] == w_data[9]

    # 3. 3D input (Batch, Heads, Sequence)
    idx_3d = Tensor([[[0, 1]], [[2, 3]]], dtype="int64")
    out_3d = emb(idx_3d)
    assert out_3d.shape == (2, 1, 2, embedding_dim)


@pytest.mark.parametrize("backend_name", available_backends())
def test_embedding_backward_gradient_accumulation(backend_name):
    set_backend(backend_name)
    num_embeddings, embedding_dim = 5, 3
    emb = nn.Embedding(num_embeddings, embedding_dim)
    # Indices containing duplicates: token 1 appears twice, token 3 appears once
    idx = Tensor([1, 3, 1], dtype="int64")
    out = emb(idx)

    loss = out.sum()
    loss.backward()

    assert emb.weight.grad is not None
    grad = emb.weight.grad.tolist()

    # Token 1 appeared twice, each element in output contributes 1.0 upstream grad
    assert grad[1] == [2.0, 2.0, 2.0]
    # Token 3 appeared once
    assert grad[3] == [1.0, 1.0, 1.0]
    # Unreferenced tokens have 0.0 gradient
    assert grad[0] == [0.0, 0.0, 0.0]
    assert grad[2] == [0.0, 0.0, 0.0]
    assert grad[4] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("backend_name", available_backends())
def test_embedding_padding_idx_masking(backend_name):
    set_backend(backend_name)
    num_embeddings, embedding_dim = 6, 4
    emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)

    # padding_idx row must be strictly zeroed at initialization
    assert emb.weight.tolist()[0] == [0.0, 0.0, 0.0, 0.0]

    # Input referencing padding_idx and regular tokens
    idx = Tensor([0, 2, 0], dtype="int64")
    out = emb(idx)
    assert out.tolist()[0] == [0.0, 0.0, 0.0, 0.0]

    loss = out.sum()
    loss.backward()

    grad = emb.weight.grad.tolist()
    # padding_idx gradient must remain zero
    assert grad[0] == [0.0, 0.0, 0.0, 0.0]
    # token 2 gets gradient
    assert grad[2] == [1.0, 1.0, 1.0, 1.0]


@pytest.mark.parametrize("backend_name", available_backends())
def test_embedding_index_out_of_bounds_errors(backend_name):
    set_backend(backend_name)
    emb = nn.Embedding(5, 3)

    # Index >= num_embeddings
    with pytest.raises(IndexError, match="out of range"):
        _ = emb(Tensor([5], dtype="int64"))

    # Negative index
    with pytest.raises(IndexError, match="out of range"):
        _ = emb(Tensor([-1], dtype="int64"))


@pytest.mark.parametrize("backend_name", available_backends())
def test_embedding_state_dict_and_load(backend_name):
    set_backend(backend_name)
    emb1 = nn.Embedding(4, 2)
    sd = emb1.state_dict()
    assert "weight" in sd
    assert len(sd["weight"]) == 4
    assert len(sd["weight"][0]) == 2

    emb2 = nn.Embedding(4, 2)
    emb2.load_state_dict(sd)
    assert emb2.weight.tolist() == emb1.weight.tolist()


def test_embedding_constructor_invariants():
    with pytest.raises(ValueError):
        _ = nn.Embedding(0, 4)
    with pytest.raises(ValueError):
        _ = nn.Embedding(10, 0)
    with pytest.raises(ValueError):
        _ = nn.Embedding(10, 4, padding_idx=10)
    with pytest.raises(ValueError):
        _ = nn.Embedding(10, 4, padding_idx=-1)
