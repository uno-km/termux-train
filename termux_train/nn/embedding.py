"""
termux_train.nn.embedding
=========================
Embedding Layer for Mapping Discrete Tokens to Differentiable Dense Vectors.
Supports arbitrary N-D index tensors, padding_idx zeroing, and vectorized backend scatter-add.
"""

from typing import Optional
from .module import Module
from .parameter import Parameter
from ..tensor import Tensor, randn, _attach_grad_fn


class Embedding(Module):
    """
    A lookup table that stores embeddings of a fixed dictionary and size.
    Maps an integer index Tensor of shape (...) to an embedding Tensor of shape (..., embedding_dim).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        _weight: Optional[Tensor] = None
    ):
        super().__init__()
        if not isinstance(num_embeddings, int) or num_embeddings <= 0:
            raise ValueError(f"num_embeddings must be a positive integer, got {num_embeddings}")
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be a positive integer, got {embedding_dim}")

        if padding_idx is not None:
            if not isinstance(padding_idx, int) or not (0 <= padding_idx < num_embeddings):
                raise ValueError(f"padding_idx must be in [0, {num_embeddings - 1}], got {padding_idx}")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        if _weight is not None:
            if _weight.shape != (num_embeddings, embedding_dim):
                raise ValueError(
                    f"Weight shape {_weight.shape} does not match (num_embeddings={num_embeddings}, "
                    f"embedding_dim={embedding_dim})"
                )
            self.weight = Parameter(_weight)
        else:
            init_t = randn((num_embeddings, embedding_dim), mean=0.0, std=1.0)
            self.weight = Parameter(init_t)

        if self.padding_idx is not None:
            self._zero_padding_idx()

    def _zero_padding_idx(self) -> None:
        if self.padding_idx is None:
            return
        backend = self.weight.backend
        w_flat = backend.to_flat_list(self.weight._data)
        start = self.padding_idx * self.embedding_dim
        for j in range(self.embedding_dim):
            w_flat[start + j] = 0.0
        new_data = backend.from_data(
            backend.reshape(w_flat, (self.num_embeddings, self.embedding_dim)),
            dtype="float32"
        )
        self.weight._replace_data(new_data, bump_version=False)

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward lookup:
          input shape: (...) with int64 values in [0, num_embeddings - 1]
          output shape: (..., embedding_dim)
        """
        backend = self.weight.backend
        input_t = self.weight._ensure_tensor_on_self_backend(input)
        flat_indices = backend.to_flat_list(input_t._data)

        v_size = self.num_embeddings
        e_dim = self.embedding_dim

        valid_int_indices = []
        for idx_val in flat_indices:
            idx = int(idx_val)
            if not (0 <= idx < v_size):
                raise IndexError(
                    f"Index out of range in Embedding lookup: got {idx}, but num_embeddings is {v_size}"
                )
            valid_int_indices.append(idx)

        out_shape = input_t.shape + (e_dim,)
        out_data = backend.gather_rows(self.weight._data, valid_int_indices, out_shape)
        out = Tensor(
            out_data,
            dtype="float32",
            requires_grad=self.weight.requires_grad,
            _prev=(self.weight,),
            _op="embedding",
            backend=backend
        )

        if out.requires_grad:
            p_idx = self.padding_idx
            w_weight = self.weight

            def _backward():
                if out.grad is not None and w_weight.requires_grad:
                    d_w_zeros = backend.zeros((v_size, e_dim), dtype="float32")
                    d_w_data = backend.scatter_add_rows(
                        d_w_zeros,
                        valid_int_indices,
                        out.grad._data,
                        padding_idx=p_idx
                    )
                    w_weight._accumulate_grad_data(d_w_data)

            _attach_grad_fn(out, (self.weight,), _backward)
        return out

    def __repr__(self) -> str:
        pad_str = f", padding_idx={self.padding_idx}" if self.padding_idx is not None else ""
        return f"Embedding({self.num_embeddings}, {self.embedding_dim}{pad_str})"
