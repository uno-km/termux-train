"""
termux_train.data.mmap_dataset
==============================
Memory-Mapped (mmap) Binary Token Streaming Dataset.
Enables training on 100GB+ text corpora on mobile devices with O(1) RAM usage.
Includes automated file descriptors and disk buffer lifecycle management (close / unlink / context-manager).
"""

import os
import struct
import mmap
from typing import List, Tuple, Optional, Iterator
from ..tensor import Tensor
from ..backend import get_backend, BaseBackend


class MMapTokenDataset:
    """
    Zero-RAM Streaming Dataset using OS virtual memory-mapping (mmap).
    Stores token sequences as contiguous 64-bit integer (int64) binary records on disk.
    """

    def __init__(
        self,
        filepath: str,
        seq_len: int,
        backend: Optional[BaseBackend] = None
    ):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Binary token file not found: {filepath}")

        self.filepath = os.path.abspath(filepath)
        self.seq_len = seq_len
        self.backend = backend or get_backend()

        self._file = open(self.filepath, "rb")
        self._file_size = os.path.getsize(self.filepath)

        if self._file_size < 8:
            self.close()
            raise ValueError(f"File {filepath} is too small to contain a valid token header.")

        # Read 8-byte header: total number of int64 tokens
        header_bytes = self._file.read(8)
        self.total_tokens = struct.unpack("<Q", header_bytes)[0]

        expected_size = 8 + self.total_tokens * 8
        if self._file_size < expected_size:
            self.close()
            raise ValueError(f"Corrupt binary dataset: file size {self._file_size} < expected {expected_size}")

        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._is_closed = False

        # Number of sliding sequence windows of length seq_len + 1 (input + target)
        self._num_samples = max(0, self.total_tokens - self.seq_len)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        if self._is_closed:
            raise RuntimeError("Cannot read from closed MMapTokenDataset.")

        if not (0 <= idx < self._num_samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {self._num_samples}")

        # Offset in bytes: 8 (header) + idx * 8 (each token is 8 bytes int64)
        byte_offset = 8 + idx * 8
        needed_bytes = (self.seq_len + 1) * 8

        raw_slice = self._mmap[byte_offset:byte_offset + needed_bytes]
        fmt = f"<{self.seq_len + 1}q"
        tokens = list(struct.unpack(fmt, raw_slice))

        inp_tokens = tokens[:-1]
        tgt_tokens = tokens[1:]

        x_data = self.backend.from_data([inp_tokens], dtype="int64")
        y_data = self.backend.from_data([tgt_tokens], dtype="int64")

        x = Tensor(x_data, dtype="int64", backend=self.backend)
        y = Tensor(y_data, dtype="int64", backend=self.backend)
        return x, y

    def close(self) -> None:
        """Closes memory map and underlying file descriptor to release OS handles."""
        if not self._is_closed:
            if hasattr(self, "_mmap") and self._mmap is not None:
                self._mmap.close()
                self._mmap = None
            if hasattr(self, "_file") and self._file is not None and not self._file.closed:
                self._file.close()
                self._file = None
            self._is_closed = True

    def unlink(self) -> None:
        """Closes handles and permanently removes the underlying binary file from disk."""
        self.close()
        if os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
            except OSError as _rm_err:
                import logging
                logging.getLogger(__name__).warning("MMapTokenDataset unlink error on '%s': %s", self.filepath, _rm_err)

    def __enter__(self) -> 'MMapTokenDataset':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @classmethod
    def create_from_tokens(
        cls,
        tokens: List[int],
        filepath: str,
        seq_len: int,
        backend: Optional[BaseBackend] = None
    ) -> 'MMapTokenDataset':
        """
        Creates a new binary dump file from a list of integer tokens and returns an open MMapTokenDataset.
        """
        parent = os.path.dirname(os.path.abspath(filepath))
        if parent:
            os.makedirs(parent, exist_ok=True)

        header = struct.pack("<Q", len(tokens))
        data_bytes = struct.pack(f"<{len(tokens)}q", *tokens)

        tmp_path = f"{filepath}.tmp"
        try:
            with open(tmp_path, "wb") as f:
                f.write(header)
                f.write(data_bytes)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, filepath)
        except Exception as primary_err:
            cleanup_err: Exception | None = None
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError as _clean_exc:
                    cleanup_err = _clean_exc
            if cleanup_err is not None:
                raise IOError(
                    f"Failed creating dataset '{filepath}': {primary_err!r}. "
                    f"Additionally, temp cleanup failed: {cleanup_err!r}. "
                    f"Quarantined path: {tmp_path}"
                ) from primary_err
            raise

        return cls(filepath=filepath, seq_len=seq_len, backend=backend)
