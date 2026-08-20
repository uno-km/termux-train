"""
termux_train.checkpoint.safetensors
===================================
HuggingFace-Compatible Zero-Copy SafeTensors Binary Serialization Engine.
Uses C-level direct byte stream (tobytes / frombuffer) on NumPy to eliminate O(N) Python list conversion.
"""

import json
import os
import shutil
import struct
from typing import Dict, Any, Optional, Tuple
from ..tensor import Tensor
from ..backend import get_backend, BaseBackend

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


DTYPE_TO_SAFETENSORS = {
    "float32": "F32",
    "int64": "I64",
    "bool": "BOOL",
}

SAFETENSORS_TO_DTYPE = {
    "F32": "float32",
    "I64": "int64",
    "BOOL": "bool",
}

DTYPE_TO_NP = {
    "float32": "float32",
    "int64": "int64",
    "bool": "bool",
}

STRUCT_FORMAT = {
    "float32": "f",
    "int64": "q",
    "bool": "?",
}


def _tensor_to_raw_bytes(tensor: Tensor) -> bytes:
    """Converts tensor data to contiguous raw little-endian binary bytes with zero-copy NumPy fast path."""
    dtype = tensor.dtype
    if HAS_NUMPY and isinstance(tensor._data, np.ndarray):
        np_dt = np.dtype(DTYPE_TO_NP.get(dtype, "float32")).newbyteorder("<")
        return np.ascontiguousarray(tensor._data, dtype=np_dt).tobytes()

    backend = tensor.backend
    flat_data = backend.to_flat_list(tensor._data)
    fmt_char = STRUCT_FORMAT.get(dtype, "f")

    # Chunked packing to avoid CPython *args argument limits on large tensors
    chunk_size = 32768
    byte_chunks = []
    for i in range(0, len(flat_data), chunk_size):
        chunk = flat_data[i:i + chunk_size]
        byte_chunks.append(struct.pack(f"<{len(chunk)}{fmt_char}", *chunk))
    return b"".join(byte_chunks)


def _raw_bytes_to_tensor_data(raw_bytes: bytes, shape: Tuple[int, ...], dtype: str, backend: BaseBackend) -> Any:
    """Reconstructs native backend tensor data from contiguous raw little-endian bytes with zero-copy NumPy."""
    if HAS_NUMPY and getattr(backend, "name", "").lower() == "numpy":
        np_dt = np.dtype(DTYPE_TO_NP.get(dtype, "float32")).newbyteorder("<")
        arr = np.frombuffer(raw_bytes, dtype=np_dt).reshape(shape).copy()
        return arr

    fmt_char = STRUCT_FORMAT.get(dtype, "f")
    num_elements = 1
    for d in shape:
        num_elements *= d

    chunk_size = 32768
    itemsize = 4 if dtype == "float32" else (8 if dtype == "int64" else 1)
    unpacked_list = []
    for i in range(0, num_elements, chunk_size):
        c_len = min(chunk_size, num_elements - i)
        c_bytes = raw_bytes[i * itemsize:(i + c_len) * itemsize]
        unpacked_list.extend(struct.unpack(f"<{c_len}{fmt_char}", c_bytes))

    return backend.from_data(backend.reshape(unpacked_list, shape), dtype=dtype)


def save_safetensors(
    tensors_dict: Dict[str, Tensor],
    filepath: str,
    metadata: Optional[Dict[str, str]] = None
) -> None:
    """
    Saves dictionary of Tensors into HuggingFace-compatible .safetensors binary file format.
    Uses atomic temporary write with guaranteed cleanup on failure.
    """
    parent_dir = os.path.dirname(os.path.abspath(filepath))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    header: Dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata

    binary_buffers = []
    current_offset = 0

    for name, tensor in tensors_dict.items():
        raw_b = _tensor_to_raw_bytes(tensor)
        b_len = len(raw_b)
        st_dtype = DTYPE_TO_SAFETENSORS.get(tensor.dtype, "F32")
        header[name] = {
            "dtype": st_dtype,
            "shape": list(tensor.shape),
            "data_offsets": [current_offset, current_offset + b_len]
        }
        binary_buffers.append(raw_b)
        current_offset += b_len

    header_json_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_json_bytes)

    # 8-byte little endian unsigned int (u64)
    header_len_bytes = struct.pack("<Q", header_len)

    tmp_path = os.path.join(parent_dir, f".tmp_safe_{os.path.basename(filepath)}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(header_len_bytes)
            f.write(header_json_bytes)
            for buf in binary_buffers:
                f.write(buf)
            f.flush()
            os.fsync(f.fileno())

        try:
            os.replace(tmp_path, filepath)
        except OSError as os_err:
            if getattr(os_err, "errno", None) == 18:  # EXDEV cross-device link
                shutil.move(tmp_path, filepath)
            else:
                raise
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise IOError(f"Failed to atomically write safetensors to {filepath}: {e}") from e


def load_safetensors(
    filepath: str,
    backend: Optional[BaseBackend] = None
) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    """
    Loads Tensors and optional metadata from a .safetensors binary file.
    Returns (tensors_dict, metadata_dict).
    """
    b = backend or get_backend()
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"SafeTensors file not found: {filepath}")

    with open(filepath, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError("Invalid safetensors file: insufficient header length bytes.")

        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_json_bytes = f.read(header_len)
        header = json.loads(header_json_bytes.decode("utf-8"))

        data_start_offset = 8 + header_len
        metadata = header.pop("__metadata__", {})

        tensors: Dict[str, Tensor] = {}
        for name, info in header.items():
            dtype_str = SAFETENSORS_TO_DTYPE.get(info["dtype"], "float32")
            shape = tuple(info["shape"])
            start_off, end_off = info["data_offsets"]
            f.seek(data_start_offset + start_off)
            raw_bytes = f.read(end_off - start_off)

            data = _raw_bytes_to_tensor_data(raw_bytes, shape, dtype_str, backend=b)
            tensors[name] = Tensor(data, dtype=dtype_str, backend=b)

    return tensors, metadata
