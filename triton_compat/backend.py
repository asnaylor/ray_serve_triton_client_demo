from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


MODEL_NAME = "simple"
MODEL_VERSION = "1"


@dataclass(frozen=True)
class TensorSpec:
    name: str
    datatype: str
    shape: Tuple[int, ...]

    @property
    def element_count(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return int(n)


INPUT_SPECS: Tuple[TensorSpec, ...] = (
    TensorSpec(name="INPUT0", datatype="INT32", shape=(1, 16)),
    TensorSpec(name="INPUT1", datatype="INT32", shape=(1, 16)),
)
OUTPUT_SPECS: Tuple[TensorSpec, ...] = (
    TensorSpec(name="OUTPUT0", datatype="INT32", shape=(1, 16)),
    TensorSpec(name="OUTPUT1", datatype="INT32", shape=(1, 16)),
)

_TRITON_DTYPE_TO_NUMPY: Dict[str, np.dtype] = {
    "BOOL": np.dtype("?"),
    "UINT8": np.dtype("u1"),
    "UINT16": np.dtype("<u2"),
    "UINT32": np.dtype("<u4"),
    "UINT64": np.dtype("<u8"),
    "INT8": np.dtype("i1"),
    "INT16": np.dtype("<i2"),
    "INT32": np.dtype("<i4"),
    "INT64": np.dtype("<i8"),
    "FP16": np.dtype("<u2"),  # raw 16-bit (no float16 repeated field in proto)
    "BF16": np.dtype("<u2"),  # raw 16-bit
    "FP32": np.dtype("<f4"),
    "FP64": np.dtype("<f8"),
}


def triton_datatype_byte_size(datatype: str) -> int:
    if datatype == "BYTES":
        return -1
    dt = _TRITON_DTYPE_TO_NUMPY.get(datatype)
    if dt is None:
        raise ValueError(f"Unsupported datatype: {datatype}")
    return int(dt.itemsize)


def decode_raw_tensor(datatype: str, shape: Iterable[int], raw: bytes) -> np.ndarray:
    shape_t = tuple(int(d) for d in shape)
    element_count = int(np.prod(shape_t, dtype=np.int64))

    if datatype == "BYTES":
        values = decode_triton_bytes_tensor(raw, element_count)
        return np.array(values, dtype=object).reshape(shape_t)

    dt = _TRITON_DTYPE_TO_NUMPY.get(datatype)
    if dt is None:
        raise ValueError(f"Unsupported datatype: {datatype}")

    expected = element_count * int(dt.itemsize)
    if len(raw) != expected:
        raise ValueError(
            f"Raw buffer size mismatch for datatype={datatype} shape={shape_t}: "
            f"got {len(raw)} bytes, expected {expected} bytes"
        )

    arr = np.frombuffer(raw, dtype=dt, count=element_count)
    return arr.reshape(shape_t)


def encode_raw_tensor(datatype: str, arr: np.ndarray) -> bytes:
    if datatype == "BYTES":
        flat: List[bytes] = [bytes(x) for x in arr.reshape(-1).tolist()]
        return encode_triton_bytes_tensor(flat)

    dt = _TRITON_DTYPE_TO_NUMPY.get(datatype)
    if dt is None:
        raise ValueError(f"Unsupported datatype: {datatype}")
    return np.asarray(arr, dtype=dt, order="C").tobytes(order="C")


def decode_triton_bytes_tensor(raw: bytes, element_count: int) -> List[bytes]:
    """Decode Triton BYTES raw tensor payload.

    Each element is encoded as:
      uint32 length (little-endian) + payload bytes.
    """
    out: List[bytes] = []
    offset = 0
    for _ in range(int(element_count)):
        if offset + 4 > len(raw):
            raise ValueError("BYTES tensor decode: truncated length prefix")
        n = int.from_bytes(raw[offset : offset + 4], byteorder="little", signed=False)
        offset += 4
        if offset + n > len(raw):
            raise ValueError("BYTES tensor decode: truncated payload")
        out.append(raw[offset : offset + n])
        offset += n
    if offset != len(raw):
        raise ValueError("BYTES tensor decode: extra trailing bytes")
    return out


def encode_triton_bytes_tensor(values: Iterable[bytes]) -> bytes:
    chunks: List[bytes] = []
    for v in values:
        b = bytes(v)
        chunks.append(len(b).to_bytes(4, byteorder="little", signed=False))
        chunks.append(b)
    return b"".join(chunks)


def infer_simple(input0: np.ndarray, input1: np.ndarray) -> Dict[str, np.ndarray]:
    # Inputs are INT32 in little-endian; ensure int32 math and stable dtype.
    x0 = np.asarray(input0, dtype=np.int32)
    x1 = np.asarray(input1, dtype=np.int32)
    return {
        "OUTPUT0": x0 + x1,
        "OUTPUT1": x0 - x1,
    }

