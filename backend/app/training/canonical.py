from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize scientific metadata without process- or locale-dependent fields."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_array(array: NDArray[Any]) -> NDArray[Any]:
    """Return a C-contiguous, explicitly little-endian scientific array."""

    value = np.asarray(array)
    if value.dtype.hasobject:
        raise TypeError("object arrays are forbidden in AQSE scientific artifacts")
    if value.dtype.kind not in "biufc":
        raise TypeError(f"unsupported scientific array dtype: {value.dtype}")
    if value.dtype.kind == "b":
        dtype = np.dtype("bool")
    else:
        dtype = value.dtype.newbyteorder("<")
    return np.ascontiguousarray(value, dtype=dtype)


def array_identity(array: NDArray[Any]) -> dict[str, Any]:
    value = canonical_array(array)
    header = {
        "dtype": value.dtype.str,
        "endianness": "not-applicable" if value.dtype.itemsize == 1 else "little",
        "order": "C",
        "shape": list(value.shape),
    }
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(header))
    digest.update(b"\x00")
    digest.update(value.tobytes(order="C"))
    return {**header, "content_sha256": digest.hexdigest()}


def scientific_digest(metadata: Any, arrays: dict[str, NDArray[Any]]) -> str:
    """Hash canonical metadata and ordered numeric payloads only."""

    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(metadata))
    for name in sorted(arrays):
        digest.update(b"\x00")
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        identity = array_identity(arrays[name])
        digest.update(canonical_json_bytes(identity))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
