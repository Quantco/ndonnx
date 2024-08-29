# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from ndonnx._utility import deprecated
from typing import Union
from .aliases import (
    bool,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    nbool,
    nfloat32,
    nfloat64,
    nint8,
    nint16,
    nint32,
    nint64,
    nuint8,
    nuint16,
    nuint32,
    nuint64,
    nutf8,
    uint8,
    uint16,
    uint32,
    uint64,
    utf8,
)
from .classes import (
    Floating,
    Integral,
    Nullable,
    NullableFloating,
    NullableIntegral,
    NullableNumerical,
    NullableUnsigned,
    Numerical,
    Unsigned,
    NullableCore,
    from_numpy_dtype,
    get_finfo,
    get_iinfo,
)
from .conversion import CastError, CastMixin
from .coretype import CoreType
from .schema import Schema
from .structtype import StructType


def into_nullable(dtype: StructType | CoreType) -> NullableCore:
    """Return nullable counterpart, if present.

    Parameters
    ----------
    dtype : StructType | CoreType
        A data type exported by ``ndonnx``.

    Returns
    -------
    out : NullableCore
        The nullable counterpart of the input type.

    Raises
    ------
    ValueError
        If the input type is unknown to ``ndonnx``.
    """
    if dtype == bool:
        return nbool
    elif dtype == float32:
        return nfloat32
    elif dtype == float64:
        return nfloat64
    elif dtype == int8:
        return nint8
    elif dtype == int16:
        return nint16
    elif dtype == int32:
        return nint32
    elif dtype == int64:
        return nint64
    elif dtype == uint8:
        return nuint8
    elif dtype == uint16:
        return nuint16
    elif dtype == uint32:
        return nuint32
    elif dtype == uint64:
        return nuint64
    elif dtype == utf8:
        return nutf8
    elif isinstance(dtype, NullableCore):
        return dtype
    else:
        raise ValueError(f"Cannot promote {dtype} to nullable")


Dtype = Union[CoreType, StructType]


@deprecated(
    "Function 'ndonnx.promote_nullable' will be deprecated in ndonnx 0.7. "
    "To create nullable array, use 'ndonnx.additional.make_nullable' instead."
)
def promote_nullable(dtype: StructType | CoreType) -> NullableCore:
    return into_nullable(dtype)


__all__ = [
    "CoreType",
    "StructType",
    "NullableCore",
    "NullableFloating",
    "NullableIntegral",
    "NullableUnsigned",
    "NullableNumerical",
    "Nullable",
    "Floating",
    "Integral",
    "Unsigned",
    "Numerical",
    "bool",
    "from_numpy_dtype",
    "float32",
    "float64",
    "int16",
    "int32",
    "int64",
    "int8",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
    "utf8",
    "nbool",
    "nfloat32",
    "nfloat64",
    "nint8",
    "nint16",
    "nint32",
    "nint64",
    "nuint8",
    "nuint16",
    "nuint32",
    "nuint64",
    "nutf8",
    "get_finfo",
    "get_iinfo",
    "promote_nullable",
    "Schema",
    "CastMixin",
    "CastError",
    "Dtype",
]
