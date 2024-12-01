# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

import numpy as np

from ._array import Array
from ._dtypes import DType
from ._typed_array import masked_onnx, onnx


@dataclass
class Iinfo:
    # Number of bits occupied by the type.
    bits: int
    # Largest representable number.
    max: int
    # Smallest representable number.
    min: int
    # Integer data type.
    dtype: DType


def iinfo(ty: DType | Array, /) -> Iinfo:
    if isinstance(ty, Array):
        ty = ty.dtype
    if isinstance(ty, masked_onnx.NCoreIntegerDTypes):
        npdtype = masked_onnx.as_non_nullable(ty).unwrap_numpy()
    elif isinstance(ty, onnx.IntegerDTypes):
        npdtype = ty.unwrap_numpy()
    else:
        raise ValueError(f"'Iinfo' not available for type `{ty}`")
    info = np.iinfo(npdtype)
    return Iinfo(bits=info.bits, max=info.max, min=info.min, dtype=ty)


@dataclass
class Finfo:
    # number of bits occupied by the real-valued floating-point data type.
    bits: int
    # difference between 1.0 and the next smallest representable real-valued floating-point number larger than 1.0 according to the IEEE-754 standard.
    eps: float
    # largest representable real-valued number.
    max: float
    # smallest representable real-valued number.
    min: float
    # smallest positive real-valued floating-point number with full precision.
    smallest_normal: float
    # real-valued floating-point data type.
    dtype: DType


def finfo(ty: DType | Array, /):
    if isinstance(ty, Array):
        ty = ty.dtype
    if isinstance(ty, masked_onnx.NCoreFloatingDTypes):
        npdtype = masked_onnx.as_non_nullable(ty).unwrap_numpy()
    elif isinstance(ty, onnx.FloatingDTypes):
        npdtype = ty.unwrap_numpy()
    else:
        raise ValueError(f"'FIinfo' not available for type `{ty}`")
    finfo = np.finfo(npdtype)
    return Finfo(
        bits=finfo.bits,
        max=float(finfo.max),
        min=float(finfo.min),
        dtype=ty,
        smallest_normal=float(finfo.smallest_normal),
        eps=float(finfo.eps),
    )
