# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import DType, dtypes
from .array import Array, asarray


def all(
    x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Array:
    return Array._from_data(x._data.all())


def isnan(array: Array, /) -> Array:
    return Array._from_data(array._data.isnan())


def isfinite(array: Array, /) -> Array:
    return Array._from_data(array._data.isfinite())


def arange(
    start: int | float,
    /,
    stop: int | float | None = None,
    step: int | float = 1,
    *,
    dtype: DType | None = None,
    device=None,
) -> Array:
    raise NotImplementedError


def equal(x1: Array, x2: Array, /) -> Array:
    return x1 == x2


def ones(
    shape: int | tuple[int, ...], *, dtype: DType | None = None, device=None
) -> Array:
    if dtype is None:
        dtype = dtypes.default_float

    if isinstance(shape, int):
        shape = (shape,)

    return full(shape, 1, dtype=dtype)


def full(
    shape: int | tuple[int, ...],
    fill_value: bool | int | float,
    *,
    dtype: DType | None = None,
    device=None,
) -> Array:
    if isinstance(shape, int):
        shape = (shape,)
    return broadcast_to(asarray(fill_value, dtype=dtype), shape)


def broadcast_to(x: Array, /, shape: tuple[int, ...]) -> Array:
    return Array._from_data(x._data.broadcast_to(shape))


def reshape(x: Array, /, shape: tuple[int, ...], *, copy: bool | None = None) -> Array:
    if copy is not None:
        raise ValueError("'copy' semantics are not implemented, yet")
    if len(shape) == 0:
        return x
    return Array._from_data(x._data.reshape(shape))


def zeros(
    shape: int | tuple[int, ...], *, dtype: DType | None = None, device=None
) -> Array:
    if dtype is None:
        dtype = dtypes.default_float

    if isinstance(shape, int):
        shape = (shape,)

    return reshape(asarray(0).astype(dtype), shape)


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
    if isinstance(ty, dtypes.NCoreIntegerDTypes):
        npdtype = dtypes.as_numpy(dtypes.as_non_nullable(ty))
    elif isinstance(ty, dtypes.CoreIntegerDTypes):
        npdtype = dtypes.as_numpy(ty)
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
    if isinstance(ty, dtypes.NCoreFloatingDTypes):
        npdtype = dtypes.as_numpy(dtypes.as_non_nullable(ty))
    elif isinstance(ty, dtypes.CoreFloatingDTypes):
        npdtype = dtypes.as_numpy(ty)
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
