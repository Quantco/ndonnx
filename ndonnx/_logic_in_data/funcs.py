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


def arange(
    start: int | float,
    /,
    stop: int | float | None = None,
    step: int | float = 1,
    *,
    dtype: DType | None = None,
    device=None,
) -> Array:
    import builtins

    if dtype is None:
        if builtins.all(
            isinstance(el, int) for el in [start, stop, step] if el is not None
        ):
            dtype = dtypes.default_int
        else:
            dtype = dtypes.default_float
    dtype = dtype or dtypes.default_float
    if not isinstance(dtype, dtypes.CoreDTypes):
        raise ValueError(f"Only core data types are supported, found `{dtype}`")

    return asarray(np.arange(start, stop, step), dtype=dtype)


def broadcast_to(x: Array, /, shape: tuple[int, ...] | Array) -> Array:
    from ._typed_array.core import TyArrayInt64

    if isinstance(shape, Array):
        if not isinstance(shape._data, TyArrayInt64):
            raise ValueError(
                f"dynamic shape must be of data type int64, found `{shape.dtype}`"
            )
        return Array._from_data(x._data.broadcast_to(shape._data))
    return Array._from_data(x._data.broadcast_to(shape))


def empty(
    shape: int | tuple[int, ...], *, dtype: DType | None = None, device=None
) -> Array:
    return zeros(shape=shape, dtype=dtype)


def empty_like(x: Array, /, *, dtype: DType | None = None, device=None) -> Array:
    return zeros_like(x, dtype=dtype)


def equal(x1: Array, x2: Array, /) -> Array:
    return x1 == x2


def eye(
    n_rows: int,
    n_cols: int | None = None,
    /,
    *,
    k: int = 0,
    dtype: DType | None = None,
    device=None,
) -> Array:
    nparr = np.eye(n_rows, n_cols, k=k)
    return asarray(nparr, dtype=dtype)


def full(
    shape: int | tuple[int, ...],
    fill_value: bool | int | float,
    *,
    dtype: DType | None = None,
    device=None,
) -> Array:
    if isinstance(shape, int):
        shape = (shape,)
    if dtype is None:
        if isinstance(fill_value, bool):
            dtype = dtypes.bool_
        elif isinstance(fill_value, int):
            dtype = dtypes.default_int
        elif isinstance(fill_value, float):
            dtype = dtypes.default_float
        else:
            raise TypeError(f"Unexpected 'fill_value' type `{type(fill_value)}`")
    return broadcast_to(asarray(fill_value, dtype=dtype), shape)


def full_like(
    x: Array,
    /,
    fill_value: bool | int | float,
    *,
    dtype: DType | None = None,
    device=None,
) -> Array:
    shape = x.dynamic_shape
    fill = asarray(fill_value, dtype=dtype or x.dtype)
    return broadcast_to(fill, shape)


def isnan(array: Array, /) -> Array:
    return Array._from_data(array._data.isnan())


def isfinite(array: Array, /) -> Array:
    return Array._from_data(array._data.isfinite())


def linspace(
    start: int | float | complex,
    stop: int | float | complex,
    /,
    num: int,
    *,
    dtype: DType | None = None,
    device=None,
    endpoint: bool = True,
) -> Array:
    dtype = dtype or dtypes.default_float
    if not isinstance(dtype, dtypes.CoreDTypes):
        raise ValueError(f"Only core data types are supported, found `{dtype}`")
    return asarray(np.linspace(start, stop, num=num, endpoint=endpoint), dtype=dtype)


def ones(
    shape: int | tuple[int, ...], *, dtype: DType | None = None, device=None
) -> Array:
    dtype = dtype or dtypes.default_float
    return full(shape, 1, dtype=dtype)


def ones_like(x: Array, /, *, dtype: DType | None = None, device=None) -> Array:
    dtype = dtype or x.dtype
    return full_like(x, 1, dtype=dtype)


def reshape(x: Array, /, shape: tuple[int, ...], *, copy: bool | None = None) -> Array:
    if copy is not None:
        raise ValueError("'copy' semantics are not implemented, yet")
    if len(shape) == 0:
        return x
    return Array._from_data(x._data.reshape(shape))


def where(cond: Array, a: Array, b: Array) -> Array:
    from ._typed_array.funcs import typed_where

    data = typed_where(cond._data, a._data, b._data)
    return Array._from_data(data)


def zeros(
    shape: int | tuple[int, ...], *, dtype: DType | None = None, device=None
) -> Array:
    dtype = dtype or dtypes.default_float
    return full(shape, 0, dtype=dtype)


def zeros_like(x: Array, /, *, dtype: DType | None = None, device=None) -> Array:
    dtype = dtype or x.dtype
    return full_like(x, 0, dtype=dtype)


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
