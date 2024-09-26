# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

import ndonnx._logic_in_data as ndx

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
            dtype = ndx._default_int
        else:
            dtype = ndx._default_float
    dtype = dtype or ndx._default_float
    if not isinstance(dtype, dtypes.CoreDTypes):
        raise ValueError(f"Only core data types are supported, found `{dtype}`")

    return asarray(np.arange(start, stop, step), dtype=dtype)


def astype(x: Array, dtype: DType, /, *, copy: bool = True, device=None) -> Array:
    if not copy and x.dtype == dtype:
        return x
    return x.astype(dtype)


def broadcast_to(x: Array, /, shape: tuple[int, ...] | Array) -> Array:
    from ._typed_array import TyArrayInt64

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
            dtype = ndx.bool
        elif isinstance(fill_value, int):
            dtype = ndx._default_int
        elif isinstance(fill_value, float):
            dtype = ndx._default_float
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
    dtype = dtype or ndx._default_float
    if not isinstance(dtype, dtypes.CoreDTypes):
        raise ValueError(f"Only core data types are supported, found `{dtype}`")
    return asarray(np.linspace(start, stop, num=num, endpoint=endpoint), dtype=dtype)


def ones(
    shape: int | tuple[int, ...], *, dtype: DType | None = None, device=None
) -> Array:
    dtype = dtype or ndx._default_float
    return full(shape, 1, dtype=dtype)


def ones_like(x: Array, /, *, dtype: DType | None = None, device=None) -> Array:
    dtype = dtype or x.dtype
    return full_like(x, 1, dtype=dtype)


def permute_dims(x: Array, /, axes: tuple[int, ...]) -> Array:
    data = x._data.permute_dims(axes=axes)
    return Array._from_data(data)


def reshape(x: Array, /, shape: tuple[int, ...], *, copy: bool | None = None) -> Array:
    if copy is not None:
        raise ValueError("'copy' semantics are not implemented, yet")
    return Array._from_data(x._data.reshape(shape))


def sum(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: bool = False,
) -> Array:
    return Array._from_data(x._data.sum(axis=axis, dtype=dtype, keepdims=keepdims))


def where(cond: Array, a: Array, b: Array) -> Array:
    from ._typed_array import funcs as tyfuncs

    data = tyfuncs.where(cond._data, a._data, b._data)
    return Array._from_data(data)


def zeros(
    shape: int | tuple[int, ...], *, dtype: DType | None = None, device=None
) -> Array:
    dtype = dtype or ndx._default_float
    return full(shape, 0, dtype=dtype)


def zeros_like(x: Array, /, *, dtype: DType | None = None, device=None) -> Array:
    dtype = dtype or x.dtype
    return full_like(x, 0, dtype=dtype)
