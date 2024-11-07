# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

import ndonnx._logic_in_data as ndx

from ._array import Array, asarray
from ._dtypes import DType
from ._typed_array import funcs as tyfuncs
from ._typed_array import onnx

if TYPE_CHECKING:
    pass


def all(
    x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Array:
    return Array._from_data(x._data.all())


def any(
    x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Array:
    return Array._from_data(x._data.any())


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
    if not isinstance(dtype, onnx.DTypes):
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


def concat(
    arrays: tuple[Array, ...] | list[Array], /, *, axis: None | int = 0
) -> Array:
    data = tyfuncs.concat([arr._data for arr in arrays], axis=axis)
    return Array._from_data(data)


def cumulative_sum(
    x: Array,
    /,
    *,
    axis: int | None = None,
    dtype: DType | None = None,
    include_initial: bool = False,
) -> Array:
    data = x._data.cumulative_sum(
        axis=axis, dtype=dtype, include_initial=include_initial
    )
    return Array._from_data(data)


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


def flip(x: Array, /, *, axis: int | tuple[int, ...] | None = None) -> Array:
    if axis is None:
        index = [slice(None, None, -1) for _ in range(x.ndim)]
    else:
        index = [slice(None, None, 1) for _ in range(x.ndim)]
        if isinstance(axis, int):
            axis = (axis,)
        for ax in axis:
            index[ax] = slice(None, None, -1)
    return x[tuple(index)]


def full(
    shape: int | tuple[int, ...] | Array,
    fill_value: bool | int | float | str,
    *,
    dtype: DType | None = None,
    device=None,
) -> Array:
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(shape, Array) and shape.ndim == 0:
        # Ensure shape is 1D
        shape = shape[None]
    if dtype is None:
        if isinstance(fill_value, bool):
            dtype = ndx.bool
        elif isinstance(fill_value, int):
            dtype = ndx._default_int
        elif isinstance(fill_value, float):
            dtype = ndx._default_float
        elif isinstance(fill_value, str):
            dtype = ndx.utf8
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
    if not isinstance(dtype, onnx.DTypes):
        raise ValueError(f"Only core data types are supported, found `{dtype}`")
    return asarray(np.linspace(start, stop, num=num, endpoint=endpoint), dtype=dtype)


def matrix_transpose(x: Array, /) -> Array:
    return Array._from_data(x._data.mT)


def mean(
    x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Array:
    return Array._from_data(x._data.mean())


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


def prod(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: bool = False,
) -> Array:
    return Array._from_data(x._data.prod(axis=axis, dtype=dtype, keepdims=keepdims))


def reshape(
    x: Array, /, shape: tuple[int, ...] | Array, *, copy: bool | None = None
) -> Array:
    if copy is not None:
        raise ValueError("'copy' semantics are not implemented, yet")
    if isinstance(shape, Array):
        shape_data = shape._data
        if not isinstance(shape_data, onnx.TyArrayInt64):
            raise TypeError(
                "'shape' must be of data type int64 if provided as an Array"
            )
        return Array._from_data(x._data.reshape(shape_data))
    return Array._from_data(x._data.reshape(shape))


def result_type(*arrays_and_dtypes: Array | DType) -> DType:
    if len(arrays_and_dtypes) == 0:
        ValueError("at least one array or dtype is required")

    def get_dtype(obj: Array | DType) -> DType:
        if isinstance(obj, Array):
            return obj.dtype
        return obj

    return tyfuncs.result_type(*(get_dtype(el) for el in arrays_and_dtypes))


def searchsorted(
    x1: Array,
    x2: Array,
    /,
    *,
    side: Literal["left", "right"] = "left",
    sorter: Array | None = None,
) -> Array:
    if sorter is None:
        sorter_ = None
    elif not isinstance(sorter._data, onnx.TyArrayInteger):
        raise TypeError(
            f"'sorter' must have an integer data type, found `{sorter.dtype}`"
        )
    else:
        sorter_ = sorter._data
    return Array._from_data(
        tyfuncs.searchsorted(x1._data, x2._data, side=side, sorter=sorter_)
    )


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
    if not isinstance(cond._data, onnx.TyArrayBool):
        raise TypeError("'cond' must be of data type 'bool'")
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
