# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

import ndonnx._logic_in_data as ndx

from ._array import Array, asarray
from ._dtypes import DType
from ._typed_array import funcs as tyfuncs
from ._typed_array import onnx

if TYPE_CHECKING:
    pass


def from_dlpack(
    x: object, /, *, device: None = None, copy: bool | None = None
) -> Array:
    raise BufferError("ndonnx does not (yet) support the export of array data")


def all(
    x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Array:
    return Array._from_data(x._data.all(axis=axis, keepdims=keepdims))


def any(
    x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Array:
    return Array._from_data(x._data.any(axis=axis, keepdims=keepdims))


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


def argmax(x: Array, /, *, axis: int | None = None, keepdims: bool = False) -> Array:
    return Array._from_data(x._data.argmax(axis=axis, keepdims=keepdims))


def argmin(x: Array, /, *, axis: int | None = None, keepdims: bool = False) -> Array:
    return Array._from_data(x._data.argmin(axis=axis, keepdims=keepdims))


def argsort(
    x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True
) -> Array:
    return Array._from_data(
        x._data.argsort(axis=axis, descending=descending, stable=stable)
    )


def nonzero(x: Array, /) -> tuple[Array, ...]:
    return tuple(Array._from_data(el) for el in x._data.nonzero())


def astype(x: Array, dtype: DType, /, *, copy: bool = True, device=None) -> Array:
    if not copy and x.dtype == dtype:
        return x
    return x.astype(dtype)


def broadcast_arrays(*arrays: Array) -> list[Array]:
    raise NotImplementedError


def broadcast_to(x: Array, /, shape: tuple[int, ...] | Array) -> Array:
    from ._typed_array import TyArrayInt64

    if isinstance(shape, Array):
        if not isinstance(shape._data, TyArrayInt64):
            raise ValueError(
                f"dynamic shape must be of data type int64, found `{shape.dtype}`"
            )
        return Array._from_data(x._data.broadcast_to(shape._data))
    return Array._from_data(x._data.broadcast_to(shape))


def can_cast(from_: DType | Array, to: DType, /) -> bool:
    try:
        result_type(from_, to)
        return True
    except TypeError:
        return False


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


def max(
    x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Array:
    return Array._from_data(x._data.max(axis=axis, keepdims=keepdims))


def mean(
    x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Array:
    return Array._from_data(x._data.mean(axis=axis, keepdims=keepdims))


def meshgrid(*Arrays: Array, indexing: str = "xy") -> list[Array]:
    raise NotImplementedError


def moveaxis(
    x: Array, source: int | tuple[int, ...], destination: int | tuple[int, ...], /
) -> Array:
    raise NotImplementedError


def min(
    x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Array:
    return Array._from_data(x._data.min(axis=axis, keepdims=keepdims))


def prod(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: bool = False,
) -> Array:
    return Array._from_data(x._data.prod(axis=axis, dtype=dtype, keepdims=keepdims))


def sort(
    x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True
) -> Array:
    return Array._from_data(
        x._data.sort(axis=axis, descending=descending, stable=stable)
    )


def std(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Array:
    return Array._from_data(x._data.std())


def sum(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: bool = False,
) -> Array:
    return Array._from_data(x._data.sum(axis=axis, dtype=dtype, keepdims=keepdims))


def var(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Array:
    return Array._from_data(
        x._data.variance(axis=axis, correction=correction, keepdims=keepdims)
    )


def empty(
    shape: int | tuple[int, ...], *, dtype: DType | None = None, device=None
) -> Array:
    return zeros(shape=shape, dtype=dtype)


def empty_like(x: Array, /, *, dtype: DType | None = None, device=None) -> Array:
    return zeros_like(x, dtype=dtype)


def equal(x1: Array, x2: Array, /) -> Array:
    return x1 == x2


def expand_dims(x: Array, /, *, axis: int = 0) -> Array:
    if not -x.ndim - 1 <= axis <= x.ndim:
        raise IndexError(
            "'axis' must be within `[-{x.ndim}-1, {x.ndim}]`, found `{axis}`"
        )
    if axis < 0:
        axis = x.ndim + axis + 1
    key = tuple(None if el == axis else slice(None, None) for el in range(x.ndim + 1))
    return x[key]


def expm1(x: Array, /) -> Array:
    # Requires special operator to meet standards precision requirements
    raise NotImplementedError


def log1p(x: Array, /) -> Array:
    # Requires special operator to meet standards precision requirements
    raise NotImplementedError


def atan2(x1: Array, x2: Array, /) -> Array:
    # Requires special operator to meet standards precision requirements
    raise NotImplementedError


def conj(x: Array, /) -> Array:
    # Support for complex numbers is very broken in the ONNX standard
    raise NotImplementedError


def imag(x: Array, /) -> Array:
    # Support for complex numbers is very broken in the ONNX standard
    raise NotImplementedError


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

    if isinstance(shape, int):
        shape = (shape,)
    elif isinstance(shape, Array) and shape.ndim == 0:
        # Ensure shape is 1D
        shape = shape[None]

    if isinstance(shape, tuple):
        if len(shape) == 0:
            return asarray(fill_value, dtype=dtype)
        if math.prod(shape) == 0:
            # using `broadcast_to` to create an array with fewer elements
            # is technically supported by the ONNX standard, but quite odd
            # and seemingly broken in the onnxruntime.
            return reshape(asarray([], dtype=dtype), shape=shape)
    return broadcast_to(asarray([fill_value], dtype=dtype), shape)


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


def isdtype(dtype: DType, kind: DType | str | tuple[DType | str, ...]) -> bool:
    if isinstance(kind, str):
        if kind == "bool":
            return dtype == ndx.bool
        elif kind == "signed integer":
            return dtype in (ndx.int8, ndx.int16, ndx.int32, ndx.int64)
        elif kind == "unsigned integer":
            return dtype in (ndx.uint8, ndx.uint16, ndx.uint32, ndx.uint64)
        elif kind == "integral":
            return isinstance(dtype, ndx.Integer)
        elif kind == "real floating":
            return isinstance(dtype, ndx.Floating)
        elif kind == "complex floating":
            # 'complex floating' is not supported"
            return False
        elif kind == "numeric":
            return isinstance(dtype, ndx.Numeric)
    elif isinstance(kind, DType):
        return dtype == kind
    elif isinstance(kind, tuple):
        return builtins.any(isdtype(dtype, k) for k in kind)
    raise TypeError(f"kind must be a string or a dtype, not {type(kind)}")


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


def matmul(x1: Array, x2: Array, /) -> Array:
    return x1 @ x2


def matrix_transpose(x: Array, /) -> Array:
    return Array._from_data(x._data.mT)


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
        if not shape.ndim == 1 or not builtins.all(
            isinstance(el, int) for el in shape.shape
        ):
            # Otherwise, we lose the rank information which we assume
            # to always be available.
            raise ValueError(
                "'shape' must be a 1D tensor of static shape if provided as an 'Array'"
            )
        return Array._from_data(x._data.reshape(shape_data))
    return Array._from_data(x._data.reshape(shape))


def repeat(x: Array, repeats: int | Array, /, *, axis: int | None = None) -> Array:
    raise NotImplementedError


def result_type(*arrays_and_dtypes: Array | DType) -> DType:
    if len(arrays_and_dtypes) == 0:
        ValueError("at least one array or dtype is required")

    def get_dtype(obj: Array | DType) -> DType:
        if isinstance(obj, Array):
            return obj.dtype
        return obj

    return tyfuncs.result_type(*(get_dtype(el) for el in arrays_and_dtypes))


def roll(
    x: Array,
    /,
    shift: int | tuple[int, ...],
    *,
    axis: int | tuple[int, ...] | None = None,
) -> Array:
    raise NotImplementedError


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


def stack(arrays: tuple[Array, ...] | list[Array], /, *, axis: int = 0) -> Array:
    arrays = [expand_dims(x, axis=axis) for x in arrays]
    return concat(arrays, axis=axis)


def squeeze(x: Array, /, axis: int | tuple[int, ...]) -> Array:
    return Array._from_data(x._data.squeeze(axis))


def take(x: Array, indices: Array, /, *, axis: int | None = None) -> Array:
    from ._typed_array import TyArrayInt64

    if not isinstance(indices._data, TyArrayInt64):
        raise TypeError(
            "'indices' must be of data type 'int64' found `{indices.dtype}`"
        )
    return Array._from_data(x._data.take(indices._data, axis=axis))


def tile(x: Array, repetitions: tuple[int, ...], /) -> Array:
    raise NotImplementedError


def tril(x: Array, /, *, k: int = 0) -> Array:
    raise NotImplementedError


def triu(x: Array, /, *, k: int = 0) -> Array:
    raise NotImplementedError


def tensordot(
    x1: Array, x2: Array, /, *, axes: int | tuple[Sequence[int], Sequence[int]] = 2
) -> Array:
    raise NotImplementedError


def unique_all(x: Array, /) -> tuple[Array, Array, Array, Array]:
    raise NotImplementedError


def unique_counts(x: Array, /) -> tuple[Array, Array]:
    raise NotImplementedError


def unique_inverse(x: Array, /) -> tuple[Array, Array]:
    raise NotImplementedError


def unique_values(x: Array, /) -> Array:
    raise NotImplementedError


def unstack(x: Array, /, *, axis: int = 0) -> tuple[Array, ...]:
    raise NotImplementedError


def vecdot(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    raise NotImplementedError


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
