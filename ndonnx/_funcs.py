# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
import math
from collections.abc import Sequence
from typing import Literal, NamedTuple
from warnings import warn

import numpy as np
from spox import Var

import ndonnx as ndx
from ndonnx.types import NestedSequence, OnnxShape, PyScalar

from ._array import Array, DType
from ._array_tyarray_interop import unwrap_tyarray
from ._namespace_info import Device
from ._typed_array import funcs as tyfuncs
from ._typed_array import onnx


def argument(
    *,
    shape: OnnxShape,
    dtype: ndx.DType,
) -> Array:
    """Creates a new lazy ndonnx array. This is used to define inputs to an ONNX model.

    Parameters
    ----------
    shape
        The shape of the array. String-dimensions denote symbolic dimensions and must be globally consistent.
        `None`-dimensions denote unknown dimensions.
    dtype
        The data type of the array.

    Returns
    -------
    Array
        The new array representing input(s) of the computational graphs.
    """
    return Array._argument(shape=shape, dtype=dtype)


def asarray(
    obj: Array | PyScalar | np.ndarray | NestedSequence | Var,
    /,
    *,
    dtype: ndx.DType | None = None,
    device: None | Device = None,
    copy: bool | None = None,
) -> Array:
    if not copy and copy is not None:
        # Must copy or raise
        if not isinstance(obj, Array):
            raise ValueError(
                f"cannot create 'Array' from object of type `{type(obj)}` without copying"
            )
        if obj.dtype == dtype or dtype is None:
            return Array._from_tyarray(obj._tyarray)
        raise ValueError(
            f"cannot create Array with data type `{dtype}` from array "
            f"of data type `{obj.dtype}` without copying data"
        )
    if isinstance(obj, Array):
        return Array._from_tyarray(tyfuncs.astyarray(obj._tyarray, dtype=dtype))
    else:
        if isinstance(obj, Sequence) and not isinstance(obj, str | Array):
            np_arr = np.asarray(obj, dtype=object)
            if (
                builtins.all(isinstance(el, Array) for el in np_arr.flatten())
                and np_arr.size > 0
            ):
                # This is a list of ndonnx arrays
                warn(
                    "providing a sequence of 'Array's to 'asarray' is not defined by the array-api-standard and may be removed from ndonnx in the future"
                )
                dtype = result_type(*np_arr.flatten()) if dtype is None else dtype
                out = concat([a.astype(dtype)[None, ...] for a in np_arr.flatten()])
                out_shape = concat([asarray(np_arr.shape), out.dynamic_shape[1:]])
                return reshape(out, out_shape)
        return Array._from_tyarray(tyfuncs.astyarray(obj, dtype=dtype))


def from_dlpack(
    x: object, /, *, device: None = None, copy: bool | None = None
) -> Array:
    raise BufferError("ndonnx does not (yet) support the export of array data")


def all(
    x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Array:
    return Array._from_tyarray(x._tyarray.all(axis=axis, keepdims=keepdims))


def any(
    x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Array:
    return Array._from_tyarray(x._tyarray.any(axis=axis, keepdims=keepdims))


def arange(
    start: int | float,
    /,
    stop: int | float | None = None,
    step: int | float = 1,
    *,
    dtype: DType | None = None,
    device: None | Device = None,
) -> Array:
    if dtype is None:
        if builtins.all(
            isinstance(el, int) for el in [start, stop, step] if el is not None
        ):
            dtype = ndx._default_int
        else:
            dtype = ndx._default_float
    dtype = dtype or ndx._default_float

    if stop is None:
        stop = start
        start = 0

    for el in [start, stop, step]:
        if not isinstance(el, int | float):
            raise TypeError(
                f"unexpected type for 'start', 'stop', or 'step': `{type(el)}`"
            )

    return Array._from_tyarray(tyfuncs.arange(dtype, start, stop, step))


def argmax(x: Array, /, *, axis: int | None = None, keepdims: bool = False) -> Array:
    return Array._from_tyarray(x._tyarray.argmax(axis=axis, keepdims=keepdims))


def argmin(x: Array, /, *, axis: int | None = None, keepdims: bool = False) -> Array:
    return Array._from_tyarray(x._tyarray.argmin(axis=axis, keepdims=keepdims))


def count_nonzero(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    return Array._from_tyarray(x._tyarray.count_nonzero(axis=axis, keepdims=keepdims))


def argsort(
    x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True
) -> Array:
    return Array._from_tyarray(
        x._tyarray.argsort(axis=axis, descending=descending, stable=stable)
    )


def nonzero(x: Array, /) -> tuple[Array, ...]:
    return tuple(Array._from_tyarray(el) for el in x._tyarray.nonzero())


def astype(
    x: Array, dtype: DType, /, *, copy: bool = True, device: None | Device = None
) -> Array:
    if not copy and x.dtype == dtype:
        return x
    return x.astype(dtype)


def broadcast_arrays(*arrays: Array) -> list[Array]:
    if len(arrays) < 2:
        return [a.copy() for a in arrays]

    def numeric_like(x):
        if isdtype(x, "numeric"):
            return x
        else:
            return zeros_like(x, dtype=ndx.int64)

    it = iter(arrays)
    ret = numeric_like(next(it))
    while (x := next(it, None)) is not None:
        ret = ret + numeric_like(x)

    target_shape = ret.dynamic_shape

    return [broadcast_to(a, target_shape) for a in arrays]


def broadcast_to(x: Array, /, shape: tuple[int, ...] | Array) -> Array:
    if isinstance(shape, Array):
        if not isinstance(shape._tyarray, onnx.TyArrayInt64):
            raise ValueError(
                f"dynamic shape must be of data type int64, found `{shape.dtype}`"
            )
        return Array._from_tyarray(x._tyarray.broadcast_to(shape._tyarray))
    return Array._from_tyarray(x._tyarray.broadcast_to(shape)).copy()


def can_cast(from_: DType | Array, to: DType, /) -> bool:
    try:
        result_type(from_, to)
        return True
    except TypeError:
        return False


def concat(
    arrays: tuple[Array, ...] | list[Array], /, *, axis: None | int = 0
) -> Array:
    data = tyfuncs.concat([arr._tyarray for arr in arrays], axis=axis)
    return Array._from_tyarray(data)


def cumulative_prod(
    x: Array,
    /,
    *,
    axis: int | None = None,
    dtype: DType | None = None,
    include_initial: bool = False,
) -> Array:
    data = x._tyarray.cumulative_prod(
        axis=axis, dtype=dtype, include_initial=include_initial
    )
    return Array._from_tyarray(data)


def cumulative_sum(
    x: Array,
    /,
    *,
    axis: int | None = None,
    dtype: DType | None = None,
    include_initial: bool = False,
) -> Array:
    data = x._tyarray.cumulative_sum(
        axis=axis, dtype=dtype, include_initial=include_initial
    )
    return Array._from_tyarray(data)


def max(
    x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Array:
    return Array._from_tyarray(x._tyarray.max(axis=axis, keepdims=keepdims))


def mean(
    x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Array:
    return Array._from_tyarray(x._tyarray.mean(axis=axis, keepdims=keepdims))


def meshgrid(*arrays: Array, indexing: str = "xy") -> list[Array]:
    ndim = len(arrays)

    if indexing not in ("xy", "ij"):
        raise ValueError(
            f"'indexing' argument must be one 'xy' or 'ij', found `{indexing}`"
        )

    base_shape = (1,) * ndim
    output = [
        reshape(x, base_shape[:i] + (-1,) + base_shape[i + 1 :])
        for i, x in enumerate(arrays)
    ]

    if indexing == "xy" and ndim > 1:
        # switch first and second axis
        output[0] = reshape(output[0], (1, -1) + base_shape[2:])
        output[1] = reshape(output[1], (-1, 1) + base_shape[2:])

    return broadcast_arrays(*output)


def moveaxis(
    x: Array, source: int | tuple[int, ...], destination: int | tuple[int, ...], /
) -> Array:
    return Array._from_tyarray(x._tyarray.moveaxis(source, destination))


def min(
    x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Array:
    return Array._from_tyarray(x._tyarray.min(axis=axis, keepdims=keepdims))


def prod(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: bool = False,
) -> Array:
    return Array._from_tyarray(
        x._tyarray.prod(axis=axis, dtype=dtype, keepdims=keepdims)
    )


def sort(
    x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True
) -> Array:
    return Array._from_tyarray(
        x._tyarray.sort(axis=axis, descending=descending, stable=stable)
    )


def std(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Array:
    return Array._from_tyarray(
        x._tyarray.std(axis=axis, correction=correction, keepdims=keepdims)
    )


def sum(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: bool = False,
) -> Array:
    return Array._from_tyarray(
        x._tyarray.sum(axis=axis, dtype=dtype, keepdims=keepdims)
    )


def var(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Array:
    return Array._from_tyarray(
        x._tyarray.variance(axis=axis, correction=correction, keepdims=keepdims)
    )


def empty(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    device: None | Device = None,
) -> Array:
    return zeros(shape=shape, dtype=dtype)


def empty_like(
    x: Array, /, *, dtype: DType | None = None, device: None | Device = None
) -> Array:
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
    # TODO: Add upstream tracking issue
    raise NotImplementedError


def log1p(x: Array, /) -> Array:
    # Requires special operator to meet standards precision requirements
    # TODO: Add upstream tracking issue
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
    device: None | Device = None,
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
    device: None | Device = None,
) -> Array:
    if dtype is None:
        dtype = tyfuncs._infer_dtype(fill_value)

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
    return broadcast_to(asarray(fill_value, dtype=dtype), shape)


def full_like(
    x: Array,
    /,
    fill_value: bool | int | float | str,
    *,
    dtype: DType | None = None,
    device: None | Device = None,
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
            return isinstance(dtype, onnx.IntegerDTypes)
        elif kind == "real floating":
            return isinstance(dtype, onnx.Floating)
        elif kind == "complex floating":
            # 'complex floating' is not supported"
            return False
        elif kind == "numeric":
            return isinstance(dtype, onnx.NumericDTypes)
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
    device: None | Device = None,
    endpoint: bool = True,
) -> Array:
    dtype = dtype or ndx._default_float
    if not isinstance(dtype, onnx.DTypes):
        raise ValueError(f"only primitive data types are supported, found `{dtype}`")
    return asarray(np.linspace(start, stop, num=num, endpoint=endpoint), dtype=dtype)


def matmul(x1: Array, x2: Array, /) -> Array:
    return x1 @ x2


def matrix_transpose(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.mT)


def ones(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    device: None | Device = None,
) -> Array:
    dtype = dtype or ndx._default_float
    shape = (shape,) if isinstance(shape, int) else shape
    return Array._from_tyarray(tyfuncs.ones(dtype, shape))


def ones_like(
    x: Array, /, *, dtype: DType | None = None, device: None | Device = None
) -> Array:
    dtype = dtype or x.dtype
    return full_like(x, 1, dtype=dtype)


def permute_dims(x: Array, /, axes: tuple[int, ...]) -> Array:
    data = x._tyarray.permute_dims(axes=axes)
    return Array._from_tyarray(data)


def reshape(
    x: Array, /, shape: tuple[int, ...] | Array, *, copy: bool | None = None
) -> Array:
    if copy is not None and not copy:
        raise ValueError("avoiding a copy in reshape operations is unsupported")
    if isinstance(shape, Array):
        shape_data = shape._tyarray
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
        return Array._from_tyarray(x._tyarray.reshape(shape_data))
    return Array._from_tyarray(x._tyarray.reshape(shape))


def repeat(x: Array, repeats: int | Array, /, *, axis: int | None = None) -> Array:
    repeats_: int | onnx.TyArrayInt64
    if isinstance(repeats, int):
        repeats_ = repeats
    elif isinstance(repeats._tyarray, onnx.TyArrayInteger):
        repeats_ = repeats._tyarray.astype(onnx.int64)
    else:
        raise TypeError(
            f"'repeats' argument must be of type 'int' or an array with an integer data type, found `{repeats}`"
        )

    return Array._from_tyarray(x._tyarray.repeat(repeats_, axis=axis))


def result_type(*arrays_and_dtypes: Array | DType | PyScalar) -> DType:
    def dtype_or_scalar(obj: Array | DType | PyScalar) -> DType | PyScalar:
        if isinstance(obj, Array):
            return obj.dtype
        return obj

    if len(arrays_and_dtypes) == 0:
        ValueError("at least one array or dtype is required")
    items = sorted(
        arrays_and_dtypes,
        key=lambda item: int(isinstance(item, Array | DType)),
        reverse=True,
    )
    first, *others = items

    if not isinstance(first, Array | DType):
        raise ValueError(
            "arguments to 'result_type' must contain at least one 'Array' or 'DType' object"
        )

    if isinstance(first, Array):
        first = first.dtype

    return tyfuncs.result_type(first, *(dtype_or_scalar(el) for el in others))


def roll(
    x: Array,
    /,
    shift: int | tuple[int, ...],
    *,
    axis: int | tuple[int, ...] | None = None,
) -> Array:
    return Array._from_tyarray(x._tyarray.roll(shift=shift, axis=axis))


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
    elif not isinstance(sorter._tyarray, onnx.TyArrayInteger):
        raise TypeError(
            f"'sorter' must have an integer data type, found `{sorter.dtype}`"
        )
    else:
        sorter_ = sorter._tyarray
    return Array._from_tyarray(
        tyfuncs.searchsorted(x1._tyarray, x2._tyarray, side=side, sorter=sorter_)
    )


def stack(arrays: tuple[Array, ...] | list[Array], /, *, axis: int = 0) -> Array:
    arrays = [expand_dims(x, axis=axis) for x in arrays]
    return concat(arrays, axis=axis)


def squeeze(x: Array, /, axis: int | tuple[int, ...]) -> Array:
    return Array._from_tyarray(x._tyarray.squeeze(axis))


def take(x: Array, indices: Array, /, *, axis: int | None = None) -> Array:
    if not isinstance(indices._tyarray, onnx.TyArrayInt64):
        raise TypeError(
            f"'indices' must be of data type 'int64' found `{indices.dtype}`"
        )
    return Array._from_tyarray(x._tyarray.take(indices._tyarray, axis=axis))


def take_along_axis(x: Array, indices: Array, /, *, axis: int = -1) -> Array:
    if not isinstance(indices._tyarray, onnx.TyArrayInt64):
        raise TypeError(
            f"'indices' must be of data type 'int64' found `{indices.dtype}`"
        )
    return Array._from_tyarray(x._tyarray.take_along_axis(indices._tyarray, axis=axis))


def tile(x: Array, repetitions: tuple[int, ...], /) -> Array:
    return Array._from_tyarray(x._tyarray.tile(repetitions))


def tril(x: Array, /, *, k: int = 0) -> Array:
    return Array._from_tyarray(x._tyarray.tril(k=k))


def triu(x: Array, /, *, k: int = 0) -> Array:
    return Array._from_tyarray(x._tyarray.triu(k=k))


def tensordot(
    x1: Array, x2: Array, /, *, axes: int | tuple[Sequence[int], Sequence[int]] = 2
) -> Array:
    res = x1._tyarray.__ndx_tensordot__(x2._tyarray, axes=axes)
    if res == NotImplemented:
        raise TypeError(
            f"unsupported operand data types for 'tensordot': `{x1.dtype}` and `{x2.dtype}`"
        )
    return Array._from_tyarray(res)


class UniqueAll(NamedTuple):
    values: Array
    indices: Array
    inverse_indices: Array
    counts: Array


def unique_all(x: Array, /) -> UniqueAll:
    values, indices, inverse_indices, counts = (
        Array._from_tyarray(tyarr) for tyarr in x._tyarray.unique_all()
    )
    return UniqueAll(values, indices, inverse_indices, counts)


class UniqueCounts(NamedTuple):
    values: Array
    counts: Array


def unique_counts(x: Array, /) -> UniqueCounts:
    uall = unique_all(x)
    return UniqueCounts(uall.values, uall.counts)


class UniqueInverse(NamedTuple):
    values: Array
    inverse_indices: Array


def unique_inverse(x: Array, /) -> UniqueInverse:
    uall = unique_all(x)
    return UniqueInverse(uall.values, uall.inverse_indices)


def unique_values(x: Array, /) -> Array:
    uall = unique_all(x)
    return uall.values


def unstack(x: Array, /, *, axis: int = 0) -> tuple[Array, ...]:
    # Only possible for statically known dimensions
    if not isinstance(x.shape[axis], int):
        raise ValueError(
            f"'unstack' can only be applied to statically known dimensions, but axis `{axis}` has dynamic length."
        )
    return tuple(el for el in moveaxis(x, axis, 0))


def vecdot(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    if x1._tyarray.shape[axis] != x2._tyarray.shape[axis]:
        raise ValueError("summed over dimensions must match")
    prod = x1 * x2
    return sum(prod, axis=axis, dtype=prod.dtype)


def where(
    cond: Array,
    a: Array | int | float | bool | str,
    b: Array | int | float | bool | str,
) -> Array:
    if not isinstance(cond._tyarray, onnx.TyArrayBool):
        raise TypeError(f"'cond' must be of data type 'bool', found `{cond.dtype}`")
    data = tyfuncs.where(cond._tyarray, unwrap_tyarray(a), unwrap_tyarray(b))
    return Array._from_tyarray(data)


def zeros(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    device: None | Device = None,
) -> Array:
    dtype = dtype or ndx._default_float
    shape = (shape,) if isinstance(shape, int) else shape
    return Array._from_tyarray(tyfuncs.zeros(dtype, shape))


def zeros_like(
    x: Array, /, *, dtype: DType | None = None, device: None | Device = None
) -> Array:
    dtype = dtype or x.dtype
    return full_like(x, 0, dtype=dtype)


def diff(
    a: Array,
    /,
    *,
    axis: int = -1,
    n: int = 1,
    prepend: Array | None = None,
    append: Array | None = None,
) -> Array:
    res = a._tyarray.diff(
        axis=axis,
        n=n,
        prepend=None if prepend is None else prepend._tyarray,
        append=None if append is None else append._tyarray,
    )
    return Array._from_tyarray(res)
