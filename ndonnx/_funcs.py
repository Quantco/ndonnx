# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
import functools
import operator
from collections import namedtuple
from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

import ndonnx._data_types as dtypes
from ndonnx._data_types import CastError, CastMixin, CoreType, _NullableCore
from ndonnx._data_types.structtype import StructType

from . import _opset_extensions as opx
from ._array import Array, _from_corearray
from ._utility import (
    promote,
)

# creation.py


def arange(
    start,
    stop=None,
    step=1,
    *,
    dtype: dtypes.CoreType | dtypes.StructType | None = None,
    device=None,
):
    step = asarray(step)
    if stop is None:
        stop = start
        start = asarray(0)
    elif start is None:
        start = asarray(0)

    start, stop, step = promote(start, stop, step)

    if not builtins.all(isinstance(x.dtype, CoreType) for x in (start, stop, step)):
        raise TypeError(
            "Cannot perform arange with nullable `start`, `stop` or `step` parameters."
        )

    if dtype is None:
        if builtins.all(
            isinstance(x.dtype, dtypes.Integral) for x in [start, stop, step]
        ):
            dtype = dtypes.int64
        else:
            dtype = dtypes.float64

    return astype(
        _from_corearray(opx.range(start._core(), stop._core(), step._core())), dtype
    )


def asarray(
    x: npt.ArrayLike | Array,
    dtype: dtypes.CoreType | dtypes.StructType | None = None,
    copy=None,
    device=None,
) -> Array:
    if not isinstance(x, Array):
        arr = np.asanyarray(
            x,
            dtype=(
                dtype.to_numpy_dtype() if isinstance(dtype, dtypes.CoreType) else None
            ),
        )
        if dtype is None:
            dtype = dtypes.from_numpy_dtype(arr.dtype)
            if isinstance(arr, np.ma.masked_array):
                dtype = dtypes.promote_nullable(dtype)

        ret = Array._construct(
            shape=arr.shape, dtype=dtype, eager_values=dtype._parse_input(arr)
        )
    else:
        ret = x.copy() if copy is True else x

    if dtype is not None:
        ret = ret.astype(dtype)

    return ret


def empty(shape, dtype: dtypes.CoreType | dtypes.StructType | None = None, device=None):
    shape = asarray(shape, dtype=dtypes.int64)
    if not isinstance(shape.dtype, dtypes.Integral):
        raise TypeError("shape must be an integer array")
    dtype = dtype or dtypes.float64
    return full(shape, 0, dtype=dtype)


def empty_like(
    x, dtype: dtypes.CoreType | dtypes.StructType | None = None, device=None
):
    dtype = x.dtype if dtype is None else dtype
    return full(asarray(x).shape, 0, dtype=dtype)


def eye(
    n_rows,
    n_cols=None,
    k=0,
    dtype: dtypes.StructType | dtypes.CoreType | None = None,
    device=None,
):
    dtype = dtype or dtypes.float64
    if n_cols is None:
        n_cols = n_rows
    n_rows = asarray(n_rows, dtype=dtypes.int64)
    n_cols = asarray(n_cols, dtype=dtypes.int64)
    n_rows = reshape(n_rows, [-1])
    n_cols = reshape(n_cols, [-1])

    if not isinstance(n_rows.dtype, CoreType):
        raise TypeError("n_rows must be a non-nullable integer array")

    if not isinstance(n_cols.dtype, CoreType):
        raise TypeError("n_cols must be a non-nullable integer array")

    shape = concat([n_rows, n_cols])
    dummy = zeros(shape, dtype=dtype)
    return _from_corearray(opx.eye_like(dummy._core(), k=k))


def full(
    shape,
    fill_value,
    *,
    dtype: dtypes.CoreType | dtypes.StructType | None = None,
    device=None,
):
    fill_value = asarray(fill_value)
    if fill_value.ndim != 0:
        raise ValueError("fill_value must be a scalar")

    dtype = fill_value.dtype if dtype is None else dtype
    shape = asarray(shape, dtype=dtypes.int64)._core()
    return fill_value._transmute(lambda corearray: opx.expand(corearray, shape)).astype(
        dtype
    )


def full_like(
    x,
    fill_value,
    dtype: dtypes.CoreType | dtypes.StructType | None = None,
    device=None,
):
    fill_value = asarray(fill_value)
    ret = broadcast_to(fill_value, asarray(x).shape)
    dtype = x.dtype if dtype is None else dtype
    return astype(ret, dtype)


def linspace(
    start: int | float,
    stop: int | float,
    num: int,
    *,
    dtype: dtypes.CoreType | dtypes.StructType | None = None,
    device=None,
    endpoint: bool = True,
) -> Array:
    if device is not None:
        raise ValueError("device argument is not supported for linspace")

    return asarray(np.linspace(start, stop, num=num, endpoint=endpoint), dtype=dtype)


def ones(shape, dtype: dtypes.CoreType | dtypes.StructType | None = None, device=None):
    dtype = dtypes.float64 if dtype is None else dtype
    return full(shape, 1, dtype=dtype)


def ones_like(x, dtype: dtypes.StructType | dtypes.CoreType | None = None, device=None):
    return full_like(x, 1, dtype=dtype)


def tril(x, *, k: int = 0) -> Array:
    return _from_corearray(
        opx.trilu(x._core(), k=asarray(k, dtype=dtypes.int64)._core(), upper=0)
    )


def triu(x, *, k: int = 0) -> Array:
    return _from_corearray(
        opx.trilu(x._core(), k=asarray(k, dtype=dtypes.int64)._core(), upper=1)
    )


def zeros(shape, dtype: dtypes.CoreType | dtypes.StructType | None = None, device=None):
    dtype = dtypes.float64 if dtype is None else dtype
    return full(shape, 0, dtype=dtype)


def zeros_like(
    x, dtype: dtypes.CoreType | dtypes.StructType | None = None, device=None
):
    return full_like(x, 0, dtype=dtype)


# data_type.py


def astype(x: Array, dtype, copy=True):
    if isinstance(dtype, np.dtype):
        dtype = dtypes.from_numpy_dtype(dtype)
    if x.dtype == dtype:
        return x if not copy else x.copy()
    if isinstance(x.dtype, CastMixin):
        try:
            out = x.dtype._cast_to(x, dtype)
            if copy:
                return out
            else:
                x._set(out)
                return x
        except CastError:
            pass
    if isinstance(dtype, CastMixin):
        try:
            out = dtype._cast_from(x)
            if copy:
                return out
            else:
                x._set(out)
                return x
        except CastError:
            pass
    raise CastError(f"Cannot cast {x.dtype} to {dtype}")


def can_cast(*args, **kwargs):
    return np.can_cast(*args, **kwargs)


def finfo(dtype):
    return dtypes.get_finfo(dtype)


def iinfo(dtype):
    return dtypes.get_iinfo(dtype)


def isdtype(dtype, kind):
    if isinstance(kind, str):
        if kind == "bool":
            return dtype == dtypes.bool
        elif kind == "signed integer":
            return dtype in (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64)
        elif kind == "unsigned integer":
            return dtype in (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
        elif kind == "integral":
            return isinstance(dtype, dtypes.Integral)
        elif kind == "real floating":
            return isinstance(dtype, dtypes.Floating)
        elif kind == "complex floating":
            raise ValueError("complex floating is not supported")
        elif kind == "numeric":
            return isinstance(dtype, dtypes.Numerical)
    elif isinstance(kind, dtypes.CoreType):
        return dtype == kind
    elif isinstance(kind, tuple):
        return any(isdtype(dtype, k) for k in kind)
    else:
        raise TypeError(f"kind must be a string or a dtype, not {type(kind)}")


def result_type(
    *objects: dtypes.CoreType | dtypes.StructType | Array,
) -> dtypes.CoreType | dtypes.StructType:
    """Find the common data type for the provided objects."""
    observed_dtypes = {obj.dtype if isinstance(obj, Array) else obj for obj in objects}
    nullable = False
    np_dtypes = []
    for dtype in observed_dtypes:
        if isinstance(dtype, dtypes.StructType):
            if isinstance(dtype, _NullableCore):
                nullable = True
                np_dtypes.append(dtype.values.to_numpy_dtype())
            else:
                raise TypeError("result_type is not defined for arbitrary Struct types")
        else:
            np_dtypes.append(dtype.to_numpy_dtype())

    ret_dtype = dtypes.from_numpy_dtype(np.result_type(*np_dtypes))
    if nullable:
        return dtypes.promote_nullable(ret_dtype)
    else:
        return ret_dtype


# elementwise.py


def abs(x):
    return _unary("abs", x)


def acos(x):
    return _unary("acos", x)


def acosh(x):
    return _unary("acosh", x)


def add(x, y):
    return _binary("add", x, y)


def asin(x):
    return _unary("asin", x)


def asinh(x):
    return _unary("asinh", x)


def atan(x):
    return _unary("atan", x)


def atan2(y, x):
    return _binary("atan2", y, x)


def atanh(x):
    return _unary("atanh", x)


def bitwise_and(x, y):
    return _binary("bitwise_and", x, y)


def bitwise_left_shift(x, y):
    return _binary("bitwise_left_shift", x, y)


def bitwise_invert(x):
    return _unary("bitwise_invert", x)


def bitwise_or(x, y):
    return _binary("bitwise_or", x, y)


def bitwise_right_shift(x, y):
    return _binary("bitwise_right_shift", x, y)


def bitwise_xor(x, y):
    return _binary("bitwise_xor", x, y)


def ceil(x):
    return _unary("ceil", x)


def cos(x):
    return _unary("cos", x)


def cosh(x):
    return _unary("cosh", x)


def divide(x, y):
    return _binary("divide", x, y)


def equal(x, y):
    return _binary("equal", x, y)


def _binary(func_name, x, y):
    x_dtype = asarray(x).dtype
    y_dtype = asarray(y).dtype
    if (out := getattr(x_dtype._ops, func_name)(x, y)) is not NotImplemented:
        return out
    if (out := getattr(y_dtype._ops, func_name)(x, y)) is not NotImplemented:
        return out
    raise TypeError(
        f"Unsupported operand type(s) for {func_name}: '{x_dtype}' and '{y_dtype}'"
    )


def _unary(func_name, x):
    x = asarray(x)
    if (out := getattr(x.dtype._ops, func_name)(x)) is not NotImplemented:
        return out
    raise TypeError(f"Unsupported operand type for {func_name}: '{x.dtype}'")


def exp(x):
    return _unary("exp", x)


def expm1(x):
    return subtract(exp(x), 1)


def floor(x):
    return _unary("floor", x)


def floor_divide(x, y):
    return _binary("floor_divide", x, y)


def greater(x, y):
    return _binary("greater", x, y)


def greater_equal(x, y):
    return _binary("greater_equal", x, y)


def isfinite(x):
    return logical_not(isinf(x))


def isinf(x):
    return _unary("isinf", x)


def isnan(x):
    return _unary("isnan", x)


def less(x, y):
    return _binary("less", x, y)


def less_equal(x, y):
    return _binary("less_equal", x, y)


def log(x):
    return _unary("log", x)


def log1p(x):
    return log(x + 1)


def log2(x):
    return log(x) / np.log(2)


def log10(x):
    return log(x) / np.log(10)


def logaddexp(x, y):
    return log(exp(x) + exp(y))


def logical_and(x, y):
    return _binary("logical_and", x, y)


def logical_not(x):
    return _unary("logical_not", x)


def logical_or(x, y):
    return _binary("logical_or", x, y)


def logical_xor(x, y):
    return _binary("logical_xor", x, y)


def multiply(x, y):
    return _binary("multiply", x, y)


def negative(x):
    return _unary("negative", x)


def not_equal(x, y):
    return _binary("not_equal", x, y)


def positive(x):
    return _unary("positive", x)


def pow(x, y):
    return _binary("pow", x, y)


def remainder(x, y):
    return _binary("remainder", x, y)


def round(x):
    return _unary("round", x)


def sign(x):
    return _unary("sign", x)


def sin(x):
    return _unary("sin", x)


def sinh(x):
    return _unary("sinh", x)


def square(x):
    return multiply(x, x)


def sqrt(x):
    return _unary("sqrt", x)


def subtract(x, y):
    return _binary("subtract", x, y)


def tan(x):
    return _unary("tan", x)


def tanh(x):
    return _unary("tanh", x)


def trunc(x):
    return _unary("trunc", x)


# linalg.py


def matmul(x, y):
    return _binary("matmul", x, y)


def matrix_transpose(x):
    return permute_dims(x, list(range(x.ndim - 2)) + [x.ndim - 1, x.ndim - 2])


# indexing.py


def take(x, indices, axis=None):
    x = asarray(x)
    if (out := x.dtype._ops.take(x, indices, axis)) is not NotImplemented:
        return out
    if axis is None:
        axis = 0
    if isinstance(indices, Array):
        indices = indices._core()
    else:
        indices = opx.const(indices, dtype=dtypes.int64)
    return x._transmute(lambda corearray: opx.gather(corearray, indices, axis=axis))


# manipulation.py


# TODO: onnx shape inference doesn't work for 2 empty arrays
def broadcast_arrays(*arrays):
    if not arrays:
        return []
    it = iter(arrays)

    def numeric_like(x):
        if isinstance(x.dtype, dtypes.NullableNumerical):
            return x
        else:
            return zeros_like(x)

    ret = numeric_like(next(it))
    while (x := next(it, None)) is not None:
        ret = ret + numeric_like(x)
    target_shape = _from_corearray(opx.shape(ret._core()))
    return [broadcast_to(x, target_shape) for x in arrays]


def broadcast_to(x, shape):
    x = asarray(x)
    if (out := x.dtype._ops.broadcast_to(x, shape)) is not NotImplemented:
        return out
    shape = asarray(shape, dtype=dtypes.int64)._core()
    return x._transmute(lambda corearray: opx.expand(corearray, shape))


# TODO: onnxruntime doesn't work for 2 empty arrays of integer type
def concat(arrays, axis=None):
    if axis is None:
        arrays = [reshape(x, [-1]) for x in arrays]
        axis = 0
    arrays = promote(*arrays)
    # TODO: implement this as a layout operation agnostic of input types
    return _from_corearray(opx.concat([array._core() for array in arrays], axis=axis))


def expand_dims(x, axis=0):
    x = asarray(x)
    if (out := x.dtype._ops.expand_dims(x, axis)) is not NotImplemented:
        return out
    return x._transmute(
        lambda corearray: opx.unsqueeze(
            corearray, axes=opx.const([axis], dtype=dtypes.int64)
        )
    )


def flip(x, axis=None):
    if (out := x.dtype._ops.flip(x, axis)) is not NotImplemented:
        return out

    if x.ndim == 0:
        return x.copy()

    if axis is None:
        axis = range(x.ndim)
    elif not isinstance(axis, Iterable):
        axis = [axis]

    axis = [x.ndim + ax if ax < 0 else ax for ax in axis]

    index = tuple(
        slice(None, None, None) if i not in axis else (slice(None, None, -1))
        for i in range(0, x.ndim)
    )

    return x[index]


def permute_dims(x, axes):
    if (out := x.dtype._ops.permute_dims(x, axes)) is not NotImplemented:
        return out
    return x._transmute(lambda corearray: opx.transpose(corearray, perm=axes))


def reshape(x, shape, *, copy=None):
    if (
        isinstance(shape, (list, tuple))
        and len(shape) == x.ndim == 1
        and shape[0] == -1
    ):
        return x.copy()
    else:
        x = asarray(x)
        if (out := x.dtype._ops.reshape(x, shape, copy=copy)) is not NotImplemented:
            return out
        shape = asarray(shape, dtype=dtypes.int64)._core()
        out = x._transmute(lambda corearray: opx.reshape(corearray, shape))
        if copy is False:
            return x._set(out)
        return out


def roll(x, shift, axis=None):
    x = asarray(x)
    if (out := x.dtype._ops.roll(x, shift, axis)) is not NotImplemented:
        return out
    if not isinstance(shift, Sequence):
        shift = [shift]

    old_shape = x.shape

    if axis is None:
        x = reshape(x, [-1])
        axis = 0

    if not isinstance(axis, Sequence):
        axis = [axis]

    if len(shift) != len(axis):
        raise ValueError("shift and axis must have the same length")

    for sh, ax in zip(shift, axis):
        len_single = opx.gather(
            asarray(x.shape, dtype=dtypes.int64)._core(), opx.const(ax)
        )
        shift_single = opx.add(opx.const(-sh, dtype=dtypes.int64), len_single)
        # Find the needed element index and then gather from it
        range = opx.cast(
            opx.range(
                opx.const(0, dtype=len_single.dtype),
                len_single,
                opx.const(1, dtype=len_single.dtype),
            ),
            to=dtypes.int64,
        )
        new_indices = opx.mod(opx.add(range, shift_single), len_single)
        x = take(x, _from_corearray(new_indices), axis=ax)

    return reshape(x, old_shape)


def squeeze(x, axis):
    x = asarray(x)
    if (out := x.dtype._ops.squeeze(x, axis)) is not NotImplemented:
        return out
    axis = [axis] if not isinstance(axis, Sequence) else axis
    if len(axis) == 0:
        return x.copy()
    axes = opx.const(axis, dtype=dtypes.int64)
    return x._transmute(lambda corearray: opx.squeeze(corearray, axes=axes))


def stack(arrays, axis=0):
    arrays = [expand_dims(x, axis=axis) for x in arrays]
    return concat(arrays, axis=axis)


# searching.py


def argmax(x, axis=None, keepdims=False):
    if (
        out := x.dtype._ops.argmax(x, axis=axis, keepdims=keepdims)
    ) is not NotImplemented:
        return out
    raise TypeError(f"Unsupported operand type for argmax: '{x.dtype}'")


def argmin(x, axis=None, keepdims=False):
    if (
        out := x.dtype._ops.argmax(x, axis=axis, keepdims=keepdims)
    ) is not NotImplemented:
        return out
    raise TypeError(f"Unsupported operand type for argmin: '{x.dtype}'")


def nonzero(x):
    return _unary("nonzero", x)


def searchsorted(
    x1, x2, *, side: Literal["left", "right"] = "left", sorter: Array | None = None
):
    if (
        out := x1.dtype._ops.searchsorted(x1, x2, side=side, sorter=sorter)
    ) is not NotImplemented:
        return out
    raise TypeError(f"Unsupported operand type for searchsorted: '{x1.dtype}'")


def where(condition, x, y):
    if (out := condition.dtype._ops.where(condition, x, y)) is not NotImplemented:
        return out
    raise TypeError(f"Unsupported condition dtype for where: '{condition.dtype}'")


# set.py
def unique_all(x: Array):
    return _unary("unique_all", x)


def unique_counts(x):
    ret = namedtuple("ret", ["values", "counts"])
    ret_all = unique_all(x)
    return ret(values=ret_all.values, counts=ret_all.counts)


def unique_inverse(x):
    ret = namedtuple("ret", ["values", "inverse_indices"])
    ret_all = unique_all(x)
    return ret(values=ret_all.values, inverse_indices=ret_all.inverse_indices)


def unique_values(x):
    return unique_all(x).values


# sorting.py


def argsort(x, *, axis=-1, descending=False, stable=True):
    if (
        out := x.dtype._ops.argsort(x, axis=axis, descending=descending, stable=stable)
    ) is not NotImplemented:
        return out
    raise TypeError(f"Unsupported operand type for argsort: '{x.dtype}'")


def sort(x, *, axis=-1, descending=False, stable=True):
    if (
        out := x.dtype._ops.sort(x, axis=axis, descending=descending, stable=stable)
    ) is not NotImplemented:
        return out
    raise TypeError(f"Unsupported operand type for sort: '{x.dtype}'")


# statistical.py
def cumulative_sum(
    x,
    *,
    axis: int | None = None,
    dtype: CoreType | StructType | None = None,
    include_initial: bool = False,
):
    if (
        out := x.dtype._ops.cumulative_sum(
            x, axis=axis, dtype=dtype, include_initial=include_initial
        )
    ) is not NotImplemented:
        return out
    raise TypeError(f"Unsupported operand type for cumulative_sum: '{x.dtype}'")


def max(x, *, axis=None, keepdims: bool = False):
    if (out := x.dtype._ops.max(x, axis=axis, keepdims=keepdims)) is not NotImplemented:
        return out
    raise TypeError(f"Unsupported operand type for max: '{x.dtype}'")


def mean(x, *, axis=None, keepdims: bool = False):
    if (
        out := x.dtype._ops.mean(x, axis=axis, keepdims=keepdims)
    ) is not NotImplemented:
        return out
    raise TypeError(f"Unsupported operand type for max: '{x.dtype}'")


def min(x, *, axis=None, keepdims: bool = False):
    if (out := x.dtype._ops.min(x, axis=axis, keepdims=keepdims)) is not NotImplemented:
        return out
    raise TypeError(f"Unsupported operand type for max: '{x.dtype}'")


def prod(
    x, *, axis=None, dtype: CoreType | StructType | None = None, keepdims: bool = False
):
    if (
        out := x.dtype._ops.prod(x, dtype=dtype, axis=axis, keepdims=keepdims)
    ) is not NotImplemented:
        return out
    raise TypeError(f"Unsupported operand type for prod: '{x.dtype}'")


def clip(x, *, min=None, max=None):
    if (out := x.dtype._ops.clip(x, min=min, max=max)) is not NotImplemented:
        return out
    raise TypeError(f"Unsupported operand type for clip: '{x.dtype}'")


def std(
    x,
    axis=None,
    keepdims: bool = False,
    correction=0.0,
    dtype: dtypes.StructType | dtypes.CoreType | None = None,
):
    dtype = x.dtype if dtype is None else dtype
    return sqrt(
        var(x, axis=axis, keepdims=keepdims, correction=correction, dtype=dtype)
    )


def sum(
    x,
    *,
    axis=None,
    dtype: dtypes.StructType | dtypes.CoreType | None = None,
    keepdims: bool = False,
):
    if (
        out := x.dtype._ops.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)
    ) is not NotImplemented:
        return out
    raise TypeError(f"Unsupported operand type for sum: '{x.dtype}'")


def var(
    x,
    *,
    axis=None,
    keepdims: bool = False,
    correction=0.0,
    dtype: dtypes.CoreType | dtypes.StructType | None = None,
):
    dtype = x.dtype if dtype is None else dtype
    if not isinstance(dtype, CoreType):
        raise TypeError("var is not supported for non-core types")
    # We keepdims, so it is still broadcastable to x
    mean_ = mean(x, axis=axis, keepdims=True).astype(dtype)

    return sum(square(x - mean_), axis=axis, keepdims=keepdims).astype(dtype) / (
        sum(full_like(x, 1), axis=axis, keepdims=keepdims).astype(dtype) - correction
    )


# utility.py


def all(x, *, axis=None, keepdims: bool = False):
    if isinstance(x.dtype, dtypes._NullableCore):
        x = where(x.null, True, x.values)
    if functools.reduce(operator.mul, x._static_shape, 1) == 0:
        return asarray(True, dtype=dtypes.bool)
    return min(x.astype(dtypes.int8), axis=axis, keepdims=keepdims).astype(dtypes.bool)


def any(x, *, axis=None, keepdims: bool = False):
    if isinstance(x.dtype, dtypes._NullableCore):
        x = where(x.null, False, x.values)
    if functools.reduce(operator.mul, x._static_shape, 1) == 0:
        return asarray(False, dtype=dtypes.bool)
    return max(x.astype(dtypes.int8), axis=axis, keepdims=keepdims).astype(dtypes.bool)


__all__ = [
    "arange",
    "asarray",
    "empty",
    "empty_like",
    "eye",
    "full",
    "full_like",
    "linspace",
    "ones",
    "ones_like",
    "tril",
    "triu",
    "zeros",
    "zeros_like",
    "astype",
    "broadcast_arrays",
    "broadcast_to",
    "can_cast",
    "finfo",
    "iinfo",
    "result_type",
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_invert",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "ceil",
    "cos",
    "cosh",
    "divide",
    "equal",
    "exp",
    "expm1",
    "floor",
    "floor_divide",
    "greater",
    "greater_equal",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "multiply",
    "negative",
    "not_equal",
    "positive",
    "pow",
    "remainder",
    "round",
    "sign",
    "sin",
    "sinh",
    "square",
    "sqrt",
    "subtract",
    "tan",
    "tanh",
    "trunc",
    "matmul",
    "matrix_transpose",
    "concat",
    "expand_dims",
    "flip",
    "permute_dims",
    "reshape",
    "roll",
    "squeeze",
    "stack",
    "argmax",
    "argmin",
    "nonzero",
    "searchsorted",
    "where",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "argsort",
    "sort",
    "cumulative_sum",
    "max",
    "mean",
    "min",
    "prod",
    "clip",
    "std",
    "sum",
    "var",
    "all",
    "any",
    "take",
]

for func_name in __all__:
    locals()[
        func_name
    ].__doc__ = f"``{func_name}`` is inherited from the Array API standard. Documentation can be found `here <https://data-apis.org/array-api/latest/API_specification/generated/array_api.{func_name}.html#{func_name}>`__."
