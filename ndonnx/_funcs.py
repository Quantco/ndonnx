# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
from collections import namedtuple
from typing import Literal

import numpy as np
import numpy.typing as npt

import ndonnx._data_types as dtypes
from ndonnx._data_types import CastError, CastMixin, CoreType, NullableCore
from ndonnx._data_types.structtype import StructType
from ndonnx.additional import shape

from . import _opset_extensions as opx
from ._array import Array, _from_corearray
from ._utility import (
    promote,
)


class UnsupportedOperationError(TypeError): ...


# creation.py


def arange(
    start,
    stop=None,
    step=1,
    *,
    dtype: dtypes.CoreType | dtypes.StructType | None = None,
    device=None,
):
    if dtype is None:
        if builtins.all(
            isinstance(asarray(x).dtype, dtypes.Integral)
            for x in [start, stop, step]
            if x is not None
        ):
            dtype = dtypes.int64
        else:
            dtype = dtypes.float64

    out = dtype._ops.arange(start, stop, step, dtype=dtype, device=device)
    if out is NotImplemented:
        raise UnsupportedOperationError(
            f"Unsupported operand type for arange: '{dtype}'"
        )
    return out


def asarray(
    x: npt.ArrayLike | Array,
    dtype: dtypes.CoreType | dtypes.StructType | None = None,
    copy=None,
    device=None,
) -> Array:
    if not isinstance(x, Array):
        eager_value = np.asanyarray(
            x,
            dtype=(
                dtype.to_numpy_dtype() if isinstance(dtype, dtypes.CoreType) else None
            ),
        )
        if dtype is None:
            dtype = dtypes.from_numpy_dtype(eager_value.dtype)
            if isinstance(eager_value, np.ma.masked_array):
                dtype = dtypes.into_nullable(dtype)

        ret = dtype._ops.make_array(
            shape=eager_value.shape,
            dtype=dtype,
            eager_value=eager_value,
        )
        if ret is NotImplemented:
            raise UnsupportedOperationError(
                f"Unsupported operand type for asarray: '{dtype}'"
            )
    else:
        ret = x.copy() if copy is True else x

    if dtype is not None:
        ret = astype(ret, dtype=dtype, copy=copy)

    return ret


def empty(shape, dtype: dtypes.CoreType | dtypes.StructType | None = None, device=None):
    dtype = dtype or dtypes.float64
    if (out := dtype._ops.empty(shape, dtype=dtype)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for empty: '{dtype}'")


def empty_like(
    x, dtype: dtypes.CoreType | dtypes.StructType | None = None, device=None
):
    dtype = dtype or x.dtype
    if (out := dtype._ops.empty_like(x, dtype=dtype)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(
        f"Unsupported operand type for empty_like: '{dtype}'"
    )


def eye(
    n_rows,
    n_cols=None,
    k=0,
    dtype: dtypes.StructType | dtypes.CoreType | None = None,
    device=None,
):
    dtype = dtype or dtypes.float64
    out = dtype._ops.eye(n_rows, n_cols=n_cols, k=k, dtype=dtype, device=device)
    if out is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for eye: '{dtype}'")


def full(
    shape,
    fill_value,
    *,
    dtype: dtypes.CoreType | dtypes.StructType | None = None,
    device=None,
):
    dtype = asarray(fill_value).dtype if dtype is None else dtype
    if (
        out := dtype._ops.full(shape, fill_value, dtype=dtype, device=device)
    ) is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for full: '{dtype}'")


def full_like(
    x,
    fill_value,
    dtype: dtypes.CoreType | dtypes.StructType | None = None,
    device=None,
):
    if (
        out := x.dtype._ops.full_like(x, fill_value, dtype=dtype, device=device)
    ) is not NotImplemented:
        return out
    raise UnsupportedOperationError(
        f"Unsupported operand type for full_like: `{x.dtype}`"
    )


def linspace(
    start: int | float,
    stop: int | float,
    num: int,
    *,
    dtype: dtypes.CoreType | dtypes.StructType | None = None,
    device=None,
    endpoint: bool = True,
) -> Array:
    dtype = dtypes.float64 if dtype is None else dtype
    out = dtype._ops.linspace(
        start, stop, num=num, dtype=dtype, endpoint=endpoint, device=device
    )
    if out is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for linspace: '{dtype}'")


def ones(shape, dtype: dtypes.CoreType | dtypes.StructType | None = None, device=None):
    dtype = dtypes.float64 if dtype is None else dtype
    if (out := dtype._ops.ones(shape, dtype=dtype)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for ones: '{dtype}'")


def ones_like(x, dtype: dtypes.StructType | dtypes.CoreType | None = None, device=None):
    if (out := x.dtype._ops.ones_like(x, dtype=dtype)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(
        f"Unsupported operand type for ones_like: '{x.dtype}'"
    )


def tril(x: Array, *, k: int = 0) -> Array:
    out = x.dtype._ops.tril(x, k=k)
    if out is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for tril: '{x.dtype}'")


def triu(x: Array, *, k: int = 0) -> Array:
    out = x.dtype._ops.triu(x, k=k)
    if out is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for triu: '{x.dtype}'")


def zeros(shape, dtype: dtypes.CoreType | dtypes.StructType | None = None, device=None):
    dtype = dtypes.float64 if dtype is None else dtype
    if (out := dtype._ops.zeros(shape, dtype=dtype)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for zeros: '{dtype}'")


def zeros_like(
    x, dtype: dtypes.CoreType | dtypes.StructType | None = None, device=None
):
    if (out := x.dtype._ops.zeros_like(x, dtype=dtype)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(
        f"Unsupported operand type for zeros_like: '{x.dtype}'"
    )


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


def can_cast(from_, to):
    if (out := from_._ops.can_cast(from_, to)) is not NotImplemented:
        return out
    elif (out := to._ops.can_cast(from_, to)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(
        f"Unsupported operand type for can_cast: '{from_}' and '{to}'"
    )


def finfo(dtype):
    return dtypes.get_finfo(dtype)


def iinfo(dtype):
    return dtypes.get_iinfo(dtype)


def isdtype(dtype, kind) -> bool:
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
            raise ValueError("'complex floating' is not supported")
        elif kind == "numeric":
            return isinstance(dtype, dtypes.Numerical)
    elif isinstance(kind, dtypes.CoreType):
        return dtype == kind
    elif isinstance(kind, tuple):
        return builtins.any(isdtype(dtype, k) for k in kind)
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
            if isinstance(dtype, NullableCore):
                nullable = True
                np_dtypes.append(dtype.values.to_numpy_dtype())
            else:
                raise TypeError("result_type is not defined for arbitrary Struct types")
        else:
            np_dtypes.append(dtype.to_numpy_dtype())

    ret_dtype = dtypes.from_numpy_dtype(np.result_type(*np_dtypes))
    if nullable:
        return dtypes.into_nullable(ret_dtype)
    else:
        return ret_dtype


# elementwise.py


def abs(x):
    return _unary(x.dtype._ops.abs, x)


def acos(x):
    return _unary(x.dtype._ops.acos, x)


def acosh(x):
    return _unary(x.dtype._ops.acosh, x)


def add(x, y):
    return _binary("add", x, y)


def asin(x):
    return _unary(x.dtype._ops.asin, x)


def asinh(x):
    return _unary(x.dtype._ops.asinh, x)


def atan(x):
    return _unary(x.dtype._ops.atan, x)


def atan2(y, x):
    return _binary("atan2", y, x)


def atanh(x):
    return _unary(x.dtype._ops.atanh, x)


def bitwise_and(x, y):
    return _binary("bitwise_and", x, y)


def bitwise_left_shift(x, y):
    return _binary("bitwise_left_shift", x, y)


def bitwise_invert(x):
    return _unary(x.dtype._ops.bitwise_invert, x)


def bitwise_or(x, y):
    return _binary("bitwise_or", x, y)


def bitwise_right_shift(x, y):
    return _binary("bitwise_right_shift", x, y)


def bitwise_xor(x, y):
    return _binary("bitwise_xor", x, y)


def ceil(x):
    return _unary(x.dtype._ops.ceil, x)


def cos(x):
    return _unary(x.dtype._ops.cos, x)


def cosh(x):
    return _unary(x.dtype._ops.cosh, x)


def divide(x, y):
    return _binary("divide", x, y)


def equal(x, y):
    return _binary("equal", x, y)


def exp(x):
    return _unary(x.dtype._ops.exp, x)


def expm1(x):
    return _unary(x.dtype._ops.expm1, x)


def floor(x):
    return _unary(x.dtype._ops.floor, x)


def floor_divide(x, y):
    return _binary("floor_divide", x, y)


def greater(x, y):
    return _binary("greater", x, y)


def greater_equal(x, y):
    return _binary("greater_equal", x, y)


def isfinite(x):
    return logical_not(isinf(x))


def isinf(x):
    return _unary(x.dtype._ops.isinf, x)


def isnan(x):
    return _unary(x.dtype._ops.isnan, x)


def less(x, y):
    return _binary("less", x, y)


def less_equal(x, y):
    return _binary("less_equal", x, y)


def log(x):
    return _unary(x.dtype._ops.log, x)


def log1p(x):
    return _unary(x.dtype._ops.log1p, x)


def log2(x):
    return _unary(x.dtype._ops.log2, x)


def log10(x):
    return _unary(x.dtype._ops.log10, x)


def logaddexp(x, y):
    return _binary("logaddexp", x, y)


def logical_and(x, y):
    return _binary("logical_and", x, y)


def logical_not(x):
    return _unary(x.dtype._ops.logical_not, x)


def logical_or(x, y):
    return _binary("logical_or", x, y)


def logical_xor(x, y):
    return _binary("logical_xor", x, y)


def multiply(x, y):
    return _binary("multiply", x, y)


def negative(x):
    return _unary(x.dtype._ops.negative, x)


def not_equal(x, y):
    return _binary("not_equal", x, y)


def positive(x):
    return _unary(x.dtype._ops.positive, x)


def pow(x, y):
    return _binary("pow", x, y)


def remainder(x, y):
    return _binary("remainder", x, y)


def round(x):
    return _unary(x.dtype._ops.round, x)


def sign(x):
    return _unary(x.dtype._ops.sign, x)


def sin(x):
    return _unary(x.dtype._ops.sin, x)


def sinh(x):
    return _unary(x.dtype._ops.sinh, x)


def square(x):
    return multiply(x, x)


def sqrt(x):
    return _unary(x.dtype._ops.sqrt, x)


def subtract(x, y):
    return _binary("subtract", x, y)


def tan(x):
    return _unary(x.dtype._ops.tan, x)


def tanh(x):
    return _unary(x.dtype._ops.tanh, x)


def trunc(x):
    return _unary(x.dtype._ops.trunc, x)


# linalg.py


def matmul(x, y):
    return _binary("matmul", x, y)


def matrix_transpose(x):
    return _unary(x.dtype._ops.matrix_transpose, x)


# indexing.py


def take(x, indices, axis=None):
    if (out := x.dtype._ops.take(x, indices, axis)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for take: '{x.dtype}'")


# manipulation.py


# TODO: onnx shape inference doesn't work for 2 empty arrays
# TODO: what is the appropriate strategy to dispatch? (iterate over the inputs and keep trying is reasonable but it can
# change the outcome based on order if poorly implemented)


def broadcast_arrays(*arrays):
    if not arrays:
        return []
    it = iter(arrays)

    def numeric_like(x):
        if isinstance(x.dtype, (dtypes.NullableNumerical, dtypes.Numerical)):
            return x
        else:
            return zeros_like(x, dtype=dtypes.int64)

    ret = numeric_like(next(it))
    while (x := next(it, None)) is not None:
        ret = ret + numeric_like(x)
    target_shape = shape(ret)
    return [broadcast_to(x, target_shape) for x in arrays]


def broadcast_to(x, shape):
    if (out := x.dtype._ops.broadcast_to(x, shape)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(
        f"Unsupported operand type for broadcast_to: '{x.dtype}'"
    )


# TODO: onnxruntime doesn't work for 2 empty arrays of integer type
# TODO: what is the appropriate strategy to dispatch? (iterate over the inputs and keep trying is reasonable but it can
# change the outcome based on order if poorly implemented)
def concat(arrays, /, *, axis: int | None = 0):
    if axis is None:
        arrays = [reshape(x, [-1]) for x in arrays]
        axis = 0
    if builtins.any(not isinstance(array.dtype, CoreType) for array in arrays):
        raise UnsupportedOperationError(
            "concat is not supported for non-core types at this time"
        )
    arrays = promote(*arrays)
    return _from_corearray(opx.concat([array._core() for array in arrays], axis=axis))


def expand_dims(x, axis=0):
    if (out := x.dtype._ops.expand_dims(x, axis)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(
        f"Unsupported operand type for expand_dims: '{x.dtype}'"
    )


def flip(x, axis=None):
    if (out := x.dtype._ops.flip(x, axis=axis)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for flip: '{x.dtype}'")


def permute_dims(x, axes):
    if (out := x.dtype._ops.permute_dims(x, axes)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(
        f"Unsupported operand type for permute_dims: '{x.dtype}'"
    )


def reshape(x, shape, *, copy=None):
    if (out := x.dtype._ops.reshape(x, shape, copy=copy)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(
        f"Unsupported operand type for reshape: '{x.dtype}'"
    )


def roll(x, shift, axis=None):
    if (out := x.dtype._ops.roll(x, shift, axis)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for roll: '{x.dtype}'")


def squeeze(x, axis):
    if (out := x.dtype._ops.squeeze(x, axis)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(
        f"Unsupported operand type for squeeze: '{x.dtype}'"
    )


def stack(arrays, axis=0):
    arrays = [expand_dims(x, axis=axis) for x in arrays]
    return concat(arrays, axis=axis)


# searching.py


def argmax(x, axis=None, keepdims=False):
    if (
        out := x.dtype._ops.argmax(x, axis=axis, keepdims=keepdims)
    ) is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for argmax: '{x.dtype}'")


def argmin(x, axis=None, keepdims=False):
    if (
        out := x.dtype._ops.argmin(x, axis=axis, keepdims=keepdims)
    ) is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for argmin: '{x.dtype}'")


def nonzero(x) -> tuple[Array, ...]:
    return _unary(x.dtype._ops.nonzero, x)


def searchsorted(
    x1, x2, *, side: Literal["left", "right"] = "left", sorter: Array | None = None
):
    if (
        out := x1.dtype._ops.searchsorted(x1, x2, side=side, sorter=sorter)
    ) is not NotImplemented:
        return out
    raise UnsupportedOperationError(
        f"Unsupported operand type for searchsorted: '{x1.dtype}'"
    )


def where(condition, x, y):
    condition, x, y = map(asarray, (condition, x, y))
    if condition.dtype not in (dtypes.bool, dtypes.nbool):
        raise TypeError(
            f"condition must be a boolean array, received {condition.dtype}"
        )
    if (out := x.dtype._ops.where(condition, x, y)) is not NotImplemented:
        return out
    if (out := y.dtype._ops.where(condition, x, y)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(
        f"Unsupported dtypes for where arguments: '{x.dtype}' and '{y.dtype}'"
    )


# set.py
def unique_all(x: Array):
    if (out := x.dtype._ops.unique_all(x)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for unique: '{x.dtype}'")


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
    raise UnsupportedOperationError(
        f"Unsupported operand type for argsort: '{x.dtype}'"
    )


def sort(x, *, axis=-1, descending=False, stable=True):
    if (
        out := x.dtype._ops.sort(x, axis=axis, descending=descending, stable=stable)
    ) is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for sort: '{x.dtype}'")


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
    raise UnsupportedOperationError(
        f"Unsupported operand type for cumulative_sum: '{x.dtype}'"
    )


def max(x, *, axis=None, keepdims: bool = False):
    if (out := x.dtype._ops.max(x, axis=axis, keepdims=keepdims)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for max: '{x.dtype}'")


def mean(x, *, axis=None, keepdims: bool = False):
    if (
        out := x.dtype._ops.mean(x, axis=axis, keepdims=keepdims)
    ) is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for max: '{x.dtype}'")


def min(x, *, axis=None, keepdims: bool = False):
    if (out := x.dtype._ops.min(x, axis=axis, keepdims=keepdims)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for max: '{x.dtype}'")


def prod(
    x, *, axis=None, dtype: CoreType | StructType | None = None, keepdims: bool = False
):
    if (
        out := x.dtype._ops.prod(x, dtype=dtype, axis=axis, keepdims=keepdims)
    ) is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for prod: '{x.dtype}'")


def clip(x, /, min=None, max=None):
    if (out := x.dtype._ops.clip(x, min=min, max=max)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for clip: '{x.dtype}'")


def std(
    x,
    axis=None,
    keepdims: bool = False,
    correction=0.0,
    dtype: dtypes.StructType | dtypes.CoreType | None = None,
):
    dtype = x.dtype if dtype is None else dtype
    out = dtype._ops.std(
        x, axis=axis, keepdims=keepdims, correction=correction, dtype=dtype
    )
    if out is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for std: '{dtype}'")


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
    raise UnsupportedOperationError(f"Unsupported operand type for sum: '{x.dtype}'")


def var(
    x,
    *,
    axis=None,
    keepdims: bool = False,
    correction=0.0,
    dtype: dtypes.CoreType | dtypes.StructType | None = None,
):
    dtype = x.dtype if dtype is None else dtype
    out = dtype._ops.var(
        x, axis=axis, keepdims=keepdims, correction=correction, dtype=dtype
    )
    if out is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for var: '{dtype}'")


# utility.py


def all(x, *, axis=None, keepdims: bool = False):
    out = x.dtype._ops.all(x, axis=axis, keepdims=keepdims)
    if out is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for all: '{x.dtype}'")


def any(x, *, axis=None, keepdims: bool = False):
    out = x.dtype._ops.any(x, axis=axis, keepdims=keepdims)
    if out is not NotImplemented:
        return out
    raise UnsupportedOperationError(f"Unsupported operand type for any: '{x.dtype}'")


def _binary(func_name, x, y):
    x_dtype = asarray(x).dtype
    y_dtype = asarray(y).dtype
    if (out := getattr(x_dtype._ops, func_name)(x, y)) is not NotImplemented:
        return out
    if (out := getattr(y_dtype._ops, func_name)(x, y)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(
        f"Unsupported operand type(s) for {func_name}: '{x.dtype}' and '{y.dtype}'"
    )


def _unary(func, x):
    if (out := func(x)) is not NotImplemented:
        return out
    raise UnsupportedOperationError(
        f"Unsupported operand type for {func.__name__}: '{x.dtype}'"
    )


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
