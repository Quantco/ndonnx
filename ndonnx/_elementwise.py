# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause
"""Element-wise free functions.

Each function directly dispatches to the inner typed array.
"""

import builtins
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

import numpy as np

import ndonnx as ndx
from ndonnx import Array, DType
from ndonnx._array_tyarray_interop import unwrap_tyarray
from ndonnx._typed_array import funcs as tyfuncs
from ndonnx._typed_array.types import Bool, Floating, Integer

F = TypeVar("F", bound=Callable[..., Array])


def _ensure_array_in_args(fn: F) -> F:
    """Decorator for element-wise binary functions."""

    @wraps(fn)
    def wrapped(a, b) -> Array:
        if isinstance(a, Array) or isinstance(b, Array):
            return fn(a, b)
        raise TypeError(
            f"at least one argument to '{fn.__name__}' must be of type 'ndonnx.Array'"
        )

    return wrapped  # type: ignore


@_ensure_array_in_args
def add(a: Array | Integer | Floating, b: Array | Integer | Floating) -> Array:
    return ndx.asarray(a + b)


def abs(array: Array, /) -> Array:
    return Array._from_tyarray(builtins.abs(array._tyarray))


def acos(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.acos())


def acosh(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.acosh())


def asin(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.asin())


def asinh(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.asinh())


def atan(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.atan())


@_ensure_array_in_args
def atan2(x1: Array | Integer | Floating, x2: Array | Integer | Floating, /) -> Array:
    # Requires special operator to meet standards precision requirements
    # TODO: Add upstream tracking issue
    raise NotImplementedError


def atanh(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.atanh())


@_ensure_array_in_args
def bitwise_and(x1: Array | Integer | Bool, x2: Array | Integer | Bool, /) -> Array:
    return ndx.asarray(x1 & x2)


@_ensure_array_in_args
def bitwise_left_shift(x1: Array | Integer, x2: Array | Integer, /) -> Array:
    return ndx.asarray(x1 << x2)


def bitwise_invert(x: Array, /) -> Array:
    return ~x


@_ensure_array_in_args
def bitwise_or(x1: Array | Integer | Bool, x2: Array | Integer | Bool, /) -> Array:
    return ndx.asarray(x1 | x2)


@_ensure_array_in_args
def bitwise_right_shift(x1: Array | Integer, x2: Array | Integer, /) -> Array:
    return ndx.asarray(x1 >> x2)


@_ensure_array_in_args
def bitwise_xor(x1: Array | Integer | Bool, x2: Array | Integer | Bool, /) -> Array:
    return ndx.asarray(x1 ^ x2)


def ceil(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.ceil())


def clip(
    x: Array,
    /,
    min: None | Integer | Floating | Array = None,
    max: None | Integer | Floating | Array = None,
) -> Array:
    min_max: list[int | float | DType] = []
    if min is not None:
        if isinstance(min, Array):
            min_max.append(min.dtype)
        elif isinstance(min, np.generic):
            min_max.append(ndx.from_numpy_dtype(np.dtype(min)))
        else:
            min_max.append(min)
    if max is not None:
        if isinstance(max, Array):
            min_max.append(max.dtype)
        elif isinstance(max, np.generic):
            min_max.append(ndx.from_numpy_dtype(np.dtype(max)))
        else:
            min_max.append(max)
    dtype = tyfuncs.result_type(x.dtype, *min_max)

    min_ = None if min is None else ndx.asarray(min, dtype=dtype)._tyarray
    max_ = None if max is None else ndx.asarray(max, dtype=dtype)._tyarray
    return Array._from_tyarray(x._tyarray.astype(dtype).clip(min=min_, max=max_))


def cos(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.cos())


def cosh(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.cosh())


@_ensure_array_in_args
def copysign(
    x1: Array | Integer | Floating, x2: Array | Integer | Floating, /
) -> Array:
    raise NotImplementedError


@_ensure_array_in_args
def divide(x1: Array | Integer | Floating, x2: Array | Integer | Floating, /) -> Array:
    return ndx.asarray(x1 / x2)


def exp(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.exp())


def expm1(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.expm1())


@_ensure_array_in_args
def equal(
    x1: Array | Integer | Floating | Bool, x2: Array | Integer | Floating | Bool, /
) -> Array:
    return ndx.asarray(x1 == x2)


def floor(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.floor())


@_ensure_array_in_args
def floor_divide(
    x1: Array | Integer | Floating, x2: Array | Integer | Floating, /
) -> Array:
    return ndx.asarray(x1 // x2)


@_ensure_array_in_args
def greater(x1: Array | Integer | Floating, x2: Array | Integer | Floating, /) -> Array:
    return ndx.asarray(x1 > x2)


@_ensure_array_in_args
def greater_equal(
    x1: Array | Integer | Floating, x2: Array | Integer | Floating, /
) -> Array:
    return ndx.asarray(x1 >= x2)


@_ensure_array_in_args
def hypot(x1: Array | Integer | Floating, x2: Array | Integer | Floating, /) -> Array:
    raise NotImplementedError


def isfinite(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.isfinite())


def isinf(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.isinf())


def isnan(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.isnan())


@_ensure_array_in_args
def less(x1: Array | Integer | Floating, x2: Array | Integer | Floating, /) -> Array:
    return ndx.asarray(x1 < x2)


@_ensure_array_in_args
def less_equal(
    x1: Array | Integer | Floating, x2: Array | Integer | Floating, /
) -> Array:
    return ndx.asarray(x1 <= x2)


def log(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.log())


def log1p(x: Array, /) -> Array:
    raise NotImplementedError


def log2(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.log2())


def log10(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.log10())


@_ensure_array_in_args
def logaddexp(
    x1: Array | Integer | Floating, x2: Array | Integer | Floating, /
) -> Array:
    return Array._from_tyarray(
        tyfuncs.logaddexp(unwrap_tyarray(x1), unwrap_tyarray(x2))
    )


@_ensure_array_in_args
def logical_and(x1: Array | Bool, x2: Array | Bool, /) -> Array:
    return Array._from_tyarray(
        tyfuncs.logical_and(unwrap_tyarray(x1), unwrap_tyarray(x2))
    )


def logical_not(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.logical_not())


@_ensure_array_in_args
def logical_or(x1: Array | Bool, x2: Array | Bool, /) -> Array:
    return Array._from_tyarray(
        tyfuncs.logical_or(unwrap_tyarray(x1), unwrap_tyarray(x2))
    )


@_ensure_array_in_args
def logical_xor(x1: Array | Bool, x2: Array | Bool, /) -> Array:
    return Array._from_tyarray(
        tyfuncs.logical_xor(unwrap_tyarray(x1), unwrap_tyarray(x2))
    )


@_ensure_array_in_args
def maximum(x1: Array | Integer | Floating, x2: Array | Integer | Floating, /) -> Array:
    return Array._from_tyarray(tyfuncs.maximum(unwrap_tyarray(x1), unwrap_tyarray(x2)))


@_ensure_array_in_args
def minimum(x1: Array | Integer | Floating, x2: Array | Integer | Floating, /) -> Array:
    return Array._from_tyarray(tyfuncs.minimum(unwrap_tyarray(x1), unwrap_tyarray(x2)))


@_ensure_array_in_args
def multiply(
    x1: Array | Integer | Floating, x2: Array | Integer | Floating, /
) -> Array:
    return ndx.asarray(x1 * x2)


def negative(x: Array, /) -> Array:
    return Array._from_tyarray(-x._tyarray)


@_ensure_array_in_args
def nextafter(
    x1: Array | Integer | Floating, x2: Array | Integer | Floating, /
) -> Array:
    # Requires special ONNX operator
    # TODO: Add upstream tracking issue
    raise NotImplementedError


@_ensure_array_in_args
def not_equal(
    x1: Array | Integer | Floating | Bool, x2: Array | Integer | Floating | Bool, /
) -> Array:
    return ndx.asarray(x1 != x2)


def positive(x: Array, /) -> Array:
    return Array._from_tyarray(+x._tyarray)


@_ensure_array_in_args
def pow(x1: Array | Integer | Floating, x2: Array | Integer | Floating, /) -> Array:
    return ndx.asarray(x1**x2)


def real(x: Array, /) -> Array:
    raise NotImplementedError


def reciprocal(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.reciprocal())


@_ensure_array_in_args
def remainder(
    x1: Array | Integer | Floating, x2: Array | Integer | Floating, /
) -> Array:
    return ndx.asarray(x1 % x2)


def round(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.round())


def sign(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.sign())


def signbit(x: Array, /) -> Array:
    raise NotImplementedError


def sin(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.sin())


def sinh(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.sinh())


def square(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray * x._tyarray)


def sqrt(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.sqrt())


@_ensure_array_in_args
def subtract(
    x1: Array | Integer | Floating, x2: Array | Integer | Floating, /
) -> Array:
    return ndx.asarray(x1 - x2)


def tan(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.tan())


def tanh(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.tanh())


def trunc(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.trunc())
