# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
"""Element-wise free functions.

Each function directly dispatches to the inner typed array.
"""

import builtins

import ndonnx as ndx
from ndonnx import Array, DType

from ._typed_array import funcs as tyfuncs


def add(a: Array, b: Array) -> Array:
    return a + b


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


def atanh(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.atanh())


def bitwise_and(x1: Array, x2: Array, /) -> Array:
    return x1 & x2


def bitwise_left_shift(x1: Array, x2: Array, /) -> Array:
    return x1 << x2


def bitwise_invert(x: Array, /) -> Array:
    return ~x


def bitwise_or(x1: Array, x2: Array, /) -> Array:
    return x1 | x2


def bitwise_right_shift(x1: Array, x2: Array, /) -> Array:
    return x1 >> x2


def bitwise_xor(x1: Array, x2: Array, /) -> Array:
    return x1 ^ x2


def ceil(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.ceil())


def clip(
    x: Array,
    /,
    min: None | int | float | Array = None,
    max: None | int | float | Array = None,
) -> Array:
    min_max: list[int | float | DType] = []
    if min is not None:
        if isinstance(min, Array):
            min_max.append(min.dtype)
        else:
            min_max.append(min)
    if max is not None:
        if isinstance(max, Array):
            min_max.append(max.dtype)
        else:
            min_max.append(max)
    dtype = tyfuncs.result_type(x.dtype, *min_max)

    min_ = None if min is None else ndx.asarray(min, dtype=dtype)._tyarray
    max_ = None if max is None else ndx.asarray(max, dtype=dtype)._tyarray
    return Array._from_tyarray(x._tyarray.clip(min=min_, max=max_))


def cos(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.cos())


def cosh(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.cosh())


def copysign(x1: Array, x2: Array, /) -> Array:
    raise NotImplementedError


def divide(x1: Array, x2: Array, /) -> Array:
    return x1 / x2


def exp(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.exp())


def expm1(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.expm1())


def equal(x1: Array, x2: Array, /) -> Array:
    return x1 == x2


def floor(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.floor())


def floor_divide(x1: Array, x2: Array, /) -> Array:
    return x1 // x2


def greater(x1: Array, x2: Array, /) -> Array:
    return x1 > x2


def greater_equal(x1: Array, x2: Array, /) -> Array:
    return x1 >= x2


def hypot(x1: Array, x2: Array, /) -> Array:
    raise NotImplementedError


def isfinite(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.isfinite())


def isinf(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.isinf())


def isnan(array: Array, /) -> Array:
    return Array._from_tyarray(array._tyarray.isnan())


def less(x1: Array, x2: Array, /) -> Array:
    return x1 < x2


def less_equal(x1: Array, x2: Array, /) -> Array:
    return x1 <= x2


def log(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.log())


def log1p(x: Array, /) -> Array:
    raise NotImplementedError


def log2(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.log2())


def log10(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.log10())


def logaddexp(x1: Array, x2: Array, /) -> Array:
    return log(exp(x1) + exp(x2))


def logical_and(x1: Array, x2: Array, /) -> Array:
    return Array._from_tyarray(tyfuncs.logical_and(x1._tyarray, x2._tyarray))


def logical_not(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.logical_not())


def logical_or(x1: Array, x2: Array, /) -> Array:
    return Array._from_tyarray(tyfuncs.logical_or(x1._tyarray, x2._tyarray))


def logical_xor(x1: Array, x2: Array, /) -> Array:
    return Array._from_tyarray(tyfuncs.logical_xor(x1._tyarray, x2._tyarray))


def maximum(x1: Array, x2: Array, /) -> Array:
    return Array._from_tyarray(tyfuncs.maximum(x1._tyarray, x2._tyarray))


def minimum(x1: Array, x2: Array, /) -> Array:
    return Array._from_tyarray(tyfuncs.minimum(x1._tyarray, x2._tyarray))


def multiply(x1: Array, x2: Array, /) -> Array:
    return Array._from_tyarray(x1._tyarray * x2._tyarray)


def negative(x: Array, /) -> Array:
    return Array._from_tyarray(-x._tyarray)


def not_equal(x1: Array, x2: Array, /) -> Array:
    return Array._from_tyarray(x1._tyarray != x2._tyarray)


def positive(x: Array, /) -> Array:
    return Array._from_tyarray(+x._tyarray)


def pow(x1: Array, x2: Array, /) -> Array:
    return x1**x2


def real(x: Array, /) -> Array:
    raise NotImplementedError


def remainder(x1: Array, x2: Array, /) -> Array:
    return Array._from_tyarray(x1._tyarray % x2._tyarray)


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


def subtract(x1: Array, x2: Array, /) -> Array:
    return Array._from_tyarray(x1._tyarray - x2._tyarray)


def tan(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.tan())


def tanh(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.tanh())


def trunc(x: Array, /) -> Array:
    return Array._from_tyarray(x._tyarray.trunc())
