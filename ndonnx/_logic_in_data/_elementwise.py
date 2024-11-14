# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
"""Element-wise free functions.

Each function directly dispatches to the inner typed array.
"""

import builtins

from ._array import Array
from ._typed_array import funcs as tyfuncs


def add(a: Array, b: Array) -> Array:
    return a + b


def abs(array: Array, /) -> Array:
    return Array._from_data(builtins.abs(array._data))


def acos(array: Array, /) -> Array:
    return Array._from_data(array._data.acos())


def acosh(array: Array, /) -> Array:
    return Array._from_data(array._data.acosh())


def asin(array: Array, /) -> Array:
    return Array._from_data(array._data.asin())


def asinh(array: Array, /) -> Array:
    return Array._from_data(array._data.asinh())


def atan(array: Array, /) -> Array:
    return Array._from_data(array._data.atan())


def atanh(array: Array, /) -> Array:
    return Array._from_data(array._data.atanh())


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
    return Array._from_data(array._data.ceil())


def clip(
    x: Array,
    /,
    min: None | int | float | Array = None,
    max: None | int | float | Array = None,
) -> Array:
    min_ = None if min is None else tyfuncs.astyarray(min, use_py_scalars=True)
    max_ = None if max is None else tyfuncs.astyarray(max, use_py_scalars=True)
    return Array._from_data(x._data.clip(min=min_, max=max_))


def cos(x: Array, /) -> Array:
    return Array._from_data(x._data.cos())


def cosh(x: Array, /) -> Array:
    return Array._from_data(x._data.cosh())


def copysign(x1: Array, x2: Array, /) -> Array:
    raise NotImplementedError


def divide(x1: Array, x2: Array, /) -> Array:
    return x1 / x2


def exp(array: Array, /) -> Array:
    return Array._from_data(array._data.exp())


def expm1(array: Array, /) -> Array:
    return Array._from_data(array._data.expm1())


def equal(x1: Array, x2: Array, /) -> Array:
    return x1 == x2


def floor(array: Array, /) -> Array:
    return Array._from_data(array._data.floor())


def floor_divide(x1: Array, x2: Array, /) -> Array:
    return x1 // x2


def greater(x1: Array, x2: Array, /) -> Array:
    return x1 > x2


def greater_equal(x1: Array, x2: Array, /) -> Array:
    return x1 >= x2


def hypot(x1: Array, x2: Array, /) -> Array:
    raise NotImplementedError


def isfinite(array: Array, /) -> Array:
    return Array._from_data(array._data.isfinite())


def isinf(array: Array, /) -> Array:
    return Array._from_data(array._data.isinf())


def isnan(array: Array, /) -> Array:
    return Array._from_data(array._data.isnan())


def less(x1: Array, x2: Array, /) -> Array:
    return x1 < x2


def less_equal(x1: Array, x2: Array, /) -> Array:
    return x1 <= x2


def log(x: Array, /) -> Array:
    return Array._from_data(x._data.log())


def log1p(x: Array, /) -> Array:
    raise NotImplementedError


def log2(x: Array, /) -> Array:
    return Array._from_data(x._data.log2())


def log10(x: Array, /) -> Array:
    return Array._from_data(x._data.log10())


def logaddexp(x1: Array, x2: Array, /) -> Array:
    return log(exp(x1) + exp(x2))


def logical_and(x1: Array, x2: Array, /) -> Array:
    return Array._from_data(x1._data and x2._data)


def logical_not(x: Array, /) -> Array:
    return Array._from_data(x._data.logical_not())


def logical_or(x1: Array, x2: Array, /) -> Array:
    return Array._from_data(x1._data or x2._data)


def logical_xor(x1: Array, x2: Array, /) -> Array:
    return Array._from_data(x1._data or x2._data)


def maximum(x1: Array, x2: Array, /) -> Array:
    return Array._from_data(tyfuncs.maximum(x1._data, x2._data))


def minimum(x1: Array, x2: Array, /) -> Array:
    return Array._from_data(tyfuncs.minimum(x1._data, x2._data))


def multiply(x1: Array, x2: Array, /) -> Array:
    return Array._from_data(x1._data * x2._data)


def negative(x: Array, /) -> Array:
    return Array._from_data(-x._data)


def not_equal(x1: Array, x2: Array, /) -> Array:
    return Array._from_data(x1._data != x2._data)


def positive(x: Array, /) -> Array:
    return Array._from_data(+x._data)


def pow(x1: Array, x2: Array, /) -> Array:
    return x1**x2


def real(x: Array, /) -> Array:
    raise NotImplementedError


def remainder(x1: Array, x2: Array, /) -> Array:
    return Array._from_data(x1._data % x2._data)


def round(x: Array, /) -> Array:
    return Array._from_data(x._data.round())


def sign(x: Array, /) -> Array:
    return Array._from_data(x._data.sign())


def signbit(x: Array, /) -> Array:
    raise NotImplementedError


def sin(x: Array, /) -> Array:
    return Array._from_data(x._data.sin())


def sinh(x: Array, /) -> Array:
    return Array._from_data(x._data.sinh())


def square(x: Array, /) -> Array:
    return Array._from_data(x._data * x._data)


def sqrt(x: Array, /) -> Array:
    return Array._from_data(x._data.sqrt())


def subtract(x1: Array, x2: Array, /) -> Array:
    return Array._from_data(x1._data - x2._data)


def tan(x: Array, /) -> Array:
    return Array._from_data(x._data.tan())


def tanh(x: Array, /) -> Array:
    return Array._from_data(x._data.tanh())


def trunc(x: Array, /) -> Array:
    return Array._from_data(x._data.trunc())
