# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
"""Element-wise free functions.

Each function directly dispatches to the inner typed array.
"""

from ._array import Array
from ._typed_array import funcs as tyfuncs


def add(a: Array, b: Array) -> Array:
    return a + b


def abs(array: Array, /) -> Array:
    return Array._from_data(array._data.__abs__())


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
    raise NotImplementedError


def cos(x: Array, /) -> Array:
    raise NotImplementedError


def cosh(x: Array, /) -> Array:
    raise NotImplementedError


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
    return x1 / x2


def greater(x1: Array, x2: Array, /) -> Array:
    return x1 < x2


def greater_equal(x1: Array, x2: Array, /) -> Array:
    return x1 <= x2


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
    raise NotImplementedError


def log10(x: Array, /) -> Array:
    raise NotImplementedError


def logaddexp(x1: Array, x2: Array, /) -> Array:
    raise NotImplementedError


def logical_and(x1: Array, x2: Array, /) -> Array:
    return Array._from_data(x1._data and x2._data)


def logical_not(x: Array, /) -> Array:
    raise NotImplementedError


def logical_or(x1: Array, x2: Array, /) -> Array:
    return Array._from_data(x1._data or x2._data)


def logical_xor(x1: Array, x2: Array, /) -> Array:
    raise NotImplementedError


def maximum(x1: Array, x2: Array, /) -> Array:
    return Array._from_data(tyfuncs.maximum(x1._data, x2._data))


def minimum(x1: Array, x2: Array, /) -> Array:
    raise NotImplementedError


def multiply(x1: Array, x2: Array, /) -> Array:
    return Array._from_data(x1._data * x2._data)


def negative(x: Array, /) -> Array:
    raise NotImplementedError


def not_equal(x1: Array, x2: Array, /) -> Array:
    return Array._from_data(x1._data != x2._data)


def positive(x: Array, /) -> Array:
    raise NotImplementedError


def pow(x1: Array, x2: Array, /) -> Array:
    return x1**x2


def real(x: Array, /) -> Array:
    raise NotImplementedError


def remainder(x1: Array, x2: Array, /) -> Array:
    raise NotImplementedError


def round(x: Array, /) -> Array:
    raise NotImplementedError


def sign(x: Array, /) -> Array:
    raise NotImplementedError


def signbit(x: Array, /) -> Array:
    raise NotImplementedError


def sin(x: Array, /) -> Array:
    raise NotImplementedError


def sinh(x: Array, /) -> Array:
    raise NotImplementedError


def square(x: Array, /) -> Array:
    raise NotImplementedError


def sqrt(x: Array, /) -> Array:
    raise NotImplementedError


def subtract(x1: Array, x2: Array, /) -> Array:
    return Array._from_data(x1._data - x2._data)


def tan(x: Array, /) -> Array:
    raise NotImplementedError


def tanh(x: Array, /) -> Array:
    raise NotImplementedError


def trunc(x: Array, /) -> Array:
    raise NotImplementedError
