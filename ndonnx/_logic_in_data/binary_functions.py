# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
"""Free functions taking exactly two arrays as arguments.

These functions must find a common type and rely on reflexive implementations in the
typed arrays.
"""

from .array import Array
from .elementwise import atan


def add(a: Array, b: Array) -> Array:
    return a + b


def atan2(x1: Array, x2: Array, /) -> Array:
    # TODO: Consider special cases defined in standard
    return atan(x1 / x2)


def bitwise_and(x1: Array, x2: Array, /) -> Array:
    return x1 & x2


def bitwise_left_shift(x1: Array, x2: Array, /) -> Array:
    return x1 << x2


def bitwise_or(x1: Array, x2: Array, /) -> Array:
    return x1 | x2


def bitwise_right_shift(x1: Array, x2: Array, /) -> Array:
    return x1 >> x2


def bitwise_xor(x1: Array, x2: Array, /) -> Array:
    return x1 ^ x2


def copysign(x1: Array, x2: Array, /) -> Array:
    raise NotImplementedError


def divide(x1: Array, x2: Array, /) -> Array:
    return x1 / x2


def equal(x1: Array, x2: Array, /) -> Array:
    return x1 == x2


def floor_divide(x1: Array, x2: Array, /) -> Array:
    return x1 / x2


def greater(x1: Array, x2: Array, /) -> Array:
    return x1 < x2


def greater_equal(x1: Array, x2: Array, /) -> Array:
    return x1 <= x2


def hypot(x1: Array, x2: Array, /) -> Array:
    raise NotImplementedError


def less(x1: Array, x2: Array, /) -> Array:
    return x1 < x2


def less_equal(x1: Array, x2: Array, /) -> Array:
    return x1 <= x2


def logaddexp(x1: Array, x2: Array, /) -> Array:
    raise NotImplementedError


def logical_and(x1: Array, x2: Array, /) -> Array:
    return x1 & x2


def logical_or(x1: Array, x2: Array, /) -> Array:
    return x1 | x2


def logical_xor(x1: Array, x2: Array, /) -> Array:
    return x1 ^ x2


def maximum(x1: Array, x2: Array, /) -> Array:
    from ._typed_array.funcs import maximum

    data = maximum(x1._data, x2._data)
    return Array._from_data(data)


def minimum(x1: Array, x2: Array, /) -> Array:
    raise NotImplementedError


def multiply(x1: Array, x2: Array, /) -> Array:
    return x1 * x2


def not_equal(x1: Array, x2: Array, /) -> Array:
    return x1 != x2


def pow(x1: Array, x2: Array, /) -> Array:
    return x1**x2


def remainder(x1: Array, x2: Array, /) -> Array:
    raise NotImplementedError


def subtract(x1: Array, x2: Array, /) -> Array:
    return x1 - x2
