# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
"""Element-wise free functions.

Each function directly dispatches to the inner typed array.
"""

from .array import Array


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


def ceil(array: Array, /) -> Array:
    return Array._from_data(array._data.ceil())


def exp(array: Array, /) -> Array:
    return Array._from_data(array._data.exp())


def expm1(array: Array, /) -> Array:
    return Array._from_data(array._data.expm1())


def floor(array: Array, /) -> Array:
    return Array._from_data(array._data.floor())


def isnan(array: Array, /) -> Array:
    return Array._from_data(array._data.isnan())


def isfinite(array: Array, /) -> Array:
    return Array._from_data(array._data.isfinite())


def isinf(array: Array, /) -> Array:
    return Array._from_data(array._data.isinf())


# TODO: Expose the remaining unary functions
