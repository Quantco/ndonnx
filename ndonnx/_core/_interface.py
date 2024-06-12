# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Literal

import numpy as np

import ndonnx as ndx
import ndonnx._data_types as dtypes


class OperationsBlock:
    """Interface for data types to implement top-level functions exported by ndonnx."""

    # elementwise.py

    def abs(self, x) -> ndx.Array:
        return NotImplemented

    def acos(self, x) -> ndx.Array:
        return NotImplemented

    def acosh(self, x) -> ndx.Array:
        return NotImplemented

    def add(self, x, y) -> ndx.Array:
        return NotImplemented

    def asin(self, x) -> ndx.Array:
        return NotImplemented

    def asinh(self, x) -> ndx.Array:
        return NotImplemented

    def atan(self, x) -> ndx.Array:
        return NotImplemented

    def atan2(self, y, x) -> ndx.Array:
        return ndx.atan(ndx.divide(y, x))

    def atanh(self, x) -> ndx.Array:
        return NotImplemented

    def bitwise_and(self, x, y) -> ndx.Array:
        return NotImplemented

    def bitwise_left_shift(self, x, y) -> ndx.Array:
        return NotImplemented

    def bitwise_invert(self, x) -> ndx.Array:
        return NotImplemented

    def bitwise_or(self, x, y) -> ndx.Array:
        return NotImplemented

    def bitwise_right_shift(self, x, y) -> ndx.Array:
        return NotImplemented

    def bitwise_xor(self, x, y) -> ndx.Array:
        return NotImplemented

    def ceil(self, x) -> ndx.Array:
        return NotImplemented

    def conj(*args):
        return NotImplemented

    def cos(self, x) -> ndx.Array:
        return NotImplemented

    def cosh(self, x) -> ndx.Array:
        return NotImplemented

    def divide(self, x, y) -> ndx.Array:
        return NotImplemented

    def equal(self, x, y) -> ndx.Array:
        return NotImplemented

    def exp(self, x) -> ndx.Array:
        return NotImplemented

    def expm1(self, x) -> ndx.Array:
        return ndx.subtract(ndx.exp(x), 1)

    def floor(self, x) -> ndx.Array:
        return NotImplemented

    def floor_divide(self, x, y) -> ndx.Array:
        return ndx.floor(ndx.divide(x, y))

    def greater(self, x, y) -> ndx.Array:
        return NotImplemented

    def greater_equal(self, x, y) -> ndx.Array:
        return NotImplemented

    def imag(*args):
        return NotImplemented

    def isfinite(self, x) -> ndx.Array:
        return ndx.logical_not(ndx.isinf(x))

    def isinf(self, x) -> ndx.Array:
        return NotImplemented

    def isnan(self, x) -> ndx.Array:
        return NotImplemented

    def less(self, x, y) -> ndx.Array:
        return NotImplemented

    def less_equal(self, x, y) -> ndx.Array:
        return NotImplemented

    def log(self, x) -> ndx.Array:
        return NotImplemented

    def log1p(self, x) -> ndx.Array:
        return ndx.add(ndx.log(x), ndx.asarray(1, x.dtype))

    def log2(self, x) -> ndx.Array:
        return ndx.log(x) / np.log(2)

    def log10(self, x) -> ndx.Array:
        return ndx.log(x) / np.log(10)

    def logaddexp(self, x, y) -> ndx.Array:
        return ndx.log(ndx.exp(x) + ndx.exp(y))

    def logical_and(self, x, y) -> ndx.Array:
        return NotImplemented

    def logical_not(self, x) -> ndx.Array:
        return NotImplemented

    def logical_or(self, x, y) -> ndx.Array:
        return NotImplemented

    def logical_xor(self, x, y) -> ndx.Array:
        return NotImplemented

    def multiply(self, x, y) -> ndx.Array:
        return NotImplemented

    def negative(self, x) -> ndx.Array:
        return NotImplemented

    def not_equal(self, x, y) -> ndx.Array:
        return ndx.logical_not(ndx.equal(x, y))

    def positive(self, x) -> ndx.Array:
        return NotImplemented

    def pow(self, x, y) -> ndx.Array:
        return NotImplemented

    def real(*args):
        return NotImplemented

    def remainder(self, x, y) -> ndx.Array:
        return NotImplemented

    def round(self, x) -> ndx.Array:
        return NotImplemented

    def sign(self, x) -> ndx.Array:
        return NotImplemented

    def sin(self, x) -> ndx.Array:
        return NotImplemented

    def sinh(self, x) -> ndx.Array:
        return NotImplemented

    def square(self, x) -> ndx.Array:
        return ndx.multiply(x, x)

    def sqrt(self, x) -> ndx.Array:
        return NotImplemented

    def subtract(self, x, y) -> ndx.Array:
        return NotImplemented

    def tan(self, x) -> ndx.Array:
        return NotImplemented

    def tanh(self, x) -> ndx.Array:
        return NotImplemented

    def trunc(self, x) -> ndx.Array:
        return NotImplemented

    # linalg.py

    def matmul(self, x, y) -> ndx.Array:
        return NotImplemented

    def matrix_transpose(self, x) -> ndx.Array:
        return ndx.permute_dims(x, list(range(x.ndim - 2)) + [x.ndim - 1, x.ndim - 2])

    # searching.py

    def argmax(self, x, axis=None, keepdims=False) -> ndx.Array:
        return NotImplemented

    def argmin(self, x, axis=None, keepdims=False) -> ndx.Array:
        return NotImplemented

    def nonzero(self, x) -> list[ndx.Array]:
        return NotImplemented

    def searchsorted(
        self,
        x1,
        x2,
        *,
        side: Literal["left", "right"] = "left",
        sorter: ndx.Array | None = None,
    ) -> ndx.Array:
        return NotImplemented

    def where(self, condition, x, y) -> ndx.Array:
        return NotImplemented

    # set.py
    def unique_all(self, x):
        return NotImplemented

    def unique_counts(self, x):
        return NotImplemented

    def unique_inverse(self, x):
        return NotImplemented

    def unique_values(self, x):
        return NotImplemented

    # sorting.py

    def argsort(self, x, *, axis=-1, descending=False, stable=True) -> ndx.Array:
        return NotImplemented

    def sort(self, x, *, axis=-1, descending=False, stable=True) -> ndx.Array:
        return NotImplemented

    # statistical.py

    def cumulative_sum(
        self,
        x,
        *,
        axis: int | None = None,
        dtype: ndx.CoreType | dtypes.StructType | None = None,
        include_initial: bool = False,
    ):
        return NotImplemented

    def max(self, x, *, axis=None, keepdims: bool = False) -> ndx.Array:
        return NotImplemented

    def mean(self, x, *, axis=None, keepdims: bool = False) -> ndx.Array:
        return NotImplemented

    def min(self, x, *, axis=None, keepdims: bool = False) -> ndx.Array:
        return NotImplemented

    def prod(
        self,
        x,
        *,
        axis=None,
        dtype: dtypes.CoreType | dtypes.StructType | None = None,
        keepdims: bool = False,
    ) -> ndx.Array:
        return NotImplemented

    def clip(self, x, *, min=None, max=None) -> ndx.Array:
        return NotImplemented

    def std(
        self,
        x,
        axis=None,
        keepdims: bool = False,
        correction=0.0,
        dtype: dtypes.StructType | dtypes.CoreType | None = None,
    ) -> ndx.Array:
        return NotImplemented

    def sum(
        self,
        x,
        *,
        axis=None,
        dtype: dtypes.CoreType | dtypes.StructType | None = None,
        keepdims: bool = False,
    ) -> ndx.Array:
        return NotImplemented

    def var(
        self,
        x,
        *,
        axis=None,
        keepdims: bool = False,
        correction=0.0,
        dtype: dtypes.CoreType | dtypes.StructType | None = None,
    ) -> ndx.Array:
        return NotImplemented

    # utility.py

    def all(self, x, *, axis=None, keepdims: bool = False) -> ndx.Array:
        return NotImplemented

    def any(self, x, *, axis=None, keepdims: bool = False) -> ndx.Array:
        return NotImplemented

    # indexing.py
    def take(self, x, indices, axis=None) -> ndx.Array:
        return NotImplemented

    def broadcast_to(self, x, shape) -> ndx.Array:
        return NotImplemented

    def expand_dims(self, x, axis) -> ndx.Array:
        return NotImplemented

    def flip(self, x, axis) -> ndx.Array:
        return NotImplemented

    def permute_dims(self, x, axes) -> ndx.Array:
        return NotImplemented

    def reshape(self, x, shape, *, copy=None) -> ndx.Array:
        return NotImplemented

    def roll(self, x, shift, axis) -> ndx.Array:
        return NotImplemented

    def squeeze(self, x, axis) -> ndx.Array:
        return NotImplemented

    # additional.py
    def shape(self, x) -> ndx.Array:
        return NotImplemented

    def fill_null(self, x, value) -> ndx.Array:
        return NotImplemented

    def make_nullable(self, x, null) -> ndx.Array:
        return NotImplemented
