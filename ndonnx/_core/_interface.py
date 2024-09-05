# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

import ndonnx as ndx
import ndonnx._data_types as dtypes

if TYPE_CHECKING:
    from ndonnx._array import IndexType
    from ndonnx._data_types import Dtype


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
        return NotImplemented

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
        return NotImplemented

    def floor(self, x) -> ndx.Array:
        return NotImplemented

    def floor_divide(self, x, y) -> ndx.Array:
        return NotImplemented

    def greater(self, x, y) -> ndx.Array:
        return NotImplemented

    def greater_equal(self, x, y) -> ndx.Array:
        return NotImplemented

    def imag(*args):
        return NotImplemented

    def isfinite(self, x) -> ndx.Array:
        return NotImplemented

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
        return NotImplemented

    def log2(self, x) -> ndx.Array:
        return NotImplemented

    def log10(self, x) -> ndx.Array:
        return NotImplemented

    def logaddexp(self, x, y) -> ndx.Array:
        return NotImplemented

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
        return NotImplemented

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
        return NotImplemented

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
        return NotImplemented

    # searching.py

    def argmax(self, x, axis=None, keepdims=False) -> ndx.Array:
        return NotImplemented

    def argmin(self, x, axis=None, keepdims=False) -> ndx.Array:
        return NotImplemented

    def nonzero(self, x) -> tuple[ndx.Array, ...]:
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
        dtype: Dtype | None = None,
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
        dtype: Dtype | None = None,
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
        dtype: Dtype | None = None,
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
        dtype: Dtype | None = None,
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

    # creation.py
    def full(self, shape, fill_value, dtype=None, device=None) -> ndx.Array:
        return NotImplemented

    def full_like(self, x, fill_value, dtype=None, device=None) -> ndx.Array:
        return NotImplemented

    def ones(
        self,
        shape,
        dtype: Dtype | None = None,
        device=None,
    ):
        return NotImplemented

    def ones_like(
        self, x, dtype: dtypes.StructType | dtypes.CoreType | None = None, device=None
    ):
        return NotImplemented

    def zeros(
        self,
        shape,
        dtype: Dtype | None = None,
        device=None,
    ):
        return NotImplemented

    def zeros_like(self, x, dtype: Dtype | None = None, device=None):
        return NotImplemented

    def empty(self, shape, dtype=None, device=None) -> ndx.Array:
        return NotImplemented

    def empty_like(self, x, dtype=None, device=None) -> ndx.Array:
        return NotImplemented

    def arange(self, start, stop=None, step=None, dtype=None, device=None) -> ndx.Array:
        return NotImplemented

    def eye(self, n_rows, n_cols=None, k=0, dtype=None, device=None) -> ndx.Array:
        return NotImplemented

    def tril(self, x, k=0) -> ndx.Array:
        return NotImplemented

    def triu(self, x, k=0) -> ndx.Array:
        return NotImplemented

    def linspace(
        self, start, stop, num, *, dtype=None, device=None, endpoint=True
    ) -> ndx.Array:
        return NotImplemented

    # additional.py
    def fill_null(self, x, value) -> ndx.Array:
        return NotImplemented

    def shape(self, x) -> ndx.Array:
        return NotImplemented

    def make_nullable(self, x, null) -> ndx.Array:
        return NotImplemented

    def can_cast(self, from_, to) -> bool:
        return NotImplemented

    def static_shape(self, x) -> tuple[int | None, ...]:
        return NotImplemented

    def make_array(
        self,
        shape: tuple[int | None | str, ...],
        dtype: Dtype,
        eager_value: np.ndarray | None = None,
    ) -> ndx.Array:
        return NotImplemented

    def getitem(
        self,
        x: ndx.Array,
        index: IndexType,
    ) -> ndx.Array:
        return NotImplemented
