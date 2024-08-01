# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import operator
from typing import TYPE_CHECKING

import numpy as np

import ndonnx as ndx
import ndonnx._data_types as dtypes
import ndonnx._opset_extensions as opx

from ._interface import OperationsBlock
from ._utils import binary_op, unary_op

if TYPE_CHECKING:
    from ndonnx import Array


class BooleanOperationsImpl(OperationsBlock):
    def equal(self, x, y) -> Array:
        return binary_op(x, y, opx.equal)

    def bitwise_and(self, x, y) -> Array:
        return self.logical_and(x, y)

    def bitwise_invert(self, x) -> Array:
        return self.logical_not(x)

    def bitwise_or(self, x, y) -> Array:
        return self.logical_or(x, y)

    def bitwise_xor(self, x, y) -> Array:
        return self.logical_xor(x, y)

    def logical_and(self, x, y):
        # If one of the operands is True and the broadcasted shape can be guaranteed to be the other array's shape,
        # we can return this other array directly.
        x, y = map(ndx.asarray, (x, y))
        if (
            x.to_numpy() is not None
            and x.to_numpy().size == 1
            and x.ndim <= y.ndim
            and x.to_numpy().item()
        ):
            return y.copy()
        elif (
            y.to_numpy() is not None
            and y.to_numpy().size == 1
            and y.ndim <= x.ndim
            and y.to_numpy().item()
        ):
            return x.copy()
        return binary_op(x, y, opx.and_)

    def logical_not(self, x) -> Array:
        return unary_op(x, opx.not_)

    def logical_or(self, x, y):
        # If one of the operands is False and the broadcasted shape can be guaranteed to be the other array's shape,
        # we can return this other array directly.
        x, y = map(ndx.asarray, (x, y))
        if (
            x.to_numpy() is not None
            and x.to_numpy().size == 1
            and x.ndim <= y.ndim
            and not x.to_numpy().item()
        ):
            return y.copy()
        elif (
            y.to_numpy() is not None
            and y.to_numpy().size == 1
            and y.ndim <= x.ndim
            and not y.to_numpy().item()
        ):
            return x.copy()
        return binary_op(x, y, opx.or_)

    def logical_xor(self, x, y):
        x, y = map(ndx.asarray, (x, y))
        return binary_op(x, y, opx.xor)

    def argmax(self, x, axis=None, keepdims=False) -> Array:
        return ndx.argmax(x.astype(ndx.int32), axis=axis, keepdims=keepdims)

    def argmin(self, x, axis=None, keepdims=False) -> Array:
        return ndx.argmin(x.astype(ndx.int32), axis=axis, keepdims=keepdims)

    def can_cast(self, from_, to) -> bool:
        if isinstance(from_, ndx.CoreType) and isinstance(to, ndx.CoreType):
            return np.can_cast(from_.to_numpy_dtype(), to.to_numpy_dtype())
        return NotImplemented

    def all(self, x, *, axis=None, keepdims: bool = False):
        if isinstance(x.dtype, dtypes._NullableCore):
            x = ndx.where(x.null, True, x.values)
        if functools.reduce(operator.mul, x._static_shape, 1) == 0:
            return ndx.asarray(True, dtype=ndx.bool)
        return ndx.min(x.astype(ndx.int8), axis=axis, keepdims=keepdims).astype(
            ndx.bool
        )

    def any(self, x, *, axis=None, keepdims: bool = False):
        if isinstance(x.dtype, dtypes._NullableCore):
            x = ndx.where(x.null, False, x.values)
        if functools.reduce(operator.mul, x._static_shape, 1) == 0:
            return ndx.asarray(False, dtype=ndx.bool)
        return ndx.max(x.astype(ndx.int8), axis=axis, keepdims=keepdims).astype(
            ndx.bool
        )

    def ones(
        self,
        shape,
        dtype: dtypes.CoreType | dtypes.StructType | None = None,
        device=None,
    ):
        dtype = dtypes.float64 if dtype is None else dtype
        return ndx.full(shape, True, dtype=dtype)

    def ones_like(
        self, x, dtype: dtypes.StructType | dtypes.CoreType | None = None, device=None
    ):
        return ndx.full_like(x, True, dtype=dtype)

    def zeros(
        self,
        shape,
        dtype: dtypes.CoreType | dtypes.StructType | None = None,
        device=None,
    ):
        dtype = dtypes.float64 if dtype is None else dtype
        return ndx.full(shape, False, dtype=dtype)

    def zeros_like(
        self, x, dtype: dtypes.CoreType | dtypes.StructType | None = None, device=None
    ):
        return ndx.full_like(x, False, dtype=dtype)

    def empty(self, shape, dtype=None, device=None) -> ndx.Array:
        shape = ndx.asarray(shape, dtype=dtypes.int64)
        return ndx.full(shape, False, dtype=dtype)

    def empty_like(self, x, dtype=None, device=None) -> ndx.Array:
        return ndx.full_like(x, False, dtype=dtype)

    def nonzero(self, x) -> tuple[Array, ...]:
        return ndx.nonzero(x.astype(ndx.int8))
