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

from ._coreimpl import CoreOperationsImpl
from ._interface import OperationsBlock
from ._nullableimpl import NullableOperationsImpl
from ._utils import binary_op, unary_op, validate_core

if TYPE_CHECKING:
    from ndonnx import Array


class _BooleanOperationsImpl(OperationsBlock):
    @validate_core
    def equal(self, x, y) -> Array:
        return binary_op(x, y, opx.equal)

    @validate_core
    def not_equal(self, x, y) -> ndx.Array:
        return self.logical_not(self.equal(x, y))

    @validate_core
    def bitwise_and(self, x, y) -> Array:
        return self.logical_and(x, y)

    @validate_core
    def bitwise_invert(self, x) -> Array:
        return self.logical_not(x)

    @validate_core
    def bitwise_or(self, x, y) -> Array:
        return self.logical_or(x, y)

    @validate_core
    def bitwise_xor(self, x, y) -> Array:
        return self.logical_xor(x, y)

    @validate_core
    def logical_and(self, x, y):
        # If one of the operands is True and the broadcasted shape can be guaranteed to be the other array's shape,
        # we can return this other array directly.
        x, y = map(ndx.asarray, (x, y))
        y_np = y.to_numpy()
        x_np = x.to_numpy()
        if x_np is not None and x_np.size == 1 and x.ndim <= y.ndim and x_np.item():
            return y.copy()
        elif y_np is not None and y_np.size == 1 and y.ndim <= x.ndim and y_np.item():
            return x.copy()
        return binary_op(x, y, opx.and_)

    @validate_core
    def logical_not(self, x) -> Array:
        return unary_op(x, opx.not_)

    @validate_core
    def logical_or(self, x, y):
        # If one of the operands is False and the broadcasted shape can be guaranteed to be the other array's shape,
        # we can return this other array directly.
        x, y = map(ndx.asarray, (x, y))
        x_np = x.to_numpy()
        y_np = y.to_numpy()
        if x_np is not None and x_np.size == 1 and x.ndim <= y.ndim and not x_np.item():
            return y.copy()
        elif (
            y_np is not None and y_np.size == 1 and y.ndim <= x.ndim and not y_np.item()
        ):
            return x.copy()
        return binary_op(x, y, opx.or_)

    @validate_core
    def logical_xor(self, x, y):
        x, y = map(ndx.asarray, (x, y))
        return binary_op(x, y, opx.xor)

    @validate_core
    def argmax(self, x, axis=None, keepdims=False) -> Array:
        return ndx.argmax(x.astype(ndx.int32), axis=axis, keepdims=keepdims)

    @validate_core
    def argmin(self, x, axis=None, keepdims=False) -> Array:
        return ndx.argmin(x.astype(ndx.int32), axis=axis, keepdims=keepdims)

    @validate_core
    def can_cast(self, from_, to) -> bool:
        if isinstance(from_, ndx.CoreType) and isinstance(to, ndx.CoreType):
            return np.can_cast(from_.to_numpy_dtype(), to.to_numpy_dtype())
        return NotImplemented

    @validate_core
    def all(self, x, *, axis=None, keepdims: bool = False):
        if isinstance(x.dtype, dtypes.NullableCore):
            x = ndx.where(x.null, True, x.values)
        if functools.reduce(operator.mul, x._static_shape, 1) == 0:
            return ndx.asarray(True, dtype=ndx.bool)
        return ndx.min(x.astype(ndx.int8), axis=axis, keepdims=keepdims).astype(
            ndx.bool
        )

    @validate_core
    def any(self, x, *, axis=None, keepdims: bool = False):
        if isinstance(x.dtype, dtypes.NullableCore):
            x = ndx.where(x.null, False, x.values)
        if functools.reduce(operator.mul, x._static_shape, 1) == 0:
            return ndx.asarray(False, dtype=ndx.bool)
        return ndx.max(x.astype(ndx.int8), axis=axis, keepdims=keepdims).astype(
            ndx.bool
        )

    @validate_core
    def ones(
        self,
        shape,
        dtype: dtypes.CoreType | dtypes.StructType | None = None,
        device=None,
    ):
        dtype = dtypes.float64 if dtype is None else dtype
        return ndx.full(shape, True, dtype=dtype)

    @validate_core
    def ones_like(
        self, x, dtype: dtypes.StructType | dtypes.CoreType | None = None, device=None
    ):
        return ndx.full_like(x, True, dtype=dtype)

    @validate_core
    def zeros(
        self,
        shape,
        dtype: dtypes.CoreType | dtypes.StructType | None = None,
        device=None,
    ):
        dtype = dtypes.float64 if dtype is None else dtype
        return ndx.full(shape, False, dtype=dtype)

    @validate_core
    def zeros_like(
        self, x, dtype: dtypes.CoreType | dtypes.StructType | None = None, device=None
    ):
        return ndx.full_like(x, False, dtype=dtype)

    @validate_core
    def empty(self, shape, dtype=None, device=None) -> ndx.Array:
        shape = ndx.asarray(shape, dtype=dtypes.int64)
        return ndx.full(shape, False, dtype=dtype)

    @validate_core
    def empty_like(self, x, dtype=None, device=None) -> ndx.Array:
        return ndx.full_like(x, False, dtype=dtype)

    @validate_core
    def nonzero(self, x) -> tuple[Array, ...]:
        return ndx.nonzero(x.astype(ndx.int8))


class BooleanOperationsImpl(CoreOperationsImpl, _BooleanOperationsImpl): ...


class NullableBooleanOperationsImpl(NullableOperationsImpl, _BooleanOperationsImpl): ...
