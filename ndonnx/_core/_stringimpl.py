# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import ndonnx as ndx
import ndonnx._data_types as dtypes
import ndonnx._opset_extensions as opx

from ._nullableimpl import NullableOperationsImpl
from ._shapeimpl import UniformShapeOperations
from ._utils import binary_op, validate_core

if TYPE_CHECKING:
    from ndonnx import Array


class StringOperationsImpl(UniformShapeOperations):
    @validate_core
    def add(self, x, y) -> Array:
        return binary_op(x, y, opx.string_concat)

    @validate_core
    def equal(self, x, y) -> Array:
        return binary_op(x, y, opx.equal)

    @validate_core
    def not_equal(self, x, y) -> ndx.Array:
        return ndx.logical_not(self.equal(x, y))

    @validate_core
    def can_cast(self, from_, to) -> bool:
        if isinstance(from_, ndx.CoreType) and isinstance(to, ndx.CoreType):
            return np.can_cast(from_.to_numpy_dtype(), to.to_numpy_dtype())
        return NotImplemented

    @validate_core
    def zeros(
        self,
        shape,
        dtype: dtypes.CoreType | dtypes.StructType | None = None,
        device=None,
    ):
        return ndx.full(shape, "", dtype=dtype)

    @validate_core
    def zeros_like(
        self, x, dtype: dtypes.CoreType | dtypes.StructType | None = None, device=None
    ):
        return ndx.full_like(x, "", dtype=dtype)

    @validate_core
    def empty(self, shape, dtype=None, device=None) -> ndx.Array:
        return ndx.zeros(shape, dtype=dtype, device=device)

    @validate_core
    def empty_like(self, x, dtype=None, device=None) -> ndx.Array:
        return ndx.zeros_like(x, dtype=dtype, device=device)

    @validate_core
    def make_nullable(self, x, null):
        if null.dtype != dtypes.bool:
            raise TypeError("null must be a boolean array")

        return ndx.Array._from_fields(
            dtypes.into_nullable(x.dtype),
            values=x.copy(),
            null=ndx.reshape(null, x.shape),
        )


class NullableStringOperationsImpl(StringOperationsImpl, NullableOperationsImpl):
    def make_nullable(self, x, null):
        return NotImplemented
