# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import ndonnx as ndx
import ndonnx._data_types as dtypes
import ndonnx._opset_extensions as opx

from ._shapeimpl import UniformShapeOperations
from ._utils import binary_op

if TYPE_CHECKING:
    from ndonnx import Array


class StringOperationsImpl(UniformShapeOperations):
    def add(self, x, y) -> Array:
        return binary_op(x, y, opx.string_concat)

    def equal(self, x, y) -> Array:
        return binary_op(x, y, opx.equal)

    def not_equal(self, x, y) -> ndx.Array:
        return ndx.logical_not(self.equal(x, y))

    def can_cast(self, from_, to) -> bool:
        if isinstance(from_, ndx.CoreType) and isinstance(to, ndx.CoreType):
            return np.can_cast(from_.to_numpy_dtype(), to.to_numpy_dtype())
        return NotImplemented

    def zeros(
        self,
        shape,
        dtype: dtypes.CoreType | dtypes.StructType | None = None,
        device=None,
    ):
        return ndx.full(shape, "", dtype=dtype)

    def zeros_like(
        self, x, dtype: dtypes.CoreType | dtypes.StructType | None = None, device=None
    ):
        return ndx.full_like(x, "", dtype=dtype)

    def empty(self, shape, dtype=None, device=None) -> ndx.Array:
        return ndx.zeros(shape, dtype=dtype, device=device)

    def empty_like(self, x, dtype=None, device=None) -> ndx.Array:
        return ndx.zeros_like(x, dtype=dtype, device=device)
