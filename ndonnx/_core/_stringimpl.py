# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import ndonnx as ndx
import ndonnx._data_types as dtypes
import ndonnx._opset_extensions as opx

from ._coreimpl import CoreOperationsImpl
from ._interface import OperationsBlock
from ._nullableimpl import NullableOperationsImpl
from ._utils import binary_op, validate_core

if TYPE_CHECKING:
    from ndonnx import Array


class _StringOperationsImpl(OperationsBlock):
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
        if dtype is not None and not isinstance(
            dtype, (dtypes.CoreType, dtypes.NullableCore)
        ):
            raise TypeError("'dtype' must be a CoreType or NullableCoreType")
        if dtype in (None, dtypes.utf8, dtypes.nutf8):
            return ndx.full_like(x, "", dtype=dtype)
        else:
            return ndx.full_like(x, 0, dtype=dtype)

    @validate_core
    def empty(self, shape, dtype=None, device=None) -> ndx.Array:
        return ndx.zeros(shape, dtype=dtype, device=device)

    @validate_core
    def empty_like(self, x, dtype=None, device=None) -> ndx.Array:
        return ndx.zeros_like(x, dtype=dtype, device=device)


class StringOperationsImpl(CoreOperationsImpl, _StringOperationsImpl): ...


class NullableStringOperationsImpl(NullableOperationsImpl, _StringOperationsImpl): ...
