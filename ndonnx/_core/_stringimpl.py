# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import ndonnx as ndx
import ndonnx._opset_extensions as opx
from ndonnx._data_types.structtype import StructType

from ._interface import OperationsBlock
from ._utils import binary_op

if TYPE_CHECKING:
    from ndonnx import Array


class StringOperationsImpl(OperationsBlock):
    def add(self, x, y) -> Array:
        return binary_op(x, y, opx.string_concat)

    def equal(self, x, y) -> Array:
        return binary_op(x, y, opx.equal)

    def can_cast(self, from_, to) -> bool:
        if isinstance(from_, ndx.CoreType) and isinstance(to, ndx.CoreType):
            return np.can_cast(from_.to_numpy_dtype(), to.to_numpy_dtype())
        return NotImplemented

    def zeros(self, shape, dtype: ndx.CoreType | StructType | None = None, device=None):
        return ndx.full(shape, "", dtype=dtype, device=device)

    def empty(self, shape, dtype=None, device=None) -> ndx.Array:
        return ndx.zeros(shape, dtype=dtype, device=device)
