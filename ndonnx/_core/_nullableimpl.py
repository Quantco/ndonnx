# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
from typing import TYPE_CHECKING, Union

import ndonnx as ndx

from ._interface import OperationsBlock
from ._utils import validate_core

if TYPE_CHECKING:
    from ndonnx._array import Array
    from ndonnx._data_types import CoreType, StructType

    Dtype = Union[CoreType, StructType]


class NullableOperationsImpl(OperationsBlock):
    @validate_core
    def fill_null(self, x: "Array", value) -> "Array":
        value = ndx.asarray(value)
        if value.dtype != x.values.dtype:
            value = value.astype(x.values.dtype)
        return ndx.where(x.null, value, x.values)

    @validate_core
    def make_nullable(self, x: "Array", null: "Array") -> "Array":
        return NotImplemented
