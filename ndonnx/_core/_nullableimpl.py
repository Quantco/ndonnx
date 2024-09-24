# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

import ndonnx as ndx

from ._shapeimpl import UniformShapeOperations
from ._utils import assemble_output_recurse, validate_core

if TYPE_CHECKING:
    from ndonnx._array import Array
    from ndonnx._data_types import CoreType, StructType

    Dtype = Union[CoreType, StructType]


class NullableOperationsImpl(UniformShapeOperations):
    @validate_core
    def fill_null(self, x: Array, value) -> Array:
        value = ndx.asarray(value)
        if value.dtype != x.values.dtype:
            value = value.astype(x.values.dtype)
        return ndx.where(x.null, value, x.values)

    @validate_core
    def make_nullable(self, x: Array, null: Array) -> Array:
        return NotImplemented

    @validate_core
    def where(self, condition, x, y):
        if x.dtype != y.dtype:
            target_dtype = ndx.result_type(x, y)
            x = ndx.astype(x, target_dtype)
            y = ndx.astype(y, target_dtype)
        return super().where(condition, x, y)

    def make_array(
        self,
        shape: tuple[int | None | str, ...],
        dtype: Dtype,
        eager_value: np.ndarray | None = None,
    ) -> Array:
        if not isinstance(dtype, ndx.Nullable):
            return NotImplemented

        if eager_value is not None:
            eager_values = dtype._parse_input(eager_value)
            values_np = assemble_output_recurse(dtype.values, eager_values["values"])
            values = ndx.asarray(values_np, dtype=dtype.values)
            null = ndx.asarray(
                np.broadcast_to(
                    assemble_output_recurse(dtype.null, eager_values["null"]),
                    values_np.shape,
                ),
                dtype=dtype.null,
            )
        else:
            values = ndx.array(shape=shape, dtype=dtype.values)
            null = ndx.array(shape=shape, dtype=dtype.null)

        return ndx.Array._from_fields(
            dtype,
            values=values,
            null=null,
        )
