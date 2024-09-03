# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from spox import Tensor, argument

import ndonnx as ndx
import ndonnx._data_types as dtypes
import ndonnx.additional as nda
from ndonnx._corearray import _CoreArray

from ._shapeimpl import UniformShapeOperations
from ._utils import validate_core

if TYPE_CHECKING:
    from ndonnx._array import Array
    from ndonnx._data_types import Dtype


class CoreOperationsImpl(UniformShapeOperations):
    def make_array(
        self,
        shape: tuple[int | None | str, ...],
        dtype: Dtype,
        eager_value: np.ndarray | None = None,
    ) -> Array:
        if not isinstance(dtype, dtypes.CoreType):
            return NotImplemented
        return ndx.Array._from_fields(
            dtype,
            data=_CoreArray(
                dtype._parse_input(eager_value)["data"]
                if eager_value is not None
                else argument(Tensor(dtype.to_numpy_dtype(), shape))
            ),
        )

    @validate_core
    def make_nullable(self, x: Array, null: Array) -> Array:
        if null.dtype != ndx.bool:
            raise TypeError("'null' must be a boolean array")

        return ndx.Array._from_fields(
            dtypes.into_nullable(x.dtype),
            values=x.copy(),
            null=ndx.broadcast_to(null, nda.shape(x)),
        )

    @validate_core
    def where(self, condition, x, y):
        if x.dtype != y.dtype:
            target_dtype = ndx.result_type(x, y)
            x = ndx.astype(x, target_dtype)
            y = ndx.astype(y, target_dtype)
        return super().where(condition, x, y)
