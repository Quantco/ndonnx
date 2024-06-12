# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
import spox.opset.ai.onnx.v20 as op
from spox import Var
from typing_extensions import Self

import ndonnx as ndx

from ._index import ScalarIndexType, construct_index
from ._utility import get_dtype, get_rank, get_shape

if TYPE_CHECKING:
    from ._data_types import CoreType

IndexType = Union[ScalarIndexType, tuple[ScalarIndexType, ...], "_CoreArray"]


class _CoreArray:
    """Thin wrapper around a `spox.Var` as well as optionally the eager propagated
    value."""

    var: Var
    _eager_value: np.ndarray | None

    def __init__(self, value: Var | np.ndarray) -> None:
        if isinstance(value, Var):
            self.var = value
            self._eager_value = None
        else:
            self.var = op.const(value)
            self._eager_value = value

    def _set(self, other: _CoreArray) -> Self:
        self.var = other.var
        self._eager_value = other.to_numpy()
        return self

    def copy(self) -> Self:
        return type(self)(
            self.var if self._eager_value is None else self._eager_value,
        )

    def to_numpy(self) -> np.ndarray | None:
        return self._eager_value

    @property
    def dtype(self) -> CoreType:
        return ndx.from_numpy_dtype(get_dtype(self))

    def astype(self, dtype: CoreType) -> _CoreArray:
        import ndonnx._opset_extensions as opx

        return opx.cast(self, dtype)

    def __getitem__(self, index: IndexType) -> _CoreArray:
        import ndonnx._opset_extensions as opx

        normalised_index = self._normalise_index(index)
        return opx.getitem(self, normalised_index)

    def __setitem__(
        self, index: IndexType, updates: int | bool | float | _CoreArray
    ) -> Self:
        import ndonnx._opset_extensions as opx

        normalised_index = self._normalise_index(index)
        normalised_updates = (
            updates
            if isinstance(updates, _CoreArray)
            else _CoreArray(np.array(updates))
        )
        ret = opx.setitem(self, normalised_index, normalised_updates)
        return self._set(ret)

    def _normalise_index(
        self, index: IndexType
    ) -> _CoreArray | tuple[ScalarIndexType, ...]:
        if isinstance(index, _CoreArray):
            return index
        else:
            index = construct_index(self, index)
            indexing_expressions = len(tuple(idx for idx in index if idx is not None))
            if self.ndim != indexing_expressions:
                raise IndexError(
                    f"Index has {indexing_expressions} expressions but Array has rank {self.ndim}"
                )
            return index

    @property
    def ndim(self) -> int:
        return get_rank(self)

    def __repr__(self) -> str:
        return f"<CoreArray : {self.dtype}[{get_shape(self)}] -> propagated: {self.to_numpy()}>"
