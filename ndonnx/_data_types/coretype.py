# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

import ndonnx as ndx
import ndonnx._core as impl
import ndonnx._opset_extensions as opx

from .conversion import CastError, CastMixin
from .schema import Schema

if TYPE_CHECKING:
    from ndonnx._array import Array

    from .structtype import StructType


class CoreType(CastMixin):
    """Base class for the standard data types present in ONNX."""

    @staticmethod
    @abstractmethod
    def to_numpy_dtype() -> np.dtype:
        """Returns the numpy dtype corresponding to this data type."""

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is type(other)
            or type(self) is other
            or self.to_numpy_dtype() == other
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return type(self).__name__

    def __hash__(self) -> int:
        return hash(type(self))

    def __str__(self) -> str:
        return repr(self)

    def _schema(self) -> Schema:
        return Schema(type_name=type(self).__name__, author="ndonnx")

    def _assemble_output(self, fields: dict[str, np.ndarray]) -> np.ndarray:
        if "data" not in fields:
            raise ValueError("Fields must contain 'data' key")
        return fields["data"]

    def _parse_input(self, data: np.ndarray) -> dict[str, np.ndarray]:
        if data.dtype != self.to_numpy_dtype():
            raise ValueError(
                f"Expected dtype {self.to_numpy_dtype()}, got {data.dtype}"
            )
        if isinstance(data, np.ma.MaskedArray):
            raise ValueError(
                "Masked arrays are not supported for core data types. Use a nullable data type instead."
            )
        return {"data": data}

    def _cast_from(self, array: Array) -> Array:
        if isinstance(array.dtype, CoreType):
            return ndx.Array._from_fields(self, data=opx.cast(array._core(), to=self))
        else:
            raise CastError(f"Cannot cast from {array.dtype} to {self}")

    def _cast_to(self, array: Array, dtype: CoreType | StructType) -> Array:
        if isinstance(dtype, CoreType):
            return dtype._cast_from(array)
        else:
            raise CastError(f"Cannot cast from {array.dtype} to {dtype}")

    _ops: impl.CoreOperationsImpl = impl.CoreOperationsImpl()
