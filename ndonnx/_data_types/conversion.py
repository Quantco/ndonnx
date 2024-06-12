# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ndonnx._array import Array
    from ndonnx._data_types import CoreType, StructType


class CastError(TypeError):
    """Error raised when a casting operation fails."""


class CastMixin:
    """Abstract interface for casting operations.

    When implemented for a data type, arrays of this data type can be casted to and from
    this data type.
    """

    @abstractmethod
    def _cast_from(self, array: Array) -> Array:
        """Casts the input array to this data type.

        Raise a `CastFailedError` if the cast is not possible.
        """
        ...

    @abstractmethod
    def _cast_to(self, array: Array, dtype: CoreType | StructType) -> Array:
        """Cast the input array from this data type to the specified data type `dtype`.

        Raise a `CastFailedError` if the cast is not possible.
        """
        ...
