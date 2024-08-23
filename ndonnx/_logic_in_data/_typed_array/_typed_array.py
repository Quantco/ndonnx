# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from types import NotImplementedType
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
from typing_extensions import Self

from ..dtypes import (
    DType,
)

if TYPE_CHECKING:
    from ..array import OnnxShape


DTYPE = TypeVar("DTYPE", bound=DType)


class _TypedArray(ABC, Generic[DTYPE]):
    @abstractmethod
    def __init__(self): ...

    @classmethod
    @abstractmethod
    def from_data(cls, data: _TypedArray):
        """Create an instances from a different data object."""
        ...

    @abstractmethod
    def __getitem__(self, index) -> Self: ...

    @property
    @abstractmethod
    def dtype(self) -> DTYPE: ...

    @property
    @abstractmethod
    def ndim(self) -> int: ...

    @property
    @abstractmethod
    def shape(self) -> OnnxShape: ...

    @classmethod
    @abstractmethod
    def from_np_schema(cls, schema: dict[str, Any], /) -> Self: ...

    @abstractmethod
    def reshape(self, shape: tuple[int, ...]) -> Self: ...

    def to_numpy(self) -> np.ndarray:
        raise TypeError(f"Cannot convert '{self.__class__}' to NumPy array.")

    def astype(self, dtype: DType) -> _TypedArray:
        """Convert `self` to the data type associated with `dtype`."""
        res = self._astype(dtype)
        if res == NotImplemented:
            # `type(self._data)` does not know about the target `dtype`
            res = dtype._data_class.from_data(self)
        if res != NotImplemented:
            return res
        raise ValueError(f"casting between `{self.dtype}` and `{dtype}` is undefined")

    @abstractmethod
    def _astype(self, dtype: DType) -> _TypedArray | NotImplementedType:
        return NotImplemented

    def __add__(self, other: _TypedArray) -> _TypedArray:
        return NotImplemented

    def __or__(self, rhs: _TypedArray) -> _TypedArray:
        return NotImplemented

    def promote(self, *others: _TypedArray) -> Sequence[_TypedArray]:
        return NotImplemented
