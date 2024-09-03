# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from types import NotImplementedType
from typing import TYPE_CHECKING, Generic, TypeVar, overload

import numpy as np
from typing_extensions import Self

from .. import dtypes
from ..dtypes import (
    DType,
)

if TYPE_CHECKING:
    from ..array import Index, OnnxShape
    from .core import TyArray, TyArrayBool


DTYPE = TypeVar("DTYPE", bound=DType)


class TyArrayBase(ABC, Generic[DTYPE]):
    dtype: DTYPE

    @abstractmethod
    def __init__(self): ...

    @classmethod
    @abstractmethod
    def from_typed_array(cls, tyarr: TyArrayBase):
        """Create an instances from another ``_TypedArray`` object.

        Returns `NotImplemented` if the conversion is not defined.

        Raises
        ------
        ValueError:
            If the conversion from `tyarr` is *known* to be invalid.

        Note
        ----
        See `_TypedArray._astype`.
        """
        ...

    @classmethod
    @abstractmethod
    def as_argument(cls, shape: OnnxShape): ...

    @abstractmethod
    def __getitem__(self, index: Index) -> Self: ...

    @property
    @abstractmethod
    def shape(self) -> OnnxShape: ...

    @abstractmethod
    def reshape(self, shape: tuple[int, ...]) -> Self: ...

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def to_numpy(self) -> np.ndarray:
        raise ValueError(f"Cannot convert '{self.__class__}' to NumPy array.")

    @overload
    def astype(self, dtype: dtypes.CoreDTypes) -> TyArray: ...

    @overload
    def astype(self, dtype: DType) -> TyArrayBase: ...

    def astype(self, dtype: DType) -> TyArrayBase:
        """Convert `self` to the `_TypedArray` associated with `dtype`."""
        res = self._astype(dtype)
        if res == NotImplemented:
            # `type(self._data)` does not know about the target `dtype`
            res = dtype._tyarr_class.from_typed_array(self)
        if res != NotImplemented:
            return res
        raise ValueError(f"casting between `{self.dtype}` and `{dtype}` is undefined")

    @abstractmethod
    def _astype(self, dtype: DType) -> TyArrayBase | NotImplementedType:
        """Reflective sibling method for `Self.from_data` which must thus not call the
        latter.

        Used this function to implement the conversion from a custom type into a built-
        in one.
        """
        return NotImplemented

    def _where(
        self, cond: TyArrayBool, y: TyArrayBase
    ) -> TyArrayBase | NotImplementedType:
        return NotImplemented

    def _rwhere(
        self, cond: TyArrayBool, y: TyArrayBase
    ) -> TyArrayBase | NotImplementedType:
        return NotImplemented

    def __add__(self, other: TyArrayBase) -> TyArrayBase:
        return NotImplemented

    def __and__(self, rhs: TyArrayBase) -> TyArrayBase:
        return NotImplemented

    def __invert__(self) -> TyArrayBase:
        return NotImplemented

    def __or__(self, rhs: TyArrayBase) -> TyArrayBase:
        return NotImplemented
