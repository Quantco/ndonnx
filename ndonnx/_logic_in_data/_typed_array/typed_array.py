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
    def as_argument(cls, shape: OnnxShape, dtype: DType):
        """Create an argument array.

        The 'dtype' parameter is needed since data types, such as categorical ones, may
        carry state not encapsulated in the typed array.
        """
        raise NotImplementedError

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
        if res is NotImplemented:
            # `type(self._data)` does not know about the target `dtype`
            res = dtype._tyarray_from_tyarray(self)
        if res is not NotImplemented:
            return res
        raise ValueError(f"casting between `{self.dtype}` and `{dtype}` is undefined")

    @abstractmethod
    def _astype(self, dtype: DType) -> TyArrayBase | NotImplementedType:
        """Reflective sibling method for `Self.from_typed_array` which must thus not
        call the latter.

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

    # mypy believes that __eq__ should return a `bool` but the docs say we can return whatever:
    # https://docs.python.org/3/reference/datamodel.html#object.__eq__
    def __eq__(self, other) -> TyArrayBase:  # type: ignore
        res = self._eqcomp(other)
        if res is NotImplemented:
            res = other._eqcomp(self)
        if res is NotImplemented:
            raise ValueError(
                f"comparison between `{type(self).__name__}` and `{type(other).__name__}` is not implemented."
            )
        return res

    @abstractmethod
    def _eqcomp(self, other: TyArrayBase) -> TyArrayBase | NotImplementedType:
        """Implementation of equal-comparison.

        '__eq__' has special semantics compared to other dunder methods.
        https://docs.python.org/3/reference/datamodel.html#object.__eq__
        """
        ...

    def __invert__(self) -> TyArrayBase:
        return NotImplemented

    def __mul__(self, rhs: TyArrayBase) -> TyArrayBase:
        return NotImplemented

    def __ne__(self, other: TyArrayBase) -> TyArrayBase:  # type: ignore
        breakpoint()
        return NotImplemented

    def __or__(self, rhs: TyArrayBase) -> TyArrayBase:
        return NotImplemented

    def __sub__(self, other: TyArrayBase) -> TyArrayBase:
        return NotImplemented
