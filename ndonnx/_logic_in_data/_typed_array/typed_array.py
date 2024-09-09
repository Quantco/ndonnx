# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from types import NotImplementedType
from typing import TYPE_CHECKING, overload

import numpy as np
from typing_extensions import Self

from .. import dtypes
from ..dtypes import (
    DType,
)

if TYPE_CHECKING:
    from ..array import Index, OnnxShape, SetitemIndex
    from ..schema import Components, Schema
    from .core import TyArray, TyArrayBool, TyArrayInt64


class TyArrayBase(ABC):
    dtype: DType

    @abstractmethod
    def __init__(self): ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.__dict__})"

    @abstractmethod
    def __getitem__(self, index: Index) -> Self: ...

    def __setitem__(
        self,
        key: SetitemIndex,
        value: Self,
        /,
    ) -> None:
        # TODO: Make abstractmethod
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self) -> OnnxShape: ...

    @property
    @abstractmethod
    def dynamic_shape(self) -> TyArrayInt64: ...

    @abstractmethod
    def reshape(self, shape: tuple[int, ...]) -> Self: ...

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def to_numpy(self) -> np.ndarray:
        raise ValueError(f"Cannot convert '{self.__class__}' to NumPy array.")

    @abstractmethod
    def disassemble(self) -> tuple[Components, Schema]:
        raise NotImplementedError

    @overload
    def astype(self, dtype: dtypes.CoreDTypes) -> TyArray: ...

    @overload
    def astype(self, dtype: dtypes._CoreDType) -> TyArray: ...

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

    @abstractmethod
    def broadcast_to(self, shape: tuple[int, ...] | TyArrayInt64) -> Self: ...

    # Aggregating functions
    def all(self) -> TyArrayBase:
        raise ValueError(f"'all' is not implemented for `{self.dtype}`")

    # Element-wise functions without additional arguments

    def isnan(self) -> TyArrayBase:
        raise ValueError(f"'isnan' is not implemented for {self.dtype}")

    def isfinite(self) -> TyArrayBase:
        raise ValueError(f"'isinfinite' is not implemented for {self.dtype}")

    def acos(self) -> Self:
        raise ValueError(f"'acos' is not implemented for {self.dtype}")

    def acosh(self) -> Self:
        raise ValueError(f"'acosh' is not implemented for {self.dtype}")

    def asin(self) -> Self:
        raise ValueError(f"'asin' is not implemented for {self.dtype}")

    def asinh(self) -> Self:
        raise ValueError(f"'asinh' is not implemented for {self.dtype}")

    def atan(self) -> Self:
        raise ValueError(f"'atan' is not implemented for {self.dtype}")

    def atanh(self) -> Self:
        raise ValueError(f"'atanh' is not implemented for {self.dtype}")

    def ceil(self) -> Self:
        raise ValueError(f"'ceil' is not implemented for {self.dtype}")

    def exp(self) -> Self:
        raise ValueError(f"'exp' is not implemented for {self.dtype}")

    def expm1(self) -> Self:
        raise ValueError(f"'expm1' is not implemented for {self.dtype}")

    def floor(self) -> Self:
        raise ValueError(f"'floor' is not implemented for {self.dtype}")

    def log(self) -> Self:
        raise ValueError(f"'log' is not implemented for {self.dtype}")

    def log1p(self) -> Self:
        raise ValueError(f"'log1p' is not implemented for {self.dtype}")

    def log2(self) -> Self:
        raise ValueError(f"'log2' is not implemented for {self.dtype}")

    # Multi-input functions
    def _where(
        self, cond: TyArrayBool, y: TyArrayBase
    ) -> TyArrayBase | NotImplementedType:
        return NotImplemented

    def _rwhere(
        self, cond: TyArrayBool, y: TyArrayBase
    ) -> TyArrayBase | NotImplementedType:
        return NotImplemented

    # Dunder-functions
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
        return NotImplemented

    def __or__(self, rhs: TyArrayBase) -> TyArrayBase:
        return NotImplemented

    def __sub__(self, other: TyArrayBase) -> TyArrayBase:
        return NotImplemented
