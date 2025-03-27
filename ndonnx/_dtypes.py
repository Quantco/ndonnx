# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from types import NotImplementedType
from typing import TYPE_CHECKING, Generic, TypeVar
from warnings import warn

import numpy as np
from spox import Var

if TYPE_CHECKING:
    from ndonnx.types import NestedSequence, OnnxShape, PyScalar

    from ._schema import DTypeInfoV1
    from ._typed_array import TyArrayBase, onnx


TY_ARRAY_BASE = TypeVar("TY_ARRAY_BASE", bound="TyArrayBase", covariant=True)


class DType(ABC, Generic[TY_ARRAY_BASE]):
    def __ndx_create__(
        self,
        val: PyScalar | np.ndarray | TyArrayBase | Var | NestedSequence,
    ) -> TY_ARRAY_BASE | NotImplementedType:
        return NotImplemented

    @abstractmethod
    def __ndx_result_type__(
        self, other: DType | PyScalar
    ) -> DType | NotImplementedType: ...

    @abstractmethod
    def __ndx_cast_from__(self, arr: TyArrayBase) -> TY_ARRAY_BASE:
        """Convert the given array to this data type.

        This function is used to implement ``TyArrayBase.astype`` and
        should not be called directly.
        """
        ...

    @abstractmethod
    def __ndx_argument__(self, shape: OnnxShape) -> TY_ARRAY_BASE: ...

    @property
    @abstractmethod
    def __ndx_infov1__(self) -> DTypeInfoV1:
        raise NotImplementedError

    # Construction functions
    def __ndx_arange__(
        self,
        start: int | float,
        stop: int | float,
        step: int | float = 1,
    ) -> TY_ARRAY_BASE:
        return NotImplemented

    def __ndx_eye__(
        self,
        n_rows: int,
        n_cols: int | None = None,
        /,
        *,
        k: int = 0,
    ) -> TY_ARRAY_BASE:
        return NotImplemented

    def __ndx_ones__(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> TY_ARRAY_BASE:
        return NotImplemented

    def __ndx_zeros__(
        self, shape: tuple[int, ...] | onnx.TyArrayInt64
    ) -> TY_ARRAY_BASE:
        return NotImplemented

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return False
        return (self.__dict__ == other.__dict__) and (self.__slots__ == other.__slots__)

    def __hash__(self) -> int:
        return hash((tuple(sorted(self.__dict__.items())), self.__slots__))

    def __repr__(self) -> str:
        return self.__class__.__name__

    def to_numpy_dtype(self) -> np.dtype:
        warn("'to_numpy_dtype' is deprecated. Use `unwarp_numpy` method instead")

        return self.unwrap_numpy()

    def unwrap_numpy(self) -> np.dtype:
        raise ValueError(f"`{self}` provides no corresponding NumPy data type")
