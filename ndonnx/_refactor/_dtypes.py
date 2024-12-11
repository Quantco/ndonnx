# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from types import NotImplementedType
from typing import TYPE_CHECKING, Generic, TypeVar
from warnings import warn

import numpy as np

from ._schema import DTypeInfoV1

if TYPE_CHECKING:
    from ._array import OnnxShape
    from ._typed_array import TyArrayBase, TyArrayInt64, onnx


TY_ARRAY_BASE = TypeVar("TY_ARRAY_BASE", bound="TyArrayBase")


class DType(ABC, Generic[TY_ARRAY_BASE]):
    @abstractmethod
    def __ndx_result_type__(self, other: DType) -> DType | NotImplementedType: ...

    @property
    @abstractmethod
    def _tyarr_class(self) -> type[TY_ARRAY_BASE]:
        """Consider using  ``TyArrayBase.astype`` or ``_argument`` instead.

        Those functions better provide the dtype instance (with it's state) to the newly
        instantiated array.
        """
        ...

    @abstractmethod
    def __ndx_cast_from__(self, arr: TyArrayBase) -> TY_ARRAY_BASE:
        """Convert the given array to this data type.

        This function is used to implement ``TyArrayBase.astype`` and
        should not be called directly.
        """
        ...

    @abstractmethod
    def _argument(self, shape: OnnxShape) -> TY_ARRAY_BASE: ...

    @property
    @abstractmethod
    def _infov1(self) -> DTypeInfoV1:
        raise NotImplementedError

    # Construction functions
    @abstractmethod
    def _arange(
        self,
        start: int | float,
        stop: int | float,
        step: int | float = 1,
    ) -> TY_ARRAY_BASE: ...

    @abstractmethod
    def _eye(
        self,
        n_rows: int,
        n_cols: int | None = None,
        /,
        *,
        k: int = 0,
    ) -> TY_ARRAY_BASE: ...

    @abstractmethod
    def _ones(self, shape: tuple[int, ...] | TyArrayInt64) -> TY_ARRAY_BASE: ...

    @abstractmethod
    def _zeros(self, shape: tuple[int, ...] | TyArrayInt64) -> TY_ARRAY_BASE: ...

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


# Helper functions


def from_numpy(np_dtype: np.dtype) -> onnx.DTypes:
    from ._typed_array import onnx

    # Ensure that this also works with np.generic such as np.int64
    np_dtype = np.dtype(np_dtype)

    if np_dtype == np.int8:
        return onnx.int8
    if np_dtype == np.int16:
        return onnx.int16
    if np_dtype == np.int32:
        return onnx.int32
    if np_dtype == np.int64:
        return onnx.int64

    if np_dtype == np.uint8:
        return onnx.uint8
    if np_dtype == np.uint16:
        return onnx.uint16
    if np_dtype == np.uint32:
        return onnx.uint32
    if np_dtype == np.uint64:
        return onnx.uint64

    if np_dtype == np.float16:
        return onnx.float16
    if np_dtype == np.float32:
        return onnx.float32
    if np_dtype == np.float64:
        return onnx.float64

    if np_dtype == np.bool_:
        return onnx.bool_

    # "T" i.e. "Text" is the kind used for `StringDType` in numpy >= 2
    # See https://numpy.org/neps/nep-0055-string_dtype.html#python-api-for-stringdtype
    if np_dtype.kind in ["U", "T"]:
        return onnx.utf8

    raise ValueError(f"'{np_dtype}' does not have a corresponding ndonnx data type")
