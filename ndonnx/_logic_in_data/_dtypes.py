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
    from ._typed_array import TyArrayBase, onnx


TY_ARRAY = TypeVar("TY_ARRAY", bound="TyArrayBase")


class DType(ABC, Generic[TY_ARRAY]):
    @abstractmethod
    def _result_type(self, other: DType) -> DType | NotImplementedType: ...

    @property
    @abstractmethod
    def _tyarr_class(self) -> type[TY_ARRAY]:
        """Consider using  ``TyArrayBase.astype`` or ``_argument`` instead of.

        Those functions better provide the dtype instance (with it's state) to the newly
        instantiated array.
        """
        ...

    @abstractmethod
    def __ndx_convert_tyarray__(self, arr: TyArrayBase) -> TY_ARRAY:
        """Convert the given array to this data type.

        This function is used to implement ``TyArrayBase.astype`` and
        should not be called directly.
        """
        ...

    @abstractmethod
    def _argument(self, shape: OnnxShape) -> TY_ARRAY: ...

    @property
    @abstractmethod
    def _infov1(self) -> DTypeInfoV1:
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return False
        return (self.__dict__ == other.__dict__) and (self.__slots__ == other.__slots__)

    def __hash__(self) -> int:
        return hash((tuple(sorted(self.__dict__.items())), self.__slots__))

    def __repr__(self) -> str:
        return self.__class__.__name__

    def to_numpy_dtype(self) -> np.dtype:
        from ._typed_array import onnx

        warn(
            "'to_numpy_dtype' is deprecated. Use `unwarp_numpy` on the array object instead."
        )
        if isinstance(self, onnx._OnnxDType):
            return as_numpy(self)
        raise NotImplementedError


# Helper functions


def from_numpy(np_dtype: np.dtype) -> onnx.DTypes:
    from ._typed_array import onnx

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

    if np_dtype == np.bool:
        return onnx.bool_

    if np_dtype == np.dtypes.StringDType() or np_dtype.kind == "U":
        return onnx.string

    raise ValueError(f"'{np_dtype}' does not have a corresponding ndonnx data type")


def as_numpy(dtype: onnx._OnnxDType) -> np.dtype:
    from ._typed_array import onnx

    if dtype == onnx.int8:
        return np.dtype("int8")
    if dtype == onnx.int16:
        return np.dtype("int16")
    if dtype == onnx.int32:
        return np.dtype("int32")
    if dtype == onnx.int64:
        return np.dtype("int64")

    if dtype == onnx.uint8:
        return np.dtype("uint8")
    if dtype == onnx.uint16:
        return np.dtype("uint16")
    if dtype == onnx.uint32:
        return np.dtype("uint32")
    if dtype == onnx.uint64:
        return np.dtype("uint64")

    if dtype == onnx.float16:
        return np.dtype("float16")
    if dtype == onnx.float32:
        return np.dtype("float32")
    if dtype == onnx.float64:
        return np.dtype("float64")

    if dtype == onnx.bool_:
        return np.dtype("bool")

    if dtype == onnx.string:
        # TODO: Migrate to numpy.StringDType
        return np.dtype(str)

    # Should never happen
    raise ValueError(f"'{dtype}' does not have a corresponding NumPy data type")
