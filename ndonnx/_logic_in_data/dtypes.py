# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from types import NotImplementedType
from typing import TYPE_CHECKING, Generic, TypeVar, overload

import numpy as np

from .schema import DTypeInfo

if TYPE_CHECKING:
    from . import _typed_array
    from ._typed_array import TyArrayBase, onnx
    from ._typed_array.onnx import CoreDTypes
    from .array import OnnxShape
    from .schema import DTypeInfo


TY_ARRAY = TypeVar("TY_ARRAY", bound="TyArrayBase")


class DType(ABC, Generic[TY_ARRAY]):
    @abstractmethod
    def _result_type(self, other: DType) -> DType | NotImplementedType: ...

    @property
    @abstractmethod
    def _tyarr_class(self) -> type[TY_ARRAY]:
        """Consider using  `_tyarray_from_tyarray` or `_argument` instead of.

        Those functions better provide the dtype instance (with it's state) to the newly
        instantiated array.
        """
        ...

    @abstractmethod
    def _tyarray_from_tyarray(self, arr: _typed_array.TyArrayBase) -> TY_ARRAY:
        # TODO: Find a better name and add a docstring
        # replaces `TyArrayBase.from_typed_array`
        ...

    @abstractmethod
    def _argument(self, shape: OnnxShape) -> TY_ARRAY: ...

    @property
    @abstractmethod
    def _info(self) -> DTypeInfo:
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return False
        return (self.__dict__ == other.__dict__) and (self.__slots__ == other.__slots__)

    def __hash__(self) -> int:
        return hash((tuple(sorted(self.__dict__.items())), self.__slots__))

    def __repr__(self) -> str:
        return self.__class__.__name__


# Helper functions


@overload
def result_type(
    first: onnx.CoreNumericDTypes, *others: onnx.CoreNumericDTypes
) -> onnx.CoreNumericDTypes: ...


@overload
def result_type(first: CoreDTypes, *others: CoreDTypes) -> CoreDTypes: ...


@overload
def result_type(first: DType, *others: DType) -> DType: ...


def result_type(first: DType, *others: DType) -> DType:
    def result_binary(a: DType, b: DType) -> DType:
        res1 = a._result_type(b)
        if res1 != NotImplemented:
            return res1
        return b._result_type(a)

    res = reduce(result_binary, others, first)
    if res == NotImplemented:
        raise TypeError("No common type found")
    return res


def from_numpy(np_dtype: np.dtype) -> CoreDTypes:
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


DTYPE = TypeVar("DTYPE", bound=DType)
