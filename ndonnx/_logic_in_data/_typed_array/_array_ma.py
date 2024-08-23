# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import spox.opset.ai.onnx.v21 as op
from spox._future import operator_overloading
from typing_extensions import Self

from .. import dtypes
from ..dtypes import (
    CoreDTypes,
    DType,
)
from ._core_types import Int8Data, _ArrayCoreType, ascoredata
from ._typed_array import _TypedArray

if TYPE_CHECKING:
    from ..array import OnnxShape


DTYPE = TypeVar("DTYPE", bound=DType)
CORE_DTYPES = TypeVar("CORE_DTYPES", bound=CoreDTypes)
ALL_NUM_DTYPES = TypeVar(
    "ALL_NUM_DTYPES", bound=dtypes.CoreNumericDTypes | dtypes.NCoreNumericDTypes
)


@dataclass
class _ArrayMa(_TypedArray[DTYPE]):
    data: _TypedArray
    mask: _ArrayCoreType | None


@dataclass
class _ArrayMaCoreType(_ArrayMa[DTYPE]):
    data: _ArrayCoreType  # Specialization of data from `Data` to `_ArrayCoreType`

    @classmethod
    def from_data(cls, data: _TypedArray):
        # TODO
        raise NotImplementedError

    def __add__(self, other: _TypedArray) -> _ArrayMaCoreType:
        if not isinstance(other, (_ArrayCoreType, _ArrayMaCoreType)):
            return NotImplemented

        other_data = other if isinstance(other, _ArrayCoreType) else other.data
        data = self.data + other_data
        mask = _merge_masks(
            self.mask, other.mask if isinstance(other, _ArrayMaCoreType) else None
        )

        return asncoredata(data, mask)

    @operator_overloading(op, True)
    def __radd__(self, lhs: _TypedArray) -> _ArrayMaCoreType:
        # This is for instance called if we do _ArrayCoreType +
        # _ArrayMaCoreType We know how to convert from
        # _ArrayCoreType into _ArrayMaCoreType and this is the
        # place to do so
        if isinstance(lhs, _ArrayCoreType):
            return asncoredata(lhs, None) + self

        return NotImplemented

    @property
    def shape(self) -> OnnxShape:
        shape = self.data.shape
        if shape is None:
            raise ValueError("Missing shape information")
        return shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def reshape(self, shape: tuple[int, ...]) -> _ArrayMaCoreType:
        data = self.data.reshape(shape)
        mask = self.mask.reshape(shape) if self.mask is not None else None
        return type(self)(data, mask)

    @classmethod
    def from_np_schema(cls, schema: dict[str, Any], /) -> _ArrayMaCoreType:
        assert len(schema) == 2
        data = schema["data"]
        mask = schema["mask"]
        return asncoredata(ascoredata(op.const(data)), ascoredata(op.const(mask)))

    def __getitem__(self, index) -> Self:
        # TODO
        raise NotImplementedError

    def _astype(self, dtype: DType) -> _TypedArray:
        raise NotImplementedError


class NBoolData(_ArrayMaCoreType[dtypes.NBool]):
    dtype = dtypes.nbool


class NInt8Data(_ArrayMaCoreType[dtypes.NInt8]):
    dtype = dtypes.nint8


class NInt16Data(_ArrayMaCoreType[dtypes.NInt16]):
    dtype = dtypes.nint16


class NInt32Data(_ArrayMaCoreType[dtypes.NInt32]):
    dtype = dtypes.nint32


class NInt64Data(_ArrayMaCoreType[dtypes.NInt64]):
    dtype = dtypes.nint64


class NUint8Data(_ArrayMaCoreType[dtypes.NUint8]):
    dtype = dtypes.nuint8


class NUint16Data(_ArrayMaCoreType[dtypes.NUint16]):
    dtype = dtypes.nuint16


class NUint32Data(_ArrayMaCoreType[dtypes.NUint32]):
    dtype = dtypes.nuint32


class NUint64Data(_ArrayMaCoreType[dtypes.NUint64]):
    dtype = dtypes.nuint64


class NFloat16Data(_ArrayMaCoreType[dtypes.NFloat16]):
    dtype = dtypes.nfloat16


class NFloat32Data(_ArrayMaCoreType[dtypes.NFloat32]):
    dtype = dtypes.nfloat32


class NFloat64Data(_ArrayMaCoreType[dtypes.NFloat64]):
    dtype = dtypes.nfloat64


def _merge_masks(
    a: _ArrayCoreType | None, b: _ArrayCoreType | None
) -> _ArrayCoreType | None:
    if a is None:
        return b
    if b is None:
        return a
    return a | b


def asncoredata(data: _ArrayCoreType, mask: _ArrayCoreType | None) -> _ArrayMaCoreType:
    try:
        mapping = {dtypes.int32: NInt32Data}
        return mapping[data.dtype](data, mask)
    except KeyError:
        raise NotImplementedError


def core_to_ncore(core: _ArrayCoreType) -> _ArrayMaCoreType:
    if isinstance(core, Int8Data):
        return NInt8Data(data=core, mask=None)
    raise ValueError
