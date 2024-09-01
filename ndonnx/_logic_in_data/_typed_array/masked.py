# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import spox.opset.ai.onnx.v21 as op
from typing_extensions import Self

from .. import dtypes
from ..dtypes import CoreDTypes, DType, NCoreDTypes, as_non_nullable
from .core import BoolData, Int8Data, _ArrayCoreType, ascoredata
from .typed_array import _TypedArray

if TYPE_CHECKING:
    from ..array import Index, OnnxShape


DTYPE = TypeVar("DTYPE", bound=DType)
CORE_DTYPES = TypeVar("CORE_DTYPES", bound=CoreDTypes)
NCORE_DTYPES = TypeVar("NCORE_DTYPES", bound=NCoreDTypes)
ALL_NUM_DTYPES = TypeVar(
    "ALL_NUM_DTYPES", bound=dtypes.CoreNumericDTypes | dtypes.NCoreNumericDTypes
)


@dataclass
class _ArrayMa(_TypedArray[DTYPE]):
    """Typed masked array object."""

    data: _TypedArray
    mask: BoolData | None


@dataclass
class _ArrayMaCoreType(_ArrayMa[NCORE_DTYPES]):
    """Masked version of core types.

    The (subclasses) of this class are implemented such that they only
    know about `_ArrayCoreType`s, but not `_PyScalar`s.
    """

    # Specialization of data from `Data` to `_ArrayCoreType`
    data: _ArrayCoreType

    @classmethod
    def from_typed_array(cls, tyarr: _TypedArray):
        if isinstance(tyarr, _ArrayCoreType):
            return asncoredata(tyarr, mask=None)
        if isinstance(tyarr, _ArrayMaCoreType):
            new_data = tyarr.data.astype(cls.dtype._unmasked_dtype)
            return cls(new_data, mask=tyarr.mask)

        return NotImplemented

    @classmethod
    def as_argument(cls, shape: OnnxShape):
        data = as_non_nullable(cls.dtype)._tyarr_class.as_argument(shape)
        mask = BoolData.as_argument(shape)
        return cls(data, mask)

    def __add__(self, other: _TypedArray) -> _ArrayMaCoreType:
        return _apply_op(self, other, operator.add)

    def __radd__(self, lhs: _TypedArray) -> _ArrayMaCoreType:
        # This is for instance called if we do _ArrayCoreType +
        # _ArrayMaCoreType. We know how to convert from
        # _ArrayCoreType into _ArrayMaCoreType and this is the
        # place to do so.
        if isinstance(lhs, _ArrayCoreType):
            return asncoredata(lhs, None) + self

        return NotImplemented

    @property
    def shape(self) -> OnnxShape:
        shape = self.data.shape
        if shape is None:
            raise ValueError("Missing shape information")
        return shape

    def reshape(self, shape: tuple[int, ...]) -> Self:
        data = self.data.reshape(shape)
        mask = self.mask.reshape(shape) if self.mask is not None else None
        return type(self)(data, mask)

    @classmethod
    def from_np_schema(cls, schema: dict[str, Any], /) -> _ArrayMaCoreType:
        assert len(schema) == 2
        data = schema["data"]
        mask = schema["mask"]
        return asncoredata(ascoredata(op.const(data)), BoolData(op.const(mask)))

    def __getitem__(self, index: Index) -> Self:
        new_data = self.data[index]
        new_mask = self.mask[index] if self.mask is not None else None

        return type(self)(data=new_data, mask=new_mask)

    def _astype(self, dtype: DType) -> _TypedArray:
        # Implemented under the assumption that we know about core, but not py_scalars
        if isinstance(dtype, dtypes.CoreDTypes):
            # TODO: Not clear what the behavior should be if we have a mask
            raise NotImplementedError
        elif isinstance(dtype, dtypes.NCoreDTypes):
            new_data = self.data.astype(dtype._unmasked_dtype)
            dtype._tyarr_class(data=new_data, mask=self.mask)
        return NotImplemented


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


def _merge_masks(a: BoolData | None, b: BoolData | None) -> BoolData | None:
    if a is None:
        return b
    if b is None:
        return a
    out = a | b
    if isinstance(out, BoolData):
        return out
    # Should never happen
    raise TypeError("Unexpected array type")


def asncoredata(core_array: _TypedArray, mask: BoolData | None) -> _ArrayMaCoreType:
    from . import core

    if isinstance(core_array, core.Int8Data):
        return NInt8Data(core_array, mask)
    if isinstance(core_array, core.Int16Data):
        return NInt16Data(core_array, mask)
    if isinstance(core_array, core.Int32Data):
        return NInt32Data(core_array, mask)
    if isinstance(core_array, core.Int64Data):
        return NInt64Data(core_array, mask)

    if isinstance(core_array, core.Uint8Data):
        return NUint8Data(core_array, mask)
    if isinstance(core_array, core.Uint16Data):
        return NUint16Data(core_array, mask)
    if isinstance(core_array, core.Uint32Data):
        return NUint32Data(core_array, mask)
    if isinstance(core_array, core.Uint64Data):
        return NUint64Data(core_array, mask)

    if isinstance(core_array, core.Float16Data):
        return NFloat16Data(core_array, mask)
    if isinstance(core_array, core.Float32Data):
        return NFloat32Data(core_array, mask)
    if isinstance(core_array, core.Float64Data):
        return NFloat64Data(core_array, mask)

    if isinstance(core_array, core.BoolData):
        return NBoolData(core_array, mask)

    raise TypeError("expected '_ArrayCoreType' found `{type(core_array)}`")


def core_to_ncore(core: _ArrayCoreType) -> _ArrayMaCoreType:
    if isinstance(core, Int8Data):
        return NInt8Data(data=core, mask=None)
    raise ValueError


def unmask_core(arr: _ArrayCoreType | _ArrayMaCoreType) -> _ArrayCoreType:
    if isinstance(arr, _ArrayCoreType):
        return arr
    return arr.data


def _apply_op(
    lhs: _ArrayMaCoreType,
    rhs: _TypedArray,
    op: Callable[[_ArrayCoreType, _ArrayCoreType], _ArrayCoreType],
) -> _ArrayMaCoreType:
    """Apply an operation by passing it through to the data member."""
    if isinstance(rhs, _ArrayCoreType):
        data = lhs.data + rhs
        mask = lhs.mask
    elif isinstance(rhs, _ArrayMaCoreType):
        data = lhs.data + rhs.data
        mask = _merge_masks(lhs.mask, rhs.mask)
    else:
        return NotImplemented

    return asncoredata(data, mask)
