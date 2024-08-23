# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from types import NotImplementedType
from typing import TYPE_CHECKING, Any, Generic, TypeGuard, TypeVar, overload

import numpy as np
import spox.opset.ai.onnx.v21 as op
from spox import Var
from spox._future import operator_overloading
from typing_extensions import Self

from . import dtypes
from .dtypes import (
    CoreDTypes,
    DType,
    result_type,
)

if TYPE_CHECKING:
    from .array import OnnxShape


DTYPE = TypeVar("DTYPE", bound=DType)
CORE_DTYPES = TypeVar("CORE_DTYPES", bound=CoreDTypes)
ALL_NUM_DTYPES = TypeVar(
    "ALL_NUM_DTYPES", bound=dtypes.CoreNumericDTypes | dtypes.NCoreNumericDTypes
)


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


class _ArrayPyScalar(_TypedArray[DTYPE]):
    value: int | float

    def __init__(self, value: int | float):
        self.value = value

    @classmethod
    def from_data(cls, data: _TypedArray):
        # TODO
        raise NotImplementedError

    def __getitem__(self, index) -> Self:
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> OnnxShape:
        return ()

    @classmethod
    def from_np_schema(cls, schema: dict[str, Any], /) -> Self:
        if "value" in schema and len(schema) == 1:
            (val,) = schema.values()
            return cls(val)
        raise ValueError("'schema' has unexpected layout")

    def reshape(self, shape: tuple[int, ...]) -> Self:
        # TODO: Should reshape be moved into a different base class?
        raise ValueError("cannot reshape Python scalar")

    def promote(self, *others: _TypedArray) -> Sequence[_TypedArray]:
        # TODO
        raise NotImplementedError

    def _promote(self, other: _ArrayCoreType) -> tuple[_ArrayCoreType, _ArrayCoreType]:
        result_type = self.dtype._result_type(other.dtype)

        self.astype(result_type)
        raise NotImplementedError

    def __add__(self, rhs: _TypedArray) -> _ArrayCoreType:
        if isinstance(rhs, _ArrayCoreNum):
            lhs, rhs = self._promote(rhs)
            return lhs + rhs
        return NotImplemented

    def __or__(self, rhs: _TypedArray) -> _ArrayCoreType:
        return NotImplemented

    def _astype(self, dtype: DType) -> _TypedArray:
        # TODO
        raise NotImplementedError


class _ArrayPyInt(_ArrayPyScalar[dtypes._PyInt]):
    dtype = dtypes._pyint


class _ArrayPyFloat(_ArrayPyScalar[dtypes._PyFloat]):
    dtype = dtypes._pyfloat


class _ArrayCoreType(_TypedArray[CORE_DTYPES]):
    var: Var

    def __init__(self, var: Var):
        self.var = var

    @classmethod
    def from_data(cls, data: _TypedArray):
        # TODO
        raise NotImplementedError

    def __getitem__(self, index) -> Self:
        # TODO
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> OnnxShape:
        shape = self.var.unwrap_tensor().shape
        if shape is None:
            raise ValueError("Missing shape information")
        return shape

    @classmethod
    def from_np_schema(cls, schema: dict[str, Any], /) -> _ArrayCoreType:
        if "var" in schema and len(schema) == 1:
            (var,) = schema.values()
            return ascoredata(op.const(var))
        raise ValueError("'schema' has unexpected layout")

    def reshape(self, shape: tuple[int, ...]) -> _ArrayCoreType:
        var = op.reshape(self.var, op.const(shape))
        return ascoredata(var)

    @overload
    def promote(self, *others: _ArrayCoreType) -> Sequence[_ArrayCoreType]: ...

    @overload
    def promote(self, *others: _TypedArray) -> Sequence[_TypedArray]: ...

    def promote(self, *others: _TypedArray) -> Sequence[_TypedArray]:
        # TODO
        raise NotImplementedError

    def as_core_dtype(self, dtype: CoreDTypes) -> _ArrayCoreType:
        raise ValueError(f"Casting between `{self.dtype}` and `{dtype}` is undefine")

    def __add__(self, lhs: _TypedArray) -> _ArrayCoreType:
        return NotImplemented

    def _promote(self, *others: _TypedArray) -> list[_ArrayCoreType]:
        """Promote with other `_ArrayCoreType` objects or return `NotImplemented`."""
        if is_sequence_of_core_data(others):
            res_type = result_type(self.dtype, *[d.dtype for d in others])
            return [self.as_core_dtype(res_type)] + [
                d.as_core_dtype(res_type) for d in others
            ]
        return NotImplemented

    def __or__(self, rhs: _TypedArray) -> _ArrayCoreType:
        return NotImplemented

    def _astype(self, dtype: DType) -> _TypedArray:
        # TODO
        raise NotImplementedError


class _ArrayCoreNum(_ArrayCoreType[CORE_DTYPES]):
    def __add__(self, rhs: _TypedArray) -> _ArrayCoreType:
        if isinstance(rhs, _ArrayCoreNum):
            # NOTE: Can't always promote for all data types (c.f. datetime / timedelta)
            a, b = self.promote(self, rhs)
            var = op.add(a.var, b.var)
            return ascoredata(var)
        return NotImplemented


class _ArrayCoreInteger(_ArrayCoreNum[CORE_DTYPES]):
    def __or__(self, rhs: _TypedArray) -> _ArrayCoreType:
        if isinstance(rhs, _ArrayCoreType):
            lhs: _ArrayCoreType = self
            if lhs.dtype != rhs.dtype:
                lhs, rhs = lhs.promote(rhs)
                return lhs.__or__(rhs)

            # Data is core & integer
            var = op.bitwise_or(lhs.var, rhs.var)
            return ascoredata(var)
        return NotImplemented


class _ArrayCoreFloating(_ArrayCoreNum[CORE_DTYPES]): ...


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


class BoolData(_ArrayCoreType[dtypes.Bool]):
    dtype = dtypes.bool_

    def __or__(self, rhs: _TypedArray) -> _ArrayCoreType:
        if isinstance(rhs, _ArrayCoreType):
            lhs: _ArrayCoreType = self
            if lhs.dtype != rhs.dtype:
                lhs, rhs = lhs.promote(rhs)
                return lhs.__or__(rhs)

            # Data is core & bool
            var = op.or_(lhs.var, rhs.var)
            return ascoredata(var)
        return NotImplemented


class Int8Data(_ArrayCoreInteger[dtypes.Int8]):
    dtype = dtypes.int8


class Int16Data(_ArrayCoreInteger[dtypes.Int16]):
    dtype = dtypes.int16


class Int32Data(_ArrayCoreInteger[dtypes.Int32]):
    dtype = dtypes.int32


class Int64Data(_ArrayCoreInteger[dtypes.Int64]):
    dtype = dtypes.int64


class Uint8Data(_ArrayCoreInteger[dtypes.Uint8]):
    dtype = dtypes.uint8


class Uint16Data(_ArrayCoreInteger[dtypes.Uint16]):
    dtype = dtypes.uint16


class Uint32Data(_ArrayCoreInteger[dtypes.Uint32]):
    dtype = dtypes.uint32


class Uint64Data(_ArrayCoreInteger[dtypes.Uint64]):
    dtype = dtypes.uint64


class Float16Data(_ArrayCoreFloating[dtypes.Float16]):
    dtype = dtypes.float16


class Float32Data(_ArrayCoreFloating[dtypes.Float32]):
    dtype = dtypes.float32


class Float64Data(_ArrayCoreFloating[dtypes.Float64]):
    dtype = dtypes.float64


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


def ascoredata(var: Var) -> _ArrayCoreType:
    np_dtype = var.unwrap_tensor().dtype

    if np_dtype == np.int32:
        return Int32Data(var)

    raise NotImplementedError


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


def is_sequence_of_core_data(
    seq: Sequence[_TypedArray],
) -> TypeGuard[Sequence[_ArrayCoreType]]:
    return all(isinstance(d, _ArrayCoreType) for d in seq)
