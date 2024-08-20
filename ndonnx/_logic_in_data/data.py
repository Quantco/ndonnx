# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeGuard, overload

import numpy as np
import spox.opset.ai.onnx.v21 as op
from spox import Var
from spox._future import operator_overloading
from typing_extensions import Self

from . import dtypes
from .dtypes import (
    CoreDTypes,
    DType,
    NCoreDTypes,
    _as_nullable,
    from_numpy,
    result_type,
)

if TYPE_CHECKING:
    from .array import OnnxShape


class Data(ABC):
    @abstractmethod
    def __init__(self): ...

    @abstractmethod
    def __getitem__(self, index) -> Self: ...

    @property
    @abstractmethod
    def dtype(self) -> DType: ...

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

    def astype(self, dtype: DType) -> Data:
        raise ValueError(f"Casting between `{self.dtype}` and `{dtype}` is undefine")

    def __add__(self, other: Data) -> Data:
        return NotImplemented

    def __or__(self, rhs: Data) -> Data:
        return NotImplemented

    def promote(self, *others: Data) -> Sequence[Data]:
        return NotImplemented


class _PyScalar(Data):
    value: int | float

    def __init__(self, value: int | float):
        self.value = value

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
        raise ValueError("Cannot reshape Python scalar.")

    def promote(self, *others: Data) -> Sequence[Data]:
        # TODO
        raise NotImplementedError

    def __add__(self, lhs: Data) -> CoreData:
        return NotImplemented

    def __or__(self, rhs: Data) -> CoreData:
        return NotImplemented


class _PyIntData(_PyScalar): ...


class _PyFloatData(_PyScalar): ...


class CoreData(Data):
    var: Var

    def __init__(self, var: Var):
        self.var = var

    def __getitem__(self, index) -> Self:
        # TODO
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self) -> CoreDTypes: ...

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
    def from_np_schema(cls, schema: dict[str, Any], /) -> CoreData:
        if "var" in schema and len(schema) == 1:
            (var,) = schema.values()
            return ascoredata(op.const(var))
        raise ValueError("'schema' has unexpected layout")

    def reshape(self, shape: tuple[int, ...]) -> CoreData:
        var = op.reshape(self.var, op.const(shape))
        return ascoredata(var)

    @overload
    def promote(self, *others: CoreData) -> Sequence[CoreData]: ...

    @overload
    def promote(self, *others: Data) -> Sequence[Data]: ...

    def promote(self, *others: Data) -> Sequence[Data]:
        # TODO
        raise NotImplementedError

    def as_core_dtype(self, dtype: CoreDTypes) -> CoreData:
        raise ValueError(f"Casting between `{self.dtype}` and `{dtype}` is undefine")

    def __add__(self, lhs: Data) -> CoreData:
        return NotImplemented

    def _promote(self, *others: Data) -> list[CoreData]:
        """Promote with other `CoreData` objects or return `NotImplemented`."""
        if is_sequence_of_core_data(others):
            res_type = result_type(self.dtype, *[d.dtype for d in others])
            return [self.as_core_dtype(res_type)] + [
                d.as_core_dtype(res_type) for d in others
            ]
        return NotImplemented

    def __or__(self, rhs: Data) -> CoreData:
        return NotImplemented


class CoreDataNumber(CoreData):
    def __add__(self, rhs: Data) -> CoreData:
        if isinstance(rhs, CoreDataNumber):
            # NOTE: Can't always promote for all data types (c.f. datetime / timedelta)
            a, b = self.promote(self, rhs)
            var = op.add(a.var, b.var)
            return ascoredata(var)
        return NotImplemented


class CoreDataInteger(CoreDataNumber):
    def __or__(self, rhs: Data) -> CoreData:
        if isinstance(rhs, CoreData):
            lhs: CoreData = self
            if lhs.dtype != rhs.dtype:
                lhs, rhs = lhs.promote(rhs)
                return lhs.__or__(rhs)

            # Data is core & integer
            var = op.bitwise_or(lhs.var, rhs.var)
            return ascoredata(var)
        return NotImplemented


class CoreDataBool(CoreData):
    def __or__(self, rhs: Data) -> CoreData:
        if isinstance(rhs, CoreData):
            lhs: CoreData = self
            if lhs.dtype != rhs.dtype:
                lhs, rhs = lhs.promote(rhs)
                return lhs.__or__(rhs)

            # Data is core & bool
            var = op.or_(lhs.var, rhs.var)
            return ascoredata(var)
        return NotImplemented


class Int32Data(CoreDataInteger):
    dtype = dtypes.int32


@dataclass
class NullableData(Data):
    data: Data
    mask: CoreData | None


@dataclass
class NullableCoreData(NullableData):
    data: CoreData  # Specialization of data from `Data` to `CoreData`

    @operator_overloading(op, True)
    def __add__(self, other: Data) -> NullableCoreData:
        if not isinstance(other, (CoreData, NullableCoreData)):
            return NotImplemented
        if isinstance(other, CoreData):
            other = NullableCoreData(other, None)

        data = self.data + other.data
        mask = _merge_masks(self.mask, other.mask)

        return NullableCoreData(data, mask)

    @operator_overloading(op, True)
    def __radd__(self, other: Data) -> NullableCoreData:
        # This is for instance called if we do CoreData + NullableCoreData
        y = self
        # We know how to convert from CoreData into NullableCoreData and this is the place to do so
        if isinstance(other, CoreData):
            x = NullableCoreData(other, None)
            return x + y

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

    def reshape(self, shape: tuple[int, ...]) -> NullableCoreData:
        data = self.data.reshape(shape)
        mask = self.mask.reshape(shape) if self.mask is not None else None
        return NullableCoreData(data, mask)

    @classmethod
    def from_np_schema(cls, schema: dict[str, Any], /) -> NullableCoreData:
        assert len(schema) == 2
        data = schema["data"]
        mask = schema["mask"]
        return asncoredata(op.const(data), op.const(mask))

    def __getitem__(self, index) -> Self:
        # TODO
        raise NotImplementedError

    @property
    def dtype(self) -> NCoreDTypes:
        return _as_nullable(self.data.dtype)


class NInt32Data(NullableCoreData):
    dtype = dtypes.nint32


def _merge_masks(a: CoreData | None, b: CoreData | None) -> CoreData | None:
    if a is None:
        return b
    if b is None:
        return a
    return a | b


def ascoredata(var: Var) -> CoreData:
    np_dtype = var.unwrap_tensor().dtype

    if np_dtype == np.int32:
        return Int32Data(var)

    raise NotImplementedError


def asncoredata(data: Var, mask: Var) -> NullableCoreData:
    np_dtype = data.unwrap_tensor().dtype
    dtype = from_numpy(np_dtype)
    try:
        mapping: dict[CoreDTypes, type[NullableCoreData]] = {dtypes.int32: NInt32Data}
        return mapping[dtype](ascoredata(data), ascoredata(mask))
    except KeyError:
        raise NotImplementedError


def is_sequence_of_core_data(seq: Sequence[Data]) -> TypeGuard[Sequence[CoreData]]:
    return all(isinstance(d, CoreData) for d in seq)
