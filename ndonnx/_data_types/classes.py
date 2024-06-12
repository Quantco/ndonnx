# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
from typing_extensions import Self

import ndonnx as ndx
from ndonnx._core import CoreOperationsImpl, OperationsBlock
from ndonnx._data_types.schema import Schema

from .conversion import CastError, CastMixin
from .coretype import CoreType
from .structtype import StructType

if TYPE_CHECKING:
    from ndonnx import Array


class Numerical(CoreType):
    """Base class for numerical data types."""


class Integral(Numerical):
    """Base class for integral data types."""


class Unsigned(Integral):
    """Base class for unsigned integral data types."""


class Fractional(Numerical):
    """Base class for fractional data types."""


class Floating(Fractional):
    """Base class for floating data types."""


class Int8(Integral):
    """8-bit signed integer."""

    @staticmethod
    def to_numpy_dtype() -> np.dtype:
        return np.dtype("int8")


class Int16(Integral):
    """16-bit signed integer."""

    @staticmethod
    def to_numpy_dtype() -> np.dtype:
        return np.dtype("int16")


class Int32(Integral):
    """32-bit signed integer."""

    @staticmethod
    def to_numpy_dtype() -> np.dtype:
        return np.dtype("int32")


class Int64(Integral):
    """64-bit signed integer."""

    @staticmethod
    def to_numpy_dtype() -> np.dtype:
        return np.dtype("int64")


class UInt8(Unsigned):
    """8-bit unsigned integer."""

    @staticmethod
    def to_numpy_dtype() -> np.dtype:
        return np.dtype("uint8")


class UInt16(Unsigned):
    """16-bit unsigned integer."""

    @staticmethod
    def to_numpy_dtype() -> np.dtype:
        return np.dtype("uint16")


class UInt32(Unsigned):
    """32-bit unsigned integer."""

    @staticmethod
    def to_numpy_dtype() -> np.dtype:
        return np.dtype("uint32")


class UInt64(Unsigned):
    """64-bit unsigned integer."""

    @staticmethod
    def to_numpy_dtype() -> np.dtype:
        return np.dtype("uint64")


class Float32(Floating):
    """32-bit floating point."""

    @staticmethod
    def to_numpy_dtype() -> np.dtype:
        return np.dtype("float32")


class Float64(Floating):
    """64-bit floating point."""

    @staticmethod
    def to_numpy_dtype() -> np.dtype:
        return np.dtype("float64")


class Boolean(CoreType):
    """Boolean type."""

    @staticmethod
    def to_numpy_dtype() -> np.dtype:
        return np.dtype("bool")


class Utf8(CoreType):
    """UTF-8 encoded string."""

    @staticmethod
    def to_numpy_dtype() -> np.dtype:
        return np.dtype("str")

    def _parse_input(self, data: np.ndarray) -> dict[str, np.ndarray]:
        if data.dtype.kind == "U" or (
            data.dtype.kind == "O" and all(isinstance(x, str) for x in data)
        ):
            return {"data": data.astype(np.str_)}
        else:
            raise TypeError(f"Expected data type with kind 'U', got {data.dtype}")

    def _assemble_output(self, fields: dict[str, np.ndarray]) -> np.ndarray:
        if "data" not in fields:
            raise ValueError(
                f"Missing fields in output: {fields.keys()}, expected `data`"
            )
        if fields["data"].dtype.kind == "O" and all(
            isinstance(x, str) for x in fields["data"]
        ):
            return fields["data"].astype(np.str_)
        elif fields["data"].dtype.kind == "U":
            return fields["data"]
        else:
            raise TypeError(
                f"Expected data type with kind 'U', got {fields['data'].dtype}"
            )


T = TypeVar("T", StructType, CoreType, covariant=True)


class Nullable(StructType, Generic[T]):
    values: T
    null: Boolean

    def _fields(self) -> dict[str, StructType | CoreType]:
        return {
            "values": self.values,
            "null": self.null,
        }


class _NullableCore(Nullable[CoreType], CastMixin):
    def copy(self) -> Self:
        return self

    def _parse_input(self, input: np.ndarray) -> dict:
        if not isinstance(input, np.ma.MaskedArray):
            raise TypeError(f"Expected numpy MaskedArray, got {type(input)}")
        return {
            "values": self.values._parse_input(input.data),
            "null": self.null._parse_input(input.mask),
        }

    def _assemble_output(self, fields: dict[str, np.ndarray]) -> np.ndarray:
        if "values" not in fields or "null" not in fields:
            raise ValueError(
                f"Missing fields in output: {fields.keys()}, expected `null` and `values`"
            )

        return np.ma.masked_array(fields["values"], mask=fields["null"])

    def _schema(self) -> Schema:
        return Schema(type_name=type(self).__name__, author="ndonnx")

    def _cast_to(self, array: Array, dtype: CoreType | StructType) -> Array:
        if isinstance(dtype, _NullableCore):
            return ndx.Array._from_fields(
                dtype,
                values=self.values._cast_to(array.values, dtype.values),
                null=array.null.copy(),
            )
        else:
            raise CastError(f"Cannot cast to {dtype} from {self}")

    def _cast_from(self, array: Array) -> Array:
        if isinstance(array.dtype, CoreType):
            # Promote to nullable variant
            return ndx.Array._from_fields(
                self,
                values=self.values._cast_from(array),
                null=ndx.zeros_like(array, dtype=Boolean()),
            )
        elif isinstance(array.dtype, _NullableCore):
            return ndx.Array._from_fields(
                self,
                values=self.values._cast_from(array.values),
                null=array.null.copy(),
            )
        else:
            raise CastError(f"Cannot cast from {array.dtype} to {self}")

    _ops: OperationsBlock = CoreOperationsImpl()


class NullableNumerical(_NullableCore):
    """Base class for nullable numerical data types."""


class NullableIntegral(NullableNumerical):
    """Base class for nullable integral data types."""


class NullableUnsigned(NullableIntegral):
    """Base class for nullable unsigned integral data types."""


class NullableFractional(NullableNumerical):
    """Base class for nullable fractional data types."""


class NullableFloating(NullableFractional):
    """Base class for nullable floating data types."""


class NInt8(NullableIntegral):
    values = Int8()
    null = Boolean()


class NInt16(NullableIntegral):
    values = Int16()
    null = Boolean()


class NInt32(NullableIntegral):
    values = Int32()
    null = Boolean()


class NInt64(NullableIntegral):
    values = Int64()
    null = Boolean()


class NUInt8(NullableUnsigned):
    values = UInt8()
    null = Boolean()


class NUInt16(NullableUnsigned):
    values = UInt16()
    null = Boolean()


class NUInt32(NullableUnsigned):
    values = UInt32()
    null = Boolean()


class NUInt64(NullableUnsigned):
    values = UInt64()
    null = Boolean()


class NFloat32(NullableFloating):
    values = Float32()
    null = Boolean()


class NFloat64(NullableFloating):
    values = Float64()
    null = Boolean()


class NBoolean(_NullableCore):
    values = Boolean()
    null = Boolean()


class NUtf8(_NullableCore):
    values = Utf8()
    null = Boolean()


def from_numpy_dtype(np_dtype: np.dtype) -> CoreType:
    """Convert a numpy dtype to an ndonnx data type."""
    if np_dtype == np.dtype("int8"):
        return Int8()
    if np_dtype == np.dtype("int16"):
        return Int16()
    if np_dtype == np.dtype("int32"):
        return Int32()
    if np_dtype == np.dtype("int64"):
        return Int64()
    if np_dtype == np.dtype("uint8"):
        return UInt8()
    if np_dtype == np.dtype("uint16"):
        return UInt16()
    if np_dtype == np.dtype("uint32"):
        return UInt32()
    if np_dtype == np.dtype("uint64"):
        return UInt64()
    if np_dtype == np.dtype("float32"):
        return Float32()
    if np_dtype == np.dtype("float64"):
        return Float64()
    if np_dtype == np.dtype("bool"):
        return Boolean()
    if np_dtype == np.dtype("str") or np_dtype.kind in ("U", "O"):
        return Utf8()
    raise TypeError(f"Unsupported data type: {np_dtype}")


@dataclass
class Iinfo:
    # Number of bits occupied by the type.
    bits: int
    # Largest representable number.
    max: int
    # Smallest representable number.
    min: int
    # Integer data type.
    dtype: CoreType

    @classmethod
    def _from_dtype(cls, dtype: CoreType) -> Iinfo:
        iinfo = np.iinfo(dtype.to_numpy_dtype())
        return cls(
            bits=iinfo.bits,
            max=iinfo.max,
            min=iinfo.min,
            dtype=dtype,
        )


@dataclass
class Finfo:
    # number of bits occupied by the real-valued floating-point data type.
    bits: int
    # difference between 1.0 and the next smallest representable real-valued floating-point number larger than 1.0 according to the IEEE-754 standard.
    eps: float
    # largest representable real-valued number.
    max: float
    # smallest representable real-valued number.
    min: float
    # smallest positive real-valued floating-point number with full precision.
    smallest_normal: float
    # real-valued floating-point data type.
    dtype: CoreType

    @classmethod
    def _from_dtype(cls, dtype: CoreType) -> Finfo:
        finfo = np.finfo(dtype.to_numpy_dtype())
        return cls(
            bits=finfo.bits,
            max=float(finfo.max),
            min=float(finfo.min),
            dtype=dtype,
            eps=float(finfo.eps),
            smallest_normal=float(finfo.smallest_normal),
        )


def get_finfo(dtype: _NullableCore | CoreType) -> Finfo:
    try:
        if isinstance(dtype, _NullableCore):
            dtype = dtype.values
        return Finfo._from_dtype(dtype)
    except KeyError:
        raise TypeError(f"'{dtype}' is not a floating point data type.")


def get_iinfo(dtype: _NullableCore | CoreType) -> Iinfo:
    try:
        if isinstance(dtype, _NullableCore):
            dtype = dtype.values
        return Iinfo._from_dtype(dtype)
    except KeyError:
        raise TypeError(f"'{dtype}' is not an integer data type.")
