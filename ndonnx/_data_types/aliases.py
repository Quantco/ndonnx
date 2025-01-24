# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ndonnx import CoreType

from .classes import (
    Boolean,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    NBoolean,
    NFloat32,
    NFloat64,
    NInt8,
    NInt16,
    NInt32,
    NInt64,
    NUInt8,
    NUInt16,
    NUInt32,
    NUInt64,
    NUtf8,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Utf8,
)

# Singleton exports of core data types
bool: Boolean = Boolean()
float32: Float32 = Float32()
float64: Float64 = Float64()
int16: Int16 = Int16()
int32: Int32 = Int32()
int64: Int64 = Int64()
int8: Int8 = Int8()
uint16: UInt16 = UInt16()
uint32: UInt32 = UInt32()
uint64: UInt64 = UInt64()
uint8: UInt8 = UInt8()
utf8: Utf8 = Utf8()

# Singleton exports of nullable instances of the core data types
nbool: NBoolean = NBoolean()
nfloat32: NFloat32 = NFloat32()
nfloat64: NFloat64 = NFloat64()
nint8: NInt8 = NInt8()
nint16: NInt16 = NInt16()
nint32: NInt32 = NInt32()
nint64: NInt64 = NInt64()
nuint8: NUInt8 = NUInt8()
nuint16: NUInt16 = NUInt16()
nuint32: NUInt32 = NUInt32()
nuint64: NUInt64 = NUInt64()
nutf8: NUtf8 = NUtf8()


def canonical_name(dtype: CoreType) -> str:
    """Return the canonical name of the data type."""
    if dtype == bool:
        return "bool"
    elif dtype == float32:
        return "float32"
    elif dtype == float64:
        return "float64"
    elif dtype == int8:
        return "int8"
    elif dtype == int16:
        return "int16"
    elif dtype == int32:
        return "int32"
    elif dtype == int64:
        return "int64"
    elif dtype == uint8:
        return "uint8"
    elif dtype == uint16:
        return "uint16"
    elif dtype == uint32:
        return "uint32"
    elif dtype == uint64:
        return "uint64"
    elif dtype == utf8:
        return "utf8"
    else:
        raise ValueError(f"Unknown data type: {dtype}")


def kinds(dtype: CoreType) -> tuple[str, ...]:
    """Return the kinds of the data type."""
    if dtype in (bool,):
        return ("bool",)
    if dtype in (int8, int16, int32, int64):
        return ("signed integer", "integer", "numeric")
    if dtype in (uint8, uint16, uint32, uint64):
        return ("unsigned integer", "integer", "numeric")
    if dtype in (float32, float64):
        return ("floating", "numeric")
    if dtype in (utf8,):
        raise ValueError(f"We don't get define a kind for {dtype}")
    else:
        raise ValueError(f"Unknown data type: {dtype}")
