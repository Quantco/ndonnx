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


_canonical_names = {
    bool: "bool",
    float32: "float32",
    float64: "float64",
    int8: "int8",
    int16: "int16",
    int32: "int32",
    int64: "int64",
    uint8: "uint8",
    uint16: "uint16",
    uint32: "uint32",
    uint64: "uint64",
    utf8: "utf8",
}


def canonical_name(dtype: CoreType) -> str:
    """Return the canonical name of the data type."""
    if dtype in _canonical_names:
        return _canonical_names[dtype]
    else:
        raise ValueError(f"Unknown data type: {dtype}")


_kinds = {
    bool: ("bool",),
    int8: ("signed integer", "integral", "numeric"),
    int16: ("signed integer", "integral", "numeric"),
    int32: ("signed integer", "integral", "numeric"),
    int64: ("signed integer", "integral", "numeric"),
    uint8: ("unsigned integer", "integral", "numeric"),
    uint16: ("unsigned integer", "integral", "numeric"),
    uint32: ("unsigned integer", "integral", "numeric"),
    uint64: ("unsigned integer", "integral", "numeric"),
    float32: ("real floating", "numeric"),
    float64: ("real floating", "numeric"),
}


def kinds(dtype: CoreType) -> tuple[str, ...]:
    """Return the kinds of the data type."""
    if dtype in _kinds:
        return _kinds[dtype]
    elif dtype == utf8:
        raise ValueError(f"We don't yet define a kind for {dtype}")
    else:
        raise ValueError(f"Unknown data type: {dtype}")
