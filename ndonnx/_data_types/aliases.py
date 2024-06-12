# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

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
