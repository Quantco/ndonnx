# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from types import NotImplementedType
from typing import overload


class DType(ABC):
    @abstractmethod
    def _result_type(self, other: DType) -> DType | NotImplementedType: ...

    def _rresult_type(self, other: DType) -> DType | NotImplementedType:
        return NotImplemented


class _CoreDType(DType): ...


class _NCoreDType(DType): ...


class _Number(_CoreDType):
    def _result_type(self, rhs: DType) -> DType | NotImplementedType:
        if isinstance(self, CoreNumericDTypes) and isinstance(rhs, CoreNumericDTypes):
            return _result_type_core_numeric(self, rhs)

        return NotImplemented


class Bool(_CoreDType):
    def _result_type(self, rhs: DType) -> DType | NotImplementedType:
        return NotImplemented


class NBool(_NCoreDType):
    def _result_type(self, rhs: DType) -> DType | NotImplementedType:
        return NotImplemented


class Int8(_Number): ...


class Int16(_Number): ...


class Int32(_Number): ...


class Int64(_Number): ...


class Uint8(_Number): ...


class Uint16(_Number): ...


class Uint32(_Number): ...


class Uint64(_Number): ...


class Float16(_Number): ...


class Float32(_Number): ...


class Float64(_Number): ...


class _NNumber(_NCoreDType):
    _core_type: CoreNumericDTypes

    def _result_type(self, rhs: DType) -> DType | NotImplementedType:
        if isinstance(self, NCoreNumericDTypes) and isinstance(rhs, CoreNumericDTypes):
            core_result = _result_type_core_numeric(self._core_type, rhs)
        elif isinstance(rhs, NCoreNumericDTypes):
            core_result = _result_type_core_numeric(self._core_type, rhs._core_type)

        else:
            # No implicit promotion for bools and strings
            return NotImplemented

        return _as_nullable(core_result)

    def _rresult_type(self, lhs: DType) -> DType | NotImplementedType:
        # e.g. called after `CoreDType._result_type(self)`

        if isinstance(lhs, _CoreDType | _NCoreDType):
            # All core types are cummutative
            return self._result_type(lhs)

        return NotImplemented


class NInt8(_NNumber): ...


class NInt16(_NNumber): ...


class NInt32(_NNumber): ...


class NInt64(_NNumber): ...


class NUint8(_NNumber): ...


class NUint16(_NNumber): ...


class NUint32(_NNumber): ...


class NUint64(_NNumber): ...


class NFloat16(_NNumber): ...


class NFloat32(_NNumber): ...


class NFloat64(_NNumber): ...


# Singleton instances
bool_ = Bool()
nbool = NBool()
int8 = Int8()
int16 = Int16()
int32 = Int32()
int64 = Int64()
uint8 = Uint8()
uint16 = Uint16()
uint32 = Uint32()
uint64 = Uint64()
float16 = Float16()
float32 = Float32()
float64 = Float64()
nint8 = NInt8()
nint16 = NInt16()
nint32 = NInt32()
nint64 = NInt64()
nuint8 = NUint8()
nuint16 = NUint16()
nuint32 = NUint32()
nuint64 = NUint64()
nfloat16 = NFloat16()
nfloat32 = NFloat32()
nfloat64 = NFloat64()

# Union types
#
# Union types are exhaustive and don't create ambiguities with respect to user-defined subtypes.
CoreNumericDTypes = (
    Int8
    | Int16
    | Int32
    | Int64
    | Uint8
    | Uint16
    | Uint32
    | Uint64
    | Float16
    | Float32
    | Float64
)

CoreDTypes = Bool | CoreNumericDTypes

NCoreNumericDTypes = (
    NInt8
    | NInt16
    | NInt32
    | NInt64
    | NUint8
    | NUint16
    | NUint32
    | NUint64
    | NFloat16
    | NFloat32
    | NFloat64
)

NCoreDTypes = NBool | NCoreNumericDTypes


# Promotion tables taken from:
# https://data-apis.org/array-api/draft/API_specification/type_promotion.html#type-promotion
# and
# https://numpy.org/neps/nep-0050-scalar-promotion.html#motivation-and-scope
_signed_signed: dict[tuple[CoreNumericDTypes, CoreNumericDTypes], CoreNumericDTypes] = {
    (int8, int8): int8,
    (int16, int8): int16,
    (int32, int8): int32,
    (int64, int8): int64,
    (int8, int16): int16,
    (int16, int16): int16,
    (int32, int16): int32,
    (int64, int16): int64,
    (int8, int32): int32,
    (int16, int32): int32,
    (int32, int32): int32,
    (int64, int32): int64,
    (int8, int64): int64,
    (int16, int64): int64,
    (int32, int64): int64,
    (int64, int64): int64,
}
_unsigned_unsigned: dict[
    tuple[CoreNumericDTypes, CoreNumericDTypes], CoreNumericDTypes
] = {
    (uint8, uint8): uint8,
    (uint16, uint8): uint16,
    (uint32, uint8): uint32,
    (uint64, uint8): uint64,
    (uint8, uint16): uint16,
    (uint16, uint16): uint16,
    (uint32, uint16): uint32,
    (uint64, uint16): uint64,
    (uint8, uint32): uint32,
    (uint16, uint32): uint32,
    (uint32, uint32): uint32,
    (uint64, uint32): uint64,
    (uint8, uint64): uint64,
    (uint16, uint64): uint64,
    (uint32, uint64): uint64,
    (uint64, uint64): uint64,
}
_mixed_integers: dict[
    tuple[CoreNumericDTypes, CoreNumericDTypes], CoreNumericDTypes
] = {
    (int8, uint8): int16,
    (int16, uint8): int16,
    (int32, uint8): int32,
    (int64, uint8): int64,
    (int8, uint16): int32,
    (int16, uint16): int32,
    (int32, uint16): int32,
    (int64, uint16): int64,
    (int8, uint32): int64,
    (int16, uint32): int64,
    (int32, uint32): int64,
    (int64, uint32): int64,
    # NOTE: Standard does not define interaction with uint64!
}

_floating_floating: dict[
    tuple[CoreNumericDTypes, CoreNumericDTypes], CoreNumericDTypes
] = {
    (float32, float32): float32,
    (float32, float64): float64,
    (float64, float32): float64,
    (float64, float64): float64,
}

# Non-standard interactions
_non_standard: dict[tuple[CoreNumericDTypes, CoreNumericDTypes], CoreNumericDTypes] = {
    (int8, uint64): float64,
    (int16, uint64): float64,
    (int32, uint64): float64,
    (int64, uint64): float64,
}

# Mixed integers and floating point numbers are not
# strictly defined, but generally we want to cast the
# integer to a floating point and then try again.
_int_to_floating: dict[CoreNumericDTypes, CoreNumericDTypes] = {
    int8: float16,
    uint8: float16,
    int16: float32,
    uint16: float32,
    int32: float64,
    uint32: float64,
    int64: float64,
    uint64: float64,
}


# Helper functions


def _as_nullable(dtype: CoreDTypes) -> NCoreDTypes:
    mapping: dict[CoreDTypes, NCoreDTypes] = {
        bool_: nbool,
        int8: nint8,
        int16: nint16,
        int32: nint32,
        int64: nint64,
        uint8: nuint8,
        uint16: nuint16,
        uint32: nuint32,
        uint64: nuint64,
        float16: nfloat16,
        float32: nfloat32,
        float64: nfloat64,
    }
    return mapping[dtype]


def _result_type_core_numeric(
    a: CoreNumericDTypes, b: CoreNumericDTypes
) -> CoreNumericDTypes:
    # Attempt promotion between known types. The implementation is not
    # using `isinstance` to avoid subclassing issues.
    if ret := _signed_signed.get((a, b)):
        return ret
    if ret := _unsigned_unsigned.get((a, b)):
        return ret
    if ret := _mixed_integers.get((a, b)):
        return ret
    if ret := _floating_floating.get((a, b)):
        return ret
    if ret := _non_standard.get((a, b)):
        return ret
    if a_floating := _int_to_floating.get(a):
        return _result_type_core_numeric(a_floating, b)
    if b_floating := _int_to_floating.get(b):
        return _result_type_core_numeric(a, b_floating)

    # TODO: Do bools and strings

    raise ValueError(f"No promotion between `{a}` and `{b}` is defined.")


@overload
def result_type(
    first: CoreNumericDTypes, *others: CoreNumericDTypes
) -> CoreNumericDTypes: ...


@overload
def result_type(first: CoreDTypes, *others: CoreDTypes) -> CoreDTypes: ...


@overload
def result_type(first: DType, *others: DType) -> DType: ...


def result_type(first: DType, *others: DType) -> DType:
    def result_binary(a: DType, b: DType) -> DType:
        res1 = a._result_type(b)
        if res1 != NotImplemented:
            return res1
        return b._rresult_type(a)

    res = reduce(result_binary, others, first)
    if res == NotImplemented:
        raise TypeError("No common type found")
    return res
