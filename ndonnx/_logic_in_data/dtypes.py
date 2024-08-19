# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from types import NotImplementedType


def result_type(*core_dtypes: CoreDType) -> CoreDType:
    return reduce(_result_type_binary, core_dtypes)


def result_type_nullable(
    *mixed_core_dtypes: CoreDType | NCoreDType,
) -> CoreDType | NCoreDType:
    core_dtypes = []
    contains_nullable = False
    for dtype in mixed_core_dtypes:
        if isinstance(dtype, CoreDType):
            core_dtypes.append(dtype)
        else:
            core_dtypes.append(dtype._core_type)
            contains_nullable = True
    core_result = reduce(_result_type_binary, core_dtypes)

    if contains_nullable:
        return as_nullable(core_result)
    return core_result


def as_nullable(dtype: CoreDType) -> NCoreDType:
    mapping: dict[CoreDType, NCoreDType] = {
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


class DType(ABC):
    @abstractmethod
    def _result_type(self, other: DType) -> DType | NotImplementedType: ...

    def _rresult_type(self, other: DType) -> DType | NotImplementedType:
        return NotImplemented


class CoreDType(DType):
    def _result_type(self, rhs: DType) -> DType | NotImplementedType:
        if isinstance(rhs, CoreDType):
            return _result_type_binary(self, rhs)

        # No implicit promotion for bools and strings
        return NotImplemented


class NCoreDType(DType):
    _core_type: CoreDType

    def _result_type(self, rhs: DType) -> DType | NotImplementedType:
        if isinstance(rhs, CoreDType):
            core_result = _result_type_binary(self._core_type, rhs)
        elif isinstance(rhs, NCoreDType):
            core_result = _result_type_binary(self._core_type, rhs._core_type)

        else:
            # No implicit promotion for bools and strings
            return NotImplemented

        return as_nullable(core_result)

    def _rresult_type(self, lhs: DType) -> DType | NotImplementedType:
        # e.g. called after `CoreDType._result_type(self)`

        if isinstance(lhs, CoreDType | NCoreDType):
            # All core types are cummutative
            return self._result_type(lhs)

        return NotImplemented


class Bool(CoreDType): ...


class NBool(NCoreDType): ...


class Number(CoreDType): ...


class Int8(Number): ...


class Int16(Number): ...


class Int32(Number): ...


class Int64(Number): ...


class Uint8(Number): ...


class Uint16(Number): ...


class Uint32(Number): ...


class Uint64(Number): ...


class Float16(Number): ...


class Float32(Number): ...


class Float64(Number): ...


class NNumber(NCoreDType): ...


class NInt8(NNumber): ...


class NInt16(NNumber): ...


class NInt32(NNumber): ...


class NInt64(NNumber): ...


class NUint8(NNumber): ...


class NUint16(NNumber): ...


class NUint32(NNumber): ...


class NUint64(NNumber): ...


class NFloat16(NNumber): ...


class NFloat32(NNumber): ...


class NFloat64(NNumber): ...


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


def _result_type_binary(a: CoreDType, b: CoreDType) -> CoreDType:
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
        return _result_type_binary(a_floating, b)
    if b_floating := _int_to_floating.get(b):
        return _result_type_binary(a, b_floating)

    # TODO: Do bools and strings

    raise ValueError(f"No promotion between `{a}` and `{b}` is defined.")


# Promotion tables taken from:
# https://data-apis.org/array-api/draft/API_specification/type_promotion.html#type-promotion
# and
# https://numpy.org/neps/nep-0050-scalar-promotion.html#motivation-and-scope
_signed_signed: dict[tuple[CoreDType, CoreDType], CoreDType] = {
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
_unsigned_unsigned: dict[tuple[CoreDType, CoreDType], CoreDType] = {
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
_mixed_integers: dict[tuple[CoreDType, CoreDType], CoreDType] = {
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

_floating_floating: dict[tuple[CoreDType, CoreDType], CoreDType] = {
    (float32, float32): float32,
    (float32, float64): float64,
    (float64, float32): float64,
    (float64, float64): float64,
}

# Non-standard interactions
_non_standard: dict[tuple[CoreDType, CoreDType], CoreDType] = {
    (int8, uint64): float64,
    (int16, uint64): float64,
    (int32, uint64): float64,
    (int64, uint64): float64,
}

# Mixed integers and floating point numbers are not
# strictly defined, but generally we want to cast the
# integer to a floating point and then try again.
_int_to_floating: dict[CoreDType, CoreDType] = {
    int8: float16,
    uint8: float16,
    int16: float32,
    uint16: float32,
    int32: float64,
    uint32: float64,
    int64: float64,
    uint64: float64,
}
