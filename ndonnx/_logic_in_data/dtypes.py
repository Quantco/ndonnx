# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from types import NotImplementedType
from typing import TYPE_CHECKING, overload

import numpy as np
from typing_extensions import Self

if TYPE_CHECKING:
    from . import _typed_array


class DType(ABC):
    @abstractmethod
    def _result_type(self, other: DType) -> DType | NotImplementedType: ...

    def _rresult_type(self, other: DType) -> DType | NotImplementedType:
        # Should allow for at least Python scalars in the signature
        return NotImplemented

    @property
    @abstractmethod
    def _tyarr_class(self) -> type[_typed_array._TypedArray[Self]]: ...


class _CoreDType(DType): ...


class _NCoreDType(DType):
    _unmasked_dtype: _CoreDType


class _Number(_CoreDType):
    def _result_type(self, rhs: DType) -> DType | NotImplementedType:
        if isinstance(self, CoreNumericDTypes) and isinstance(rhs, CoreNumericDTypes):
            return _result_type_core_numeric(self, rhs)

        return NotImplemented


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

        return as_nullable(core_result)

    def _rresult_type(self, lhs: DType) -> DType | NotImplementedType:
        # e.g. called after `CoreDType._result_type(self)`

        if isinstance(lhs, _CoreDType | _NCoreDType):
            # All core types are commutative
            return self._result_type(lhs)

        return NotImplemented


class Bool(_CoreDType):
    def _result_type(self, rhs: DType) -> DType | NotImplementedType:
        return NotImplemented

    @property
    def _tyarr_class(self) -> type[_typed_array.BoolData]:
        from ._typed_array import BoolData

        return BoolData


class Int8(_Number):
    @property
    def _tyarr_class(self) -> type[_typed_array.Int8Data]:
        from ._typed_array import Int8Data

        return Int8Data


class Int16(_Number):
    @property
    def _tyarr_class(self) -> type[_typed_array.Int16Data]:
        from ._typed_array import Int16Data

        return Int16Data


class Int32(_Number):
    @property
    def _tyarr_class(self) -> type[_typed_array.Int32Data]:
        from ._typed_array import Int32Data

        return Int32Data


class Int64(_Number):
    @property
    def _tyarr_class(self) -> type[_typed_array.Int64Data]:
        from ._typed_array import Int64Data

        return Int64Data


class Uint8(_Number):
    @property
    def _tyarr_class(self) -> type[_typed_array.Uint8Data]:
        from ._typed_array import Uint8Data

        return Uint8Data


class Uint16(_Number):
    @property
    def _tyarr_class(self) -> type[_typed_array.Uint16Data]:
        from ._typed_array import Uint16Data

        return Uint16Data


class Uint32(_Number):
    @property
    def _tyarr_class(self) -> type[_typed_array.Uint32Data]:
        from ._typed_array import Uint32Data

        return Uint32Data


class Uint64(_Number):
    @property
    def _tyarr_class(self) -> type[_typed_array.Uint64Data]:
        from ._typed_array import Uint64Data

        return Uint64Data


class Float16(_Number):
    @property
    def _tyarr_class(self) -> type[_typed_array.Float16Data]:
        from ._typed_array import Float16Data

        return Float16Data


class Float32(_Number):
    @property
    def _tyarr_class(self) -> type[_typed_array.Float32Data]:
        from ._typed_array import Float32Data

        return Float32Data


class Float64(_Number):
    @property
    def _tyarr_class(self) -> type[_typed_array.Float64Data]:
        from ._typed_array import Float64Data

        return Float64Data


class _PyInt(DType):
    def _result_type(self, other: DType) -> DType | NotImplementedType:
        if isinstance(other, CoreNumericDTypes):
            return other
        raise ValueError

    @property
    def _tyarr_class(self) -> type[_typed_array._ArrayPyInt]:
        from ._typed_array import _ArrayPyInt

        return _ArrayPyInt


class _PyFloat(DType):
    def _result_type(self, other: DType) -> DType | NotImplementedType:
        if isinstance(other, CoreIntegerDTypes):
            return float64
        if isinstance(other, NCoreIntegerDTypes):
            return nfloat64
        if isinstance(other, CoreFloatingDTypes | NCoreFloatingDTypes):
            return other
        raise ValueError

    @property
    def _tyarr_class(self) -> type[_typed_array._ArrayPyFloat]:
        from ._typed_array import _ArrayPyFloat

        return _ArrayPyFloat


# Non-nullable Singleton instances
bool_ = Bool()

float16 = Float16()
float32 = Float32()
float64 = Float64()

int16 = Int16()
int32 = Int32()
int64 = Int64()
int8 = Int8()

uint8 = Uint8()
uint16 = Uint16()
uint32 = Uint32()
uint64 = Uint64()

# scalar singleton instances
_pyint = _PyInt()
_pyfloat = _PyFloat()


class NBool(_NCoreDType):
    _unmasked_dtype = bool_

    def _result_type(self, rhs: DType) -> DType | NotImplementedType:
        return NotImplemented

    @property
    def _tyarr_class(self) -> type[_typed_array.NBoolData]:
        from ._typed_array import NBoolData

        return NBoolData


class NInt8(_NNumber):
    _unmasked_dtype = int8

    @property
    def _tyarr_class(self) -> type[_typed_array.NInt8Data]:
        from ._typed_array import NInt8Data

        return NInt8Data


class NInt16(_NNumber):
    _unmasked_dtype = int16

    @property
    def _tyarr_class(self) -> type[_typed_array.NInt16Data]:
        from ._typed_array import NInt16Data

        return NInt16Data


class NInt32(_NNumber):
    _unmasked_dtype = int32

    @property
    def _tyarr_class(self) -> type[_typed_array.NInt32Data]:
        from ._typed_array import NInt32Data

        return NInt32Data


class NInt64(_NNumber):
    _unmasked_dtype = int64

    @property
    def _tyarr_class(self) -> type[_typed_array.NInt64Data]:
        from ._typed_array import NInt64Data

        return NInt64Data


class NUint8(_NNumber):
    _unmasked_dtype = uint8

    @property
    def _tyarr_class(self) -> type[_typed_array.NUint8Data]:
        from ._typed_array import NUint8Data

        return NUint8Data


class NUint16(_NNumber):
    _unmasked_dtype = uint16

    @property
    def _tyarr_class(self) -> type[_typed_array.NUint16Data]:
        from ._typed_array import NUint16Data

        return NUint16Data


class NUint32(_NNumber):
    _unmasked_dtype = uint32

    @property
    def _tyarr_class(self) -> type[_typed_array.NUint32Data]:
        from ._typed_array import NUint32Data

        return NUint32Data


class NUint64(_NNumber):
    _unmasked_dtype = uint64

    @property
    def _tyarr_class(self) -> type[_typed_array.NUint64Data]:
        from ._typed_array import NUint64Data

        return NUint64Data


class NFloat16(_NNumber):
    _unmasked_dtype = float16

    @property
    def _tyarr_class(self) -> type[_typed_array.NFloat16Data]:
        from ._typed_array import NFloat16Data

        return NFloat16Data


class NFloat32(_NNumber):
    _unmasked_dtype = float32

    @property
    def _tyarr_class(self) -> type[_typed_array.NFloat32Data]:
        from ._typed_array import NFloat32Data

        return NFloat32Data


class NFloat64(_NNumber):
    _unmasked_dtype = float64

    @property
    def _tyarr_class(self) -> type[_typed_array.NFloat64Data]:
        from ._typed_array import NFloat64Data

        return NFloat64Data


# Non-nullable Singleton instances
nbool = NBool()

nfloat16 = NFloat16()
nfloat32 = NFloat32()
nfloat64 = NFloat64()

nint8 = NInt8()
nint16 = NInt16()
nint32 = NInt32()
nint64 = NInt64()

nuint8 = NUint8()
nuint16 = NUint16()
nuint32 = NUint32()
nuint64 = NUint64()

# Union types
#
# Union types are exhaustive and don't create ambiguities with respect to user-defined subtypes.

CoreFloatingDTypes = Float16 | Float32 | Float64

CoreIntegerDTypes = Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64

CoreNumericDTypes = CoreFloatingDTypes | CoreIntegerDTypes

CoreDTypes = Bool | CoreNumericDTypes

NCoreIntegerDTypes = (
    NInt8 | NInt16 | NInt32 | NInt64 | NUint8 | NUint16 | NUint32 | NUint64
)
NCoreFloatingDTypes = NFloat16 | NFloat32 | NFloat64

NCoreNumericDTypes = NCoreFloatingDTypes | NCoreIntegerDTypes

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

_core_to_nullable_core: dict[CoreDTypes, NCoreDTypes] = {
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


def as_nullable(dtype: CoreDTypes) -> NCoreDTypes:
    return _core_to_nullable_core[dtype]


def as_non_nullable(dtype: NCoreDTypes) -> CoreDTypes:
    return {v: k for k, v in _core_to_nullable_core.items()}[dtype]


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


def from_numpy(np_dtype: np.dtype) -> CoreDTypes:
    if np_dtype == np.int8:
        return int8
    if np_dtype == np.int16:
        return int16
    if np_dtype == np.int32:
        return int32
    if np_dtype == np.int64:
        return int64

    if np_dtype == np.uint8:
        return uint8
    if np_dtype == np.uint16:
        return uint16
    if np_dtype == np.uint32:
        return uint32
    if np_dtype == np.uint64:
        return uint64

    if np_dtype == np.float16:
        return float16
    if np_dtype == np.float32:
        return float32
    if np_dtype == np.float64:
        return float64

    if np_dtype == np.bool:
        return bool_

    raise ValueError(f"'{np_dtype}' does not have a corresponding ndonnx data type")


def as_numpy(dtype: CoreDTypes) -> np.dtype:
    if dtype == int8:
        return np.dtype("int8")
    if dtype == int16:
        return np.dtype("int16")
    if dtype == int32:
        return np.dtype("int32")
    if dtype == int64:
        return np.dtype("int64")

    if dtype == uint8:
        return np.dtype("uint8")
    if dtype == uint16:
        return np.dtype("uint16")
    if dtype == uint32:
        return np.dtype("uint32")
    if dtype == uint64:
        return np.dtype("uint64")

    if dtype == float16:
        return np.dtype("float16")
    if dtype == float32:
        return np.dtype("float32")
    if dtype == float64:
        return np.dtype("float64")

    if dtype == bool_:
        return np.dtype("bool")

    # Should never happen
    raise ValueError(f"'{dtype}' does not have a corresponding NumPy data type")


default_int = int64
default_float = float64
