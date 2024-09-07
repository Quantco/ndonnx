# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
"""Implementation of "custom" datetime-related data types."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import numpy as np

from ..dtypes import CoreIntegerDTypes, DType, _PyInt, bool_, int64
from ..schema import DTypeInfo, flatten_components
from .core import TyArrayBool, TyArrayInt64, TyArrayInteger
from .funcs import astypedarray, typed_where
from .py_scalars import _ArrayPyInt
from .typed_array import TyArrayBase
from .utils import safe_cast

if TYPE_CHECKING:
    from types import NotImplementedType

    from typing_extensions import Self

    from ..array import Index, OnnxShape
    from ..schema import Components, Schema, StructComponent


Unit = Literal["ns", "s"]

_NAT_SENTINEL = _ArrayPyInt(np.iinfo(np.int64).min)


class DateTime(DType):
    def __init__(self, unit: Unit):
        self.unit = unit

    def _result_type(self, other: DType) -> DType | NotImplementedType:
        if isinstance(other, _PyInt):
            return self
        return NotImplemented

    @property
    def _tyarr_class(self) -> type[TyArrayDateTime]:
        return TyArrayDateTime

    def _tyarray_from_tyarray(self, arr: TyArrayBase) -> TyArrayDateTime:
        if isinstance(arr, TyArrayInteger):
            data = safe_cast(TyArrayInt64, arr.astype(int64))
            is_nat = safe_cast(TyArrayBool, data == _NAT_SENTINEL)
        elif isinstance(arr, _ArrayPyInt):
            data = safe_cast(TyArrayInt64, arr.astype(int64))
            is_nat = safe_cast(TyArrayBool, data == _NAT_SENTINEL)
        else:
            raise NotImplementedError
        return TyArrayDateTime(is_nat=is_nat, data=data, unit=self.unit)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.unit}]"

    @property
    def _info(self):
        return DTypeInfo(
            defining_library="ndonnx",
            version=1,
            dtype=self.__class__.__name__,
            dtype_state={"unit": self.unit},
        )


class TimeDelta(DType):
    def __init__(self, unit: Unit):
        self.unit = unit

    def _result_type(self, other: DType) -> DType | NotImplementedType:
        # TODO
        raise NotImplementedError

    @property
    def _tyarr_class(self) -> type[TyArrayTimeDelta]:
        return TyArrayTimeDelta

    def _tyarray_from_tyarray(self, arr: TyArrayBase) -> Self:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.unit}]"

    @property
    def _info(self):
        return DTypeInfo(
            defining_library="ndonnx",
            version=1,
            dtype=self.__class__.__name__,
            dtype_state={"unit": self.unit},
        )


TIME_DTYPE = TypeVar("TIME_DTYPE", bound=DateTime | TimeDelta)


class TimeBaseArray(TyArrayBase[TIME_DTYPE]):
    # TimeDelta and DateTime share the same memory layout. We can
    # therefore share some functionality.

    is_nat: TyArrayBool
    data: TyArrayInt64

    def __init__(self, is_nat: TyArrayBool, data: TyArrayInt64, unit: Unit):
        raise NotImplementedError

    def disassemble(self) -> tuple[Components, Schema]:
        dtype_info = self.dtype._info
        component_schema: StructComponent = {
            "data": self.data.disassemble()[1],
            "is_nat": self.is_nat.disassemble()[1],
        }
        schema = Schema(dtype_info=dtype_info, components=component_schema)
        components = flatten_components(
            {
                "data": self.data.disassemble()[0],
                "is_nat": self.is_nat.disassemble()[0],
            }
        )
        return components, schema

    def __getitem__(self, index: Index) -> Self:
        is_nat = self.is_nat[index]
        data = self.data[index]

        return type(self)(is_nat=is_nat, data=data, unit=self.dtype.unit)

    @property
    def shape(self) -> OnnxShape:
        return self.data.shape

    def reshape(self, shape: tuple[int, ...]) -> Self:
        is_nat = self.is_nat.reshape(shape)
        data = self.data.reshape(shape)

        return type(self)(is_nat=is_nat, data=data, unit=self.dtype.unit)

    def _eqcomp(self, other) -> TyArrayBase:
        raise NotImplementedError


class TyArrayTimeDelta(TimeBaseArray[TimeDelta]):
    def __init__(self, is_nat: TyArrayBool, data: TyArrayInt64, unit: Unit):
        self.is_nat = is_nat
        self.data = data
        self.dtype = TimeDelta(unit)

    @classmethod
    def as_argument(cls, shape: OnnxShape, dtype: DType) -> Self:
        if not isinstance(dtype, TimeDelta):
            raise ValueError("unexpected 'dtype' `{dtype}`")

        is_nat = TyArrayBool.as_argument(shape, bool_)
        data = TyArrayInt64.as_argument(shape, int64)
        unit = dtype.unit
        return cls(is_nat=is_nat, data=data, unit=unit)

    def _astype(self, dtype: DType) -> TyArrayBase | NotImplementedType:
        if isinstance(dtype, CoreIntegerDTypes):
            data = typed_where(self.is_nat, _NAT_SENTINEL, self.data)
            return data.astype(dtype)
        if isinstance(dtype, TimeDelta):
            powers = {
                "s": 0,
                "ms": 3,
                "us": 6,
                "ns": 9,
            }
            power = powers[dtype.unit] - powers[self.dtype.unit]
            data = typed_where(
                self.is_nat, astypedarray(np.iinfo(np.int64).min), self.data
            )

            if power > 0:
                data = data * np.pow(10, power)
            if power < 0:
                data = data / np.pow(10, abs(power))

            return data
        return NotImplemented

    def __add__(self, rhs: TyArrayBase) -> TyArrayTimeDelta | TyArrayTimeDelta:
        return _apply_op(self, rhs, operator.add, True)

    def __radd__(self, lhs: TyArrayBase) -> TyArrayTimeDelta | TyArrayTimeDelta:
        return _apply_op(self, lhs, operator.add, False)

    def __mul__(self, rhs: TyArrayBase) -> TyArrayTimeDelta | TyArrayTimeDelta:
        return _apply_op(self, rhs, operator.mul, True)

    def __rmul__(self, lhs: TyArrayBase) -> TyArrayTimeDelta | TyArrayTimeDelta:
        return _apply_op(self, lhs, operator.mul, False)

    def __sub__(self, rhs: TyArrayBase) -> TyArrayTimeDelta | TyArrayTimeDelta:
        return _apply_op(self, rhs, operator.sub, True)

    def __rsub__(self, lhs: TyArrayBase) -> TyArrayTimeDelta | TyArrayTimeDelta:
        return _apply_op(self, lhs, operator.sub, False)


class TyArrayDateTime(TimeBaseArray[DateTime]):
    def __init__(self, is_nat: TyArrayBool, data: TyArrayInt64, unit: Unit):
        self.is_nat = is_nat
        self.data = data
        self.dtype = DateTime(unit)

    @classmethod
    def as_argument(cls, shape: OnnxShape, dtype: DType) -> Self:
        if not isinstance(dtype, DateTime):
            raise ValueError("unexpected 'dtype' `{dtype}`")

        is_nat = TyArrayBool.as_argument(shape, bool_)
        data = TyArrayInt64.as_argument(shape, int64)
        unit = dtype.unit
        return cls(is_nat=is_nat, data=data, unit=unit)

    def to_numpy(self) -> np.ndarray:
        is_nat = self.is_nat.to_numpy()
        data = self.data.to_numpy()

        out = data.astype(f"datetime64[{self.dtype.unit}]")
        out[is_nat] = np.array("NaT", "datetime64")
        return out

    def _astype(self, dtype: DType) -> TyArrayBase | NotImplementedType:
        if isinstance(dtype, CoreIntegerDTypes):
            data = typed_where(
                self.is_nat, astypedarray(np.iinfo(np.int64).min), self.data
            )
            return data.astype(dtype)
        if isinstance(dtype, DateTime):
            powers = {
                "s": 0,
                "ms": 3,
                "us": 6,
                "ns": 9,
            }
            power = powers[dtype.unit] - powers[self.dtype.unit]
            data = typed_where(
                self.is_nat, astypedarray(np.iinfo(np.int64).min), self.data
            )

            if power > 0:
                data = data * np.pow(10, power)
            if power < 0:
                data = data / np.pow(10, abs(power))

            return data
        return NotImplemented

    def __add__(self, rhs: TyArrayBase) -> TyArrayDateTime | TyArrayTimeDelta:
        if isinstance(rhs, _ArrayPyInt | TyArrayInt64):
            rhs_data = rhs
        elif isinstance(rhs, TyArrayTimeDelta):
            if self.dtype.unit != rhs.dtype.unit:
                raise ValueError(
                    "inter-operation between time units is not implemented"
                )
            rhs_data = rhs.data
        else:
            raise NotImplementedError

        data = cast(TyArrayInt64, (self.data + rhs_data))
        return type(self)(is_nat=self.is_nat, data=data, unit=self.dtype.unit)

    # TODO: Fix too strict type hints on all dunder methods
    def __radd__(self, lhs: TyArrayBase) -> TyArrayDateTime | TyArrayTimeDelta:
        return self.__add__(lhs)

    def _sub(self, other, forward: bool):
        op = operator.sub
        if isinstance(other, _ArrayPyInt):
            data_ = op(self.data, other) if forward else op(other, self.data)
            data = safe_cast(TyArrayInt64, data_)
            return type(self)(is_nat=self.is_nat, data=data, unit=self.dtype.unit)

        if isinstance(other, TyArrayDateTime):
            if self.dtype.unit != other.dtype.unit:
                raise NotImplementedError
            data_ = op(self.data, other.data) if forward else op(other.data, self.data)
            data = safe_cast(TyArrayInt64, data_)
            is_nat = safe_cast(TyArrayBool, self.is_nat | other.is_nat)
            return TyArrayTimeDelta(is_nat=is_nat, data=data, unit=self.dtype.unit)
        raise NotImplementedError

    def __sub__(self, rhs: TyArrayBase) -> TyArrayDateTime | TyArrayTimeDelta:
        return self._sub(rhs, True)

    def __rsub__(self, lhs: TyArrayBase) -> TyArrayDateTime | TyArrayTimeDelta:
        return self._sub(lhs, False)


def _apply_op(
    this: TyArrayTimeDelta,
    other,
    op: Callable[[Any, Any], Any],
    forward: bool,
):
    if isinstance(other, _ArrayPyInt | TyArrayInt64):
        other_data = other
    elif isinstance(other, TyArrayTimeDelta):
        if this.dtype.unit != other.dtype.unit:
            raise ValueError("inter-operation between time units is not implemented")
        other_data = other.data
    else:
        raise NotImplementedError

    data_ = op(this.data, other_data)
    data = cast(TyArrayInt64, data_)
    return type(this)(is_nat=this.is_nat, data=data, unit=this.dtype.unit)
