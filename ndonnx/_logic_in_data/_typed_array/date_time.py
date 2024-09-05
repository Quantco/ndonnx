# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
"""Implementation of "custom" datetime-related data types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar, cast

import numpy as np

from ..dtypes import CoreIntegerDTypes, DType, bool_, int64
from .core import TyArrayBool, TyArrayInt64, TyArrayInteger
from .funcs import astypedarray, typed_where
from .py_scalars import _ArrayPyInt
from .typed_array import TyArrayBase
from .utils import safe_cast

if TYPE_CHECKING:
    from types import NotImplementedType

    from typing_extensions import Self

    from ..array import Index, OnnxShape


Unit = Literal["ns", "s"]

_NAT_SENTINEL = _ArrayPyInt(np.iinfo(np.int64).min)


class DateTime(DType):
    def __init__(self, unit: Unit):
        self.unit = unit

    def _result_type(self, other: DType) -> DType | NotImplementedType:
        # TODO
        raise NotImplementedError

    @property
    def _tyarr_class(self) -> type[TyArrayDateTime]:
        return TyArrayDateTime

    def _tyarray_from_tyarray(self, arr: TyArrayBase) -> TyArrayDateTime:
        if isinstance(arr, TyArrayInteger):
            data = safe_cast(TyArrayInt64, arr.astype(int64))
            is_nat = safe_cast(TyArrayBool, data == _NAT_SENTINEL)
            return TyArrayDateTime(is_nat=is_nat, data=data, unit=self.unit)

        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.unit}]"


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


TIME_DTYPE = TypeVar("TIME_DTYPE", bound=DateTime | TimeDelta)


class TimeBaseArray(TyArrayBase[TIME_DTYPE]):
    # TimeDelta and DateTime share the same memory layout. We can
    # therefore share some functionality.

    is_nat: TyArrayBool
    data: TyArrayInt64

    def __init__(self, is_nat: TyArrayBool, data: TyArrayInt64, unit: Unit):
        raise NotImplementedError

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

    def __mul__(self, rhs: TyArrayBase) -> TyArrayTimeDelta | TyArrayTimeDelta:
        if isinstance(rhs, _ArrayPyInt):
            data = cast(TyArrayInt64, (self.data * rhs))
            return type(self)(is_nat=self.is_nat, data=data, unit=self.dtype.unit)
        raise NotImplementedError


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
        rhs_data: _ArrayPyInt | TyArrayInt64
        if isinstance(rhs, _ArrayPyInt):
            rhs_data = rhs
        elif isinstance(rhs, TyArrayTimeDelta):
            rhs_data = rhs.data
        else:
            raise NotImplementedError

        data = cast(TyArrayInt64, (self.data + rhs_data))
        return type(self)(is_nat=self.is_nat, data=data, unit=self.dtype.unit)

    def __sub__(self, rhs: TyArrayBase) -> TyArrayDateTime | TyArrayTimeDelta:
        if isinstance(rhs, _ArrayPyInt):
            data = safe_cast(TyArrayInt64, self.data - rhs)
            return type(self)(is_nat=self.is_nat, data=data, unit=self.dtype.unit)
        if isinstance(rhs, TyArrayDateTime):
            if self.dtype.unit != rhs.dtype.unit:
                raise NotImplementedError
            data = safe_cast(TyArrayInt64, self.data - rhs.data)
            is_nat = safe_cast(TyArrayBool, self.is_nat | rhs.is_nat)
            return TyArrayTimeDelta(is_nat=is_nat, data=data, unit=self.dtype.unit)
        raise NotImplementedError
