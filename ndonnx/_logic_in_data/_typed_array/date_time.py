# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
"""Implementation of "custom" datetime-related data types."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import numpy as np

from .._dtypes import TY_ARRAY, DType
from .._schema import DTypeInfoV1
from . import onnx, py_scalars
from .funcs import astyarray, where
from .typed_array import TyArrayBase
from .utils import safe_cast

if TYPE_CHECKING:
    from types import NotImplementedType

    from spox import Var
    from typing_extensions import Self

    from .._array import OnnxShape
    from .indexing import (
        GetitemIndex,
        SetitemIndex,
    )


Unit = Literal["ns", "s"]

_NAT_SENTINEL = py_scalars.TyArrayPyInt(np.iinfo(np.int64).min)


class BaseTimeDType(DType):
    def __init__(self, unit: Unit):
        self.unit = unit

    def _result_type(self, other: DType) -> DType | NotImplementedType:
        if isinstance(other, py_scalars.PyInteger):
            return self
        return NotImplemented

    def __ndx_convert_tyarray__(self, arr: TyArrayBase) -> TimeBaseArray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.unit}]"

    def _argument(self, shape: OnnxShape) -> TimeBaseArray:
        data = onnx.int64._argument(shape)
        is_nat = onnx.bool_._argument(shape)
        return self._tyarr_class(data=data, is_nat=is_nat, unit=self.unit)

    @property
    def _infov1(self) -> DTypeInfoV1:
        return DTypeInfoV1(
            author="ndonnx", type_name=self.__class__.__name__, meta={"unit": self.unit}
        )


class DateTime(BaseTimeDType):
    @property
    def _tyarr_class(self) -> type[TyArrayDateTime]:
        return TyArrayDateTime

    def __ndx_convert_tyarray__(self, arr: TyArrayBase) -> TyArrayDateTime:
        if isinstance(arr, onnx.TyArrayInteger):
            data = safe_cast(onnx.TyArrayInt64, arr.astype(onnx.int64))
            is_nat = safe_cast(onnx.TyArrayBool, data == _NAT_SENTINEL)
        elif isinstance(arr, py_scalars.TyArrayPyInt):
            data = safe_cast(onnx.TyArrayInt64, arr.astype(onnx.int64))
            is_nat = safe_cast(onnx.TyArrayBool, data == _NAT_SENTINEL)
        else:
            raise NotImplementedError
        return TyArrayDateTime(is_nat=is_nat, data=data, unit=self.unit)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.unit}]"


class TimeDelta(BaseTimeDType):
    @property
    def _tyarr_class(self) -> type[TyArrayTimeDelta]:
        return TyArrayTimeDelta


TIME_DTYPE = TypeVar("TIME_DTYPE", bound=DateTime | TimeDelta)


class TimeBaseArray(TyArrayBase):
    # TimeDelta and DateTime share the same memory layout. We can
    # therefore share some functionality.

    dtype: DateTime | TimeDelta
    is_nat: onnx.TyArrayBool
    data: onnx.TyArrayInt64

    def __init__(self, is_nat: onnx.TyArrayBool, data: onnx.TyArrayInt64, unit: Unit):
        raise NotImplementedError

    def disassemble(self) -> dict[str, Var]:
        # TODO: What was the old schema here?
        return {
            "data": self.data.disassemble(),
            "is_nat": self.is_nat.disassemble(),
        }

    def __ndx_value_repr__(self):
        return {
            "data": self.data.__ndx_value_repr__()["data"],
            "is_nat": self.is_nat.__ndx_value_repr__()["data"],
        }

    def __getitem__(self, index: GetitemIndex) -> Self:
        is_nat = self.is_nat[index]
        data = self.data[index]

        return type(self)(is_nat=is_nat, data=data, unit=self.dtype.unit)

    def __setitem__(
        self,
        key: SetitemIndex,
        value: Self,
        /,
    ) -> None:
        raise NotImplementedError

    @property
    def dynamic_shape(self) -> onnx.TyArrayInt64:
        return self.data.dynamic_shape

    @property
    def mT(self) -> Self:  # noqa: N802
        data = self.data.mT
        is_nat = self.is_nat.mT

        return type(self)(data=data, is_nat=is_nat, unit=self.dtype.unit)

    @property
    def shape(self) -> OnnxShape:
        return self.data.shape

    def broadcast_to(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> Self:
        data = self.data.broadcast_to(shape)
        is_nat = self.is_nat.broadcast_to(shape)
        return type(self)(data=data, is_nat=is_nat, unit=self.dtype.unit)

    def concat(self, others: list[Self], axis: None | int) -> Self:
        data = self.data.concat([arr.data for arr in others], axis=axis)
        is_nat = self.is_nat.concat([arr.is_nat for arr in others], axis=axis)

        return type(self)(data=data, is_nat=is_nat, unit=self.dtype.unit)

    def reshape(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> Self:
        is_nat = self.is_nat.reshape(shape)
        data = self.data.reshape(shape)

        return type(self)(is_nat=is_nat, data=data, unit=self.dtype.unit)

    def _eqcomp(self, other) -> TyArrayBase:
        raise NotImplementedError


class TyArrayTimeDelta(TimeBaseArray):
    dtype: TimeDelta

    def __init__(self, is_nat: onnx.TyArrayBool, data: onnx.TyArrayInt64, unit: Unit):
        self.is_nat = is_nat
        self.data = data
        self.dtype = TimeDelta(unit)

    def __ndx_astype__(self, dtype: DType[TY_ARRAY]) -> TY_ARRAY | NotImplementedType:
        res_type = dtype._tyarr_class
        if isinstance(dtype, onnx.IntegerDTypes):
            data = where(self.is_nat, _NAT_SENTINEL, self.data)
            return data.astype(dtype)
        if isinstance(dtype, TimeDelta):
            powers = {
                "s": 0,
                "ms": 3,
                "us": 6,
                "ns": 9,
            }
            power = powers[dtype.unit] - powers[self.dtype.unit]
            data = where(self.is_nat, astyarray(np.iinfo(np.int64).min), self.data)

            if power > 0:
                data = data * np.pow(10, power)
            if power < 0:
                data = data / np.pow(10, abs(power))

            return safe_cast(res_type, data)
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


class TyArrayDateTime(TimeBaseArray):
    dtype: DateTime

    def __init__(self, is_nat: onnx.TyArrayBool, data: onnx.TyArrayInt64, unit: Unit):
        self.is_nat = is_nat
        self.data = data
        self.dtype = DateTime(unit)

    def unwrap_numpy(self) -> np.ndarray:
        is_nat = self.is_nat.unwrap_numpy()
        data = self.data.unwrap_numpy()

        out = data.astype(f"datetime64[{self.dtype.unit}]")
        out[is_nat] = np.array("NaT", "datetime64")
        return out

    def __ndx_astype__(self, dtype: DType[TY_ARRAY]) -> TY_ARRAY | NotImplementedType:
        res_type = dtype._tyarr_class
        if isinstance(dtype, onnx.IntegerDTypes):
            data = where(self.is_nat, astyarray(np.iinfo(np.int64).min), self.data)
            return data.astype(dtype)
        if isinstance(dtype, DateTime):
            powers = {
                "s": 0,
                "ms": 3,
                "us": 6,
                "ns": 9,
            }
            power = powers[dtype.unit] - powers[self.dtype.unit]
            data = where(self.is_nat, astyarray(np.iinfo(np.int64).min), self.data)

            if power > 0:
                data = data * np.pow(10, power)
            if power < 0:
                data = data / np.pow(10, abs(power))

            return safe_cast(res_type, data)
        return NotImplemented

    def __ndx_maximum__(self, rhs: TyArrayBase) -> TyArrayBase:
        if not isinstance(rhs, TyArrayDateTime):
            return NotImplemented
        if self.dtype.unit != rhs.dtype.unit:
            raise ValueError("inter-operation between time units is not implemented")
        data = safe_cast(onnx.TyArrayInt64, self.data.__ndx_maximum__(rhs.data))
        is_nat = safe_cast(onnx.TyArrayBool, self.is_nat | rhs.is_nat)
        return type(self)(is_nat=is_nat, data=data, unit=self.dtype.unit)

    def __add__(self, rhs: TyArrayBase) -> TyArrayDateTime | TyArrayTimeDelta:
        if isinstance(rhs, py_scalars.TyArrayPyInt | onnx.TyArrayInt64):
            rhs_data = rhs
        elif isinstance(rhs, TyArrayTimeDelta):
            if self.dtype.unit != rhs.dtype.unit:
                raise ValueError(
                    "inter-operation between time units is not implemented"
                )
            rhs_data = rhs.data
        else:
            raise NotImplementedError

        data = cast(onnx.TyArrayInt64, (self.data + rhs_data))
        return type(self)(is_nat=self.is_nat, data=data, unit=self.dtype.unit)

    def __radd__(self, lhs: TyArrayBase) -> TyArrayDateTime | TyArrayTimeDelta:
        return self.__add__(lhs)

    def _sub(self, other, forward: bool):
        op = operator.sub
        if isinstance(other, py_scalars.TyArrayPyInt):
            data_ = op(self.data, other) if forward else op(other, self.data)
            data = safe_cast(onnx.TyArrayInt64, data_)
            return type(self)(is_nat=self.is_nat, data=data, unit=self.dtype.unit)

        if isinstance(other, TyArrayDateTime):
            if self.dtype.unit != other.dtype.unit:
                raise NotImplementedError
            data_ = op(self.data, other.data) if forward else op(other.data, self.data)
            data = safe_cast(onnx.TyArrayInt64, data_)
            is_nat = safe_cast(onnx.TyArrayBool, self.is_nat | other.is_nat)
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
    if isinstance(other, py_scalars.TyArrayPyInt | onnx.TyArrayInt64):
        other_data = other
    elif isinstance(other, TyArrayTimeDelta):
        if this.dtype.unit != other.dtype.unit:
            raise ValueError("inter-operation between time units is not implemented")
        other_data = other.data
    else:
        raise NotImplementedError

    data_ = op(this.data, other_data)
    data = cast(onnx.TyArrayInt64, data_)
    return type(this)(is_nat=this.is_nat, data=data, unit=this.dtype.unit)
