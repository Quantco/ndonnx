# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
"""Implementation of "custom" datetime-related data types."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import numpy as np

from .._dtypes import TY_ARRAY_BASE, DType
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
BASE_DT_ARRAY = TypeVar("BASE_DT_ARRAY", bound="TimeBaseArray")


class BaseTimeDType(DType[BASE_DT_ARRAY]):
    unit: Unit

    def __init__(self, unit: Unit):
        self.unit = unit

    def __ndx_cast_from__(self, arr: TyArrayBase) -> BASE_DT_ARRAY:
        if isinstance(arr, onnx.TyArrayInteger):
            data = safe_cast(onnx.TyArrayInt64, arr.astype(onnx.int64))
            is_nat = safe_cast(onnx.TyArrayBool, data == _NAT_SENTINEL)
        elif isinstance(arr, onnx.TyArrayFloating):
            data = safe_cast(onnx.TyArrayInt64, arr.astype(onnx.int64))
            is_nat = safe_cast(onnx.TyArrayBool, data == _NAT_SENTINEL | arr.isnan())
        elif isinstance(arr, py_scalars.TyArrayPyInt):
            data = safe_cast(onnx.TyArrayInt64, arr.astype(onnx.int64))
            is_nat = safe_cast(onnx.TyArrayBool, data == _NAT_SENTINEL)
        else:
            return NotImplemented
        return self._tyarr_class(is_nat=is_nat, data=data, unit=self.unit)

    def __ndx_result_type__(self, other: DType) -> DType:
        if isinstance(other, py_scalars.PyInteger):
            return self
        return NotImplemented

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.unit}]"

    def _argument(self, shape: OnnxShape) -> BASE_DT_ARRAY:
        data = onnx.int64._argument(shape)
        is_nat = onnx.bool_._argument(shape)
        return self._tyarr_class(data=data, is_nat=is_nat, unit=self.unit)

    @property
    def _infov1(self) -> DTypeInfoV1:
        return DTypeInfoV1(
            author="ndonnx", type_name=self.__class__.__name__, meta={"unit": self.unit}
        )

    def _arange(
        self,
        start: int | float,
        stop: int | float,
        step: int | float = 1,
    ) -> BASE_DT_ARRAY:
        from .funcs import astyarray

        data = onnx.int64._arange(start, stop, step)
        is_nat = astyarray(False, dtype=onnx.bool_).broadcast_to(data.dynamic_shape)
        return self._tyarr_class(data=data, is_nat=is_nat, unit=self.unit)

    def _eye(
        self,
        n_rows: int,
        n_cols: int | None = None,
        /,
        *,
        k: int = 0,
    ) -> BASE_DT_ARRAY:
        from .funcs import astyarray

        data = onnx.int64._eye(n_rows, n_cols, k=k)
        is_nat = astyarray(False, dtype=onnx.bool_).broadcast_to(data.dynamic_shape)
        return self._tyarr_class(data=data, is_nat=is_nat, unit=self.unit)

    def _ones(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> BASE_DT_ARRAY:
        from .funcs import astyarray

        data = onnx.int64._ones(shape)
        is_nat = astyarray(False, dtype=onnx.bool_).broadcast_to(data.dynamic_shape)
        return self._tyarr_class(data=data, is_nat=is_nat, unit=self.unit)

    def _zeros(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> BASE_DT_ARRAY:
        from .funcs import astyarray

        data = onnx.int64._zeros(shape)
        is_nat = astyarray(False, dtype=onnx.bool_).broadcast_to(data.dynamic_shape)
        return self._tyarr_class(data=data, is_nat=is_nat, unit=self.unit)


class DateTime(BaseTimeDType):
    @property
    def _tyarr_class(self) -> type[TyArrayDateTime]:
        return TyArrayDateTime

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

    def copy(self) -> Self:
        # We want to copy the component arrays, too.
        return type(self)(
            data=self.data.copy(), is_nat=self.is_nat.copy(), unit=self.dtype.unit
        )

    def disassemble(self) -> dict[str, Var]:
        return {
            "data": self.data.disassemble(),
            "is_nat": self.is_nat.disassemble(),
        }

    def __ndx_value_repr__(self):
        return {"data": str(self.unwrap_numpy().astype(str).tolist())}

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

    def squeeze(self, /, axis: int | tuple[int, ...]) -> Self:
        is_nat = self.is_nat.squeeze(axis)
        data = self.data.squeeze(axis)

        return type(self)(is_nat=is_nat, data=data, unit=self.dtype.unit)

    def permute_dims(self, axes: tuple[int, ...]) -> Self:
        is_nat = self.is_nat.permute_dims(axes)
        data = self.data.permute_dims(axes)

        return type(self)(is_nat=is_nat, data=data, unit=self.dtype.unit)

    def __ndx_maximum__(self, rhs: TyArrayBase) -> TyArrayBase:
        from .funcs import maximum

        if not isinstance(rhs, type(self)):
            return NotImplemented
        if self.dtype.unit != rhs.dtype.unit:
            raise ValueError("inter-operation between time units is not implemented")

        data = safe_cast(onnx.TyArrayInt64, maximum(self.data, rhs.data))
        is_nat = safe_cast(onnx.TyArrayBool, self.is_nat | rhs.is_nat)
        return type(self)(is_nat=is_nat, data=data, unit=self.dtype.unit)

    def __ndx_minimum__(self, rhs: TyArrayBase) -> TyArrayBase:
        from .funcs import minimum

        if not isinstance(rhs, type(self)):
            return NotImplemented
        if self.dtype.unit != rhs.dtype.unit:
            raise ValueError("inter-operation between time units is not implemented")

        data = safe_cast(onnx.TyArrayInt64, minimum(self.data, rhs.data))
        is_nat = safe_cast(onnx.TyArrayBool, self.is_nat | rhs.is_nat)
        return type(self)(is_nat=is_nat, data=data, unit=self.dtype.unit)


class TyArrayTimeDelta(TimeBaseArray):
    dtype: TimeDelta

    def __init__(self, is_nat: onnx.TyArrayBool, data: onnx.TyArrayInt64, unit: Unit):
        self.is_nat = safe_cast(onnx.TyArrayBool, is_nat)
        self.data = safe_cast(onnx.TyArrayInt64, data)
        self.dtype = TimeDelta(unit)

    def __ndx_cast_to__(
        self, dtype: DType[TY_ARRAY_BASE]
    ) -> TY_ARRAY_BASE | NotImplementedType:
        res_type = dtype._tyarr_class
        if isinstance(dtype, onnx.IntegerDTypes):
            data = where(self.is_nat, _NAT_SENTINEL.astype(onnx.int64), self.data)
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

    def unwrap_numpy(self) -> np.ndarray:
        is_nat = self.is_nat.unwrap_numpy()
        data = self.data.unwrap_numpy()

        out = data.astype(f"timedelta64[{self.dtype.unit}]")
        out[is_nat] = np.array("NaT", "timedelta64")
        return out

    def __add__(self, rhs: TyArrayBase) -> TyArrayTimeDelta:
        return _apply_op(self, rhs, operator.add, True)

    def __radd__(self, lhs: TyArrayBase) -> TyArrayTimeDelta:
        return _apply_op(self, lhs, operator.add, False)

    def __mul__(self, rhs: TyArrayBase) -> TyArrayTimeDelta:
        return _apply_op(self, rhs, operator.mul, True)

    def __rmul__(self, lhs: TyArrayBase) -> TyArrayTimeDelta:
        return _apply_op(self, lhs, operator.mul, False)

    def __sub__(self, rhs: TyArrayBase) -> TyArrayTimeDelta:
        return _apply_op(self, rhs, operator.sub, True)

    def __rsub__(self, lhs: TyArrayBase) -> TyArrayTimeDelta:
        return _apply_op(self, lhs, operator.sub, False)

    def __truediv__(self, rhs: TyArrayBase) -> TyArrayTimeDelta:
        return _apply_op(self, rhs, operator.truediv, True)

    def __rtruediv__(self, lhs: TyArrayBase) -> TyArrayTimeDelta:
        return _apply_op(self, lhs, operator.truediv, False)

    def _eqcomp(self, other) -> onnx.TyArrayBool:
        if not isinstance(other.dtype, TimeDelta):
            return NotImplemented
        if self.dtype.unit != other.dtype.unit:
            raise TypeError(
                "comparison between different units is not implemented, yet"
            )

        res = self.data == other.data
        is_nat = self.is_nat | other.is_nat

        return res & ~is_nat


class TyArrayDateTime(TimeBaseArray):
    dtype: DateTime

    def __init__(self, is_nat: onnx.TyArrayBool, data: onnx.TyArrayInt64, unit: Unit):
        self.is_nat = safe_cast(onnx.TyArrayBool, is_nat)
        self.data = safe_cast(onnx.TyArrayInt64, data)
        self.dtype = DateTime(unit)

    def unwrap_numpy(self) -> np.ndarray:
        is_nat = self.is_nat.unwrap_numpy()
        data = self.data.unwrap_numpy()

        out = data.astype(f"datetime64[{self.dtype.unit}]")
        out[is_nat] = np.array("NaT", "datetime64")
        return out

    def __ndx_cast_to__(
        self, dtype: DType[TY_ARRAY_BASE]
    ) -> TY_ARRAY_BASE | NotImplementedType:
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

    def _coerce_to_time_delta(
        self, other: TyArrayBase
    ) -> TyArrayTimeDelta | NotImplementedType:
        """Coerce to a time delta array of identical unit as ``self``.

        This is useful for comparisons and ``__add__``.
        """
        if isinstance(other, py_scalars.TyArrayPyInt):
            other = other.astype(onnx.int64)
        if isinstance(other, onnx.TyArrayInt64):
            other = other.astype(TimeDelta(unit=self.dtype.unit))
        if not isinstance(other, TyArrayTimeDelta):
            return NotImplemented

        if self.dtype.unit != other.dtype.unit:
            raise TypeError("inter operation between time units is not implemented")
        return other

    def _coerce_to_date_time(
        self, other: TyArrayBase
    ) -> TyArrayDateTime | NotImplementedType:
        """Coerce `other` to ``TyArrayDateTime``.

        This is encapsulates the promotion rules for comparison operations.
        """
        if isinstance(other, py_scalars.TyArrayPyInt):
            other = other.astype(onnx.int64)
        if isinstance(other, onnx.TyArrayInt64):
            other = other.astype(DateTime(unit=self.dtype.unit))
        if not isinstance(other, TyArrayDateTime):
            return NotImplemented

        if self.dtype.unit != other.dtype.unit:
            raise TypeError("inter operation between time units is not implemented")
        return other

    def __add__(self, rhs: TyArrayBase) -> Self:
        rhs = self._coerce_to_time_delta(rhs)
        if rhs is NotImplemented:
            return NotImplemented

        data = safe_cast(onnx.TyArrayInt64, self.data + rhs.data)
        is_nat = safe_cast(onnx.TyArrayBool, self.is_nat | rhs.is_nat)
        return type(self)(is_nat=is_nat, data=data, unit=self.dtype.unit)

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

        if isinstance(other, TyArrayTimeDelta) and forward:
            if self.dtype.unit != other.dtype.unit:
                raise NotImplementedError
            data_ = op(self.data, other.data) if forward else op(other.data, self.data)
            data = safe_cast(onnx.TyArrayInt64, data_)
            is_nat = safe_cast(onnx.TyArrayBool, self.is_nat | other.is_nat)
            return TyArrayDateTime(is_nat=is_nat, data=data, unit=self.dtype.unit)

        return NotImplemented

    def __sub__(self, rhs: TyArrayBase) -> TyArrayDateTime | TyArrayTimeDelta:
        return self._sub(rhs, True)

    def __rsub__(self, lhs: TyArrayBase) -> TyArrayDateTime | TyArrayTimeDelta:
        return self._sub(lhs, False)

    def _apply_comp(
        self,
        op: Callable[[onnx.TyArrayInt64, onnx.TyArrayInt64], onnx.TyArrayBool],
        other: TyArrayBase,
    ) -> onnx.TyArrayBool:
        other = self._coerce_to_date_time(other)
        data = op(self.data, other.data)
        is_nat = self.is_nat | other.is_nat
        return safe_cast(onnx.TyArrayBool, data & ~is_nat)

    def __le__(self, other: TyArrayBase) -> onnx.TyArrayBool:
        return self._apply_comp(operator.le, other)

    def __lt__(self, other: TyArrayBase) -> onnx.TyArrayBool:
        return self._apply_comp(operator.lt, other)

    def __ge__(self, other: TyArrayBase) -> onnx.TyArrayBool:
        return self._apply_comp(operator.ge, other)

    def __gt__(self, other: TyArrayBase) -> onnx.TyArrayBool:
        return self._apply_comp(operator.gt, other)

    def _eqcomp(self, other) -> onnx.TyArrayBool:
        if not isinstance(other.dtype, DateTime):
            return NotImplemented
        if self.dtype.unit != other.dtype.unit:
            raise TypeError(
                "comparison between different units is not implemented, yet"
            )

        res = self.data == other.data
        is_nat = self.is_nat | other.is_nat

        return res & ~is_nat


def _apply_op(
    this: TyArrayTimeDelta,
    other,
    op: Callable[[Any, Any], Any],
    forward: bool,
) -> TyArrayTimeDelta:
    other_data = _coerce_other(this, other)
    if other_data is NotImplemented:
        return NotImplemented

    if forward:
        data_ = op(this.data, other_data)
    else:
        data_ = op(other_data, this.data)
    data = cast(onnx.TyArrayInt64, data_)
    return type(this)(is_nat=this.is_nat, data=data, unit=this.dtype.unit)


def _coerce_other(
    this: TyArrayTimeDelta, other
) -> onnx.TyArrayInt64 | NotImplementedType:
    if isinstance(other, py_scalars.TyArrayPyInt | onnx.TyArrayInt64):
        return other.astype(onnx.int64)
    elif isinstance(other, TyArrayTimeDelta):
        if this.dtype.unit != other.dtype.unit:
            raise ValueError("inter-operation between time units is not implemented")
        return other.data
    return NotImplemented


def validate_unit(unit: str) -> Literal["ns", "s"]:
    if unit in ["ns", "s"]:
        return unit  # type: ignore
    raise ValueError(f"unsupported datetime unit `{unit}`")
