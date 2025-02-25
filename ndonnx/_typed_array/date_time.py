# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
"""Implementation of "custom" datetime-related data types."""

from __future__ import annotations

import operator
from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import numpy as np

from .._dtypes import TY_ARRAY_BASE, DType
from .._schema import DTypeInfoV1
from . import onnx
from .funcs import astyarray, where
from .typed_array import TyArrayBase
from .utils import safe_cast

if TYPE_CHECKING:
    from types import NotImplementedType

    from spox import Var
    from typing_extensions import Self

    from .._types import NestedSequence, OnnxShape, PyScalar
    from .indexing import (
        GetitemIndex,
        SetitemIndex,
    )


Unit = Literal["ns", "s"]

_NAT_SENTINEL = onnx.const(np.iinfo(np.int64).min).astype(onnx.int64)
BASE_DT_ARRAY = TypeVar("BASE_DT_ARRAY", bound="TimeBaseArray", covariant=True)


class BaseTimeDType(DType[BASE_DT_ARRAY]):
    unit: Unit

    def __init__(self, unit: Unit):
        from typing import get_args

        if unit not in get_args(Unit):
            raise TypeError(f"unsupported time unit `{unit}`")
        self.unit = unit

    @abstractmethod
    def _build(
        self, is_nat: onnx.TyArrayBool, data: onnx.TyArrayInt64, unit: Unit
    ) -> BASE_DT_ARRAY: ...

    def __ndx_cast_from__(self, arr: TyArrayBase) -> BASE_DT_ARRAY:
        if isinstance(arr, onnx.TyArrayInteger):
            data = arr.astype(onnx.int64)
            is_nat = safe_cast(onnx.TyArrayBool, data == _NAT_SENTINEL)
        elif isinstance(arr, onnx.TyArrayFloating):
            data = safe_cast(onnx.TyArrayInt64, arr.astype(onnx.int64))
            is_nat = safe_cast(onnx.TyArrayBool, data == _NAT_SENTINEL | arr.isnan())
        # elif isinstance(arr, py_scalars.TyArrayPyInt):
        #     data = safe_cast(onnx.TyArrayInt64, arr.astype(onnx.int64))
        #     is_nat = safe_cast(onnx.TyArrayBool, data == _NAT_SENTINEL)
        else:
            return NotImplemented
        return self._build(is_nat=is_nat, data=data, unit=self.unit)

    def __ndx_result_type__(self, other: DType | PyScalar) -> DType:
        if isinstance(other, int):
            return self
        return NotImplemented

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.unit}]"

    def __ndx_argument__(self, shape: OnnxShape) -> BASE_DT_ARRAY:
        data = onnx.int64.__ndx_argument__(shape)
        is_nat = onnx.bool_.__ndx_argument__(shape)
        return self._build(data=data, is_nat=is_nat, unit=self.unit)

    @property
    def __ndx_infov1__(self) -> DTypeInfoV1:
        return DTypeInfoV1(
            author="ndonnx", type_name=self.__class__.__name__, meta={"unit": self.unit}
        )

    def __ndx_arange__(
        self,
        start: int | float,
        stop: int | float,
        step: int | float = 1,
    ) -> BASE_DT_ARRAY:
        data = onnx.int64.__ndx_arange__(start, stop, step)
        is_nat = astyarray(False, dtype=onnx.bool_).broadcast_to(data.dynamic_shape)
        return self._build(data=data, is_nat=is_nat, unit=self.unit)

    def __ndx_eye__(
        self,
        n_rows: int,
        n_cols: int | None = None,
        /,
        *,
        k: int = 0,
    ) -> BASE_DT_ARRAY:
        data = onnx.int64.__ndx_eye__(n_rows, n_cols, k=k)
        is_nat = astyarray(False, dtype=onnx.bool_).broadcast_to(data.dynamic_shape)
        return self._build(data=data, is_nat=is_nat, unit=self.unit)

    def __ndx_ones__(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> BASE_DT_ARRAY:
        from .funcs import astyarray

        data = onnx.int64.__ndx_ones__(shape)
        is_nat = astyarray(False, dtype=onnx.bool_).broadcast_to(data.dynamic_shape)
        return self._build(data=data, is_nat=is_nat, unit=self.unit)

    def __ndx_zeros__(
        self, shape: tuple[int, ...] | onnx.TyArrayInt64
    ) -> BASE_DT_ARRAY:
        from .funcs import astyarray

        data = onnx.int64.__ndx_zeros__(shape)
        is_nat = astyarray(False, dtype=onnx.bool_).broadcast_to(data.dynamic_shape)
        return self._build(data=data, is_nat=is_nat, unit=self.unit)


class DateTime(BaseTimeDType["TyArrayDateTime"]):
    def _build(
        self, is_nat: onnx.TyArrayBool, data: onnx.TyArrayInt64, unit: Unit
    ) -> TyArrayDateTime:
        return TyArrayDateTime(is_nat=is_nat, data=data, unit=unit)

    def __ndx_create__(
        self, val: PyScalar | np.ndarray | TyArrayBase | Var | NestedSequence
    ) -> TyArrayDateTime:
        if isinstance(val, np.ndarray) and val.dtype.kind == "M":
            unit, count = np.datetime_data(val.dtype)
            unit = validate_unit(unit)
            if count != 1:
                raise ValueError(
                    "cannot create datetime with unit count other than '1'"
                )
            return astyarray(val.astype(np.int64)).astype(DateTime(unit=unit))
        elif isinstance(val, TyArrayDateTime):
            return val.copy()
        elif isinstance(val, int) or (
            isinstance(val, np.ndarray) and val.dtype.kind == "i"
        ):
            return onnx.int64.__ndx_create__(val).astype(self)
        else:
            raise ValueError(f"Unable to create an array with dtype {self} from {val}")


class TimeDelta(BaseTimeDType["TyArrayTimeDelta"]):
    def __ndx_create__(
        self, val: PyScalar | np.ndarray | TyArrayBase | Var | NestedSequence
    ) -> TyArrayTimeDelta:
        if isinstance(val, np.ndarray) and val.dtype.kind == "m":
            unit, count = np.datetime_data(val.dtype)
            unit = validate_unit(unit)
            if count != 1:
                raise ValueError(
                    "cannot create datetime with unit count other than '1'"
                )
            return astyarray(val.astype(np.int64)).astype(TimeDelta(unit=unit))
        elif isinstance(val, int) or (
            isinstance(val, np.ndarray) and val.dtype.kind == "i"
        ):
            return onnx.int64.__ndx_create__(val).astype(self)
        else:
            raise ValueError(f"Unable to create an array with dtype {self} from {val}")

    def _build(
        self, is_nat: onnx.TyArrayBool, data: onnx.TyArrayInt64, unit: Unit
    ) -> TyArrayTimeDelta:
        return TyArrayTimeDelta(is_nat=is_nat, data=data, unit=unit)


class TimeBaseArray(TyArrayBase):
    # TimeDelta and DateTime share the same memory layout. We can
    # therefore share some functionality.

    is_nat: onnx.TyArrayBool
    data: onnx.TyArrayInt64

    def __init__(self, is_nat: onnx.TyArrayBool, data: onnx.TyArrayInt64, unit: Unit):
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self) -> BaseTimeDType: ...

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
        if self.data.var._value is not None:
            return {"data": str(self.unwrap_numpy().astype(str).tolist())}
        return {"data": "*lazy*"}

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
        if self.dtype != value.dtype:
            raise TypeError(
                f"data type of 'value' must much array's, found `{value.dtype}`"
            )
        self.data[key] = value.data
        self.is_nat[key] = value.is_nat

    def put(
        self,
        key: onnx.TyArrayInt64,
        value: Self,
        /,
    ) -> None:
        if self.dtype != value.dtype:
            TypeError(f"data type of 'value' must much array's, found `{value.dtype}`")
        self.data.put(key, value.data)
        self.is_nat.put(key, value.is_nat)

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

    def isnan(self) -> onnx.TyArrayBool:
        return self.is_nat

    def __ndx_maximum__(self, rhs: TyArrayBase | PyScalar) -> TyArrayBase:
        from .funcs import maximum

        if not isinstance(rhs, type(self)):
            return NotImplemented
        if self.dtype.unit != rhs.dtype.unit:
            raise ValueError("inter-operation between time units is not implemented")

        data = safe_cast(onnx.TyArrayInt64, maximum(self.data, rhs.data))
        is_nat = safe_cast(onnx.TyArrayBool, self.is_nat | rhs.is_nat)
        return type(self)(is_nat=is_nat, data=data, unit=self.dtype.unit)

    def __ndx_minimum__(self, rhs: TyArrayBase | PyScalar) -> TyArrayBase:
        from .funcs import minimum

        if not isinstance(rhs, type(self)):
            return NotImplemented
        if self.dtype.unit != rhs.dtype.unit:
            raise ValueError("inter-operation between time units is not implemented")

        data = safe_cast(onnx.TyArrayInt64, minimum(self.data, rhs.data))
        is_nat = safe_cast(onnx.TyArrayBool, self.is_nat | rhs.is_nat)
        return type(self)(is_nat=is_nat, data=data, unit=self.dtype.unit)

    def __ndx_where__(
        self, cond: onnx.TyArrayBool, other: TyArrayBase, /
    ) -> TyArrayBase:
        from .funcs import where

        if self.dtype != other.dtype or not isinstance(other, type(self)):
            return NotImplemented

        data = safe_cast(onnx.TyArrayInt64, where(cond, self.data, other.data))
        is_nat = safe_cast(onnx.TyArrayBool, where(cond, self.is_nat, other.is_nat))

        return type(self)(is_nat=is_nat, data=data, unit=self.dtype.unit)

    def clip(
        self, /, min: TyArrayBase | None = None, max: TyArrayBase | None = None
    ) -> Self:
        if min is not None:
            if not isinstance(min, type(self)) or self.dtype != min.dtype:
                raise TypeError(
                    f"'min' be of identical data type as self; found {min.dtype}"
                )
        if max is not None:
            if not isinstance(max, type(self)) or self.dtype != max.dtype:
                raise TypeError(
                    f"'max' be of identical data type as self; found {max.dtype}"
                )

        data = self.data.clip(
            min=None if min is None else min.data,
            max=None if max is None else max.data,
        )
        is_nat = self.is_nat
        if min is not None:
            is_nat = safe_cast(onnx.TyArrayBool, is_nat | min.is_nat)

        if max is not None:
            is_nat = safe_cast(onnx.TyArrayBool, is_nat | max.is_nat)

        return type(self)(is_nat=is_nat, data=data, unit=self.dtype.unit)


class TyArrayTimeDelta(TimeBaseArray):
    _dtype: TimeDelta

    def __init__(self, is_nat: onnx.TyArrayBool, data: onnx.TyArrayInt64, unit: Unit):
        self.is_nat = safe_cast(onnx.TyArrayBool, is_nat)
        self.data = safe_cast(onnx.TyArrayInt64, data)
        self._dtype = TimeDelta(unit)

    @property
    def dtype(self) -> TimeDelta:
        return self._dtype

    def __ndx_cast_to__(
        self, dtype: DType[TY_ARRAY_BASE]
    ) -> TY_ARRAY_BASE | NotImplementedType:
        # TODO: Figure out why mypy/pyright are taking such an offense in this method
        if isinstance(dtype, onnx.IntegerDTypes):
            data = where(self.is_nat, _NAT_SENTINEL.astype(onnx.int64), self.data)
            return data.astype(dtype)  # type: ignore
        if isinstance(dtype, TimeDelta):
            powers = {
                "s": 0,
                "ms": 3,
                "us": 6,
                "ns": 9,
            }
            power = powers[dtype.unit] - powers[self.dtype.unit]  # type: ignore
            data = where(self.is_nat, astyarray(np.iinfo(np.int64).min), self.data)

            if power > 0:
                data = data * np.pow(10, power)
            if power < 0:
                data = data // np.pow(10, abs(power))

            data = safe_cast(onnx.TyArrayInt64, data)
            # TODO: Figure out why mypy does not like the blow
            return dtype._build(is_nat=self.is_nat, data=data, unit=dtype.unit)  # type: ignore

        return NotImplemented

    def unwrap_numpy(self) -> np.ndarray:
        is_nat = self.is_nat.unwrap_numpy()
        data = self.data.unwrap_numpy()

        out = data.astype(f"timedelta64[{self.dtype.unit}]")
        out[is_nat] = np.array("NaT", out.dtype)
        return out

    def __add__(self, rhs: TyArrayBase | PyScalar) -> TyArrayTimeDelta:
        return _apply_op(self, rhs, operator.add, True)

    def __radd__(self, lhs: TyArrayBase | PyScalar) -> TyArrayTimeDelta:
        return _apply_op(self, lhs, operator.add, False)

    def __mul__(self, rhs: TyArrayBase | PyScalar) -> TyArrayTimeDelta:
        return _apply_op(self, rhs, operator.mul, True)

    def __rmul__(self, lhs: TyArrayBase | PyScalar) -> TyArrayTimeDelta:
        return _apply_op(self, lhs, operator.mul, False)

    def __sub__(self, rhs: TyArrayBase | PyScalar) -> TyArrayTimeDelta:
        return _apply_op(self, rhs, operator.sub, True)

    def __rsub__(self, lhs: TyArrayBase | PyScalar) -> TyArrayTimeDelta:
        return _apply_op(self, lhs, operator.sub, False)

    def __truediv__(self, rhs: TyArrayBase | PyScalar) -> TyArrayBase:
        if isinstance(rhs, onnx.TyArrayNumber | float | int):
            data = (self.data / astyarray(rhs)).astype(onnx.int64)
            return TyArrayTimeDelta(is_nat=self.is_nat, data=data, unit=self.dtype.unit)
        if isinstance(rhs, TyArrayTimeDelta) and self.dtype == rhs.dtype:
            res = (self.data / rhs.data).astype(onnx.float64)
            res[safe_cast(onnx.TyArrayBool, self.is_nat | rhs.is_nat)] = astyarray(
                np.nan, dtype=onnx.float64
            )
            return res
        return NotImplemented

    def __rtruediv__(self, lhs: TyArrayBase | PyScalar) -> TyArrayBase:
        if isinstance(lhs, onnx.TyArrayNumber | float | int):
            data = (astyarray(lhs) / self.data).astype(onnx.int64)
            return TyArrayTimeDelta(is_nat=self.is_nat, data=data, unit=self.dtype.unit)
        return NotImplemented

    def __ndx_equal__(self, other) -> onnx.TyArrayBool:
        if not isinstance(other, TyArrayBase):
            return NotImplemented
        # TODO: Figure out what is missing to be able to narrow `other` via the data type
        if isinstance(other, TyArrayTimeDelta):
            if self.dtype.unit != other.dtype.unit:
                raise TypeError(
                    "comparison between different units is not implemented, yet"
                )

            res = self.data == other.data
            is_nat = self.is_nat | other.is_nat
            return safe_cast(onnx.TyArrayBool, res & ~is_nat)
        return NotImplemented


class TyArrayDateTime(TimeBaseArray):
    _dtype: DateTime

    def __init__(self, is_nat: onnx.TyArrayBool, data: onnx.TyArrayInt64, unit: Unit):
        self.is_nat = safe_cast(onnx.TyArrayBool, is_nat)
        self.data = safe_cast(onnx.TyArrayInt64, data)
        self._dtype = DateTime(unit)

    @property
    def dtype(self) -> DateTime:
        return self._dtype

    def unwrap_numpy(self) -> np.ndarray:
        is_nat = self.is_nat.unwrap_numpy()
        data = self.data.unwrap_numpy()

        out = data.astype(f"datetime64[{self.dtype.unit}]")
        out[is_nat] = np.array("NaT", "datetime64")
        return out

    def __ndx_cast_to__(
        self, dtype: DType[TY_ARRAY_BASE]
    ) -> TY_ARRAY_BASE | NotImplementedType:
        # TODO: Figure out why mypy/pyright don't like this method
        if isinstance(dtype, onnx.IntegerDTypes):
            data = where(self.is_nat, astyarray(np.iinfo(np.int64).min), self.data)
            return data.astype(dtype)  # type: ignore
        if isinstance(dtype, DateTime):
            powers = {
                "s": 0,
                "ms": 3,
                "us": 6,
                "ns": 9,
            }
            power = powers[dtype.unit] - powers[self.dtype.unit]  # type: ignore
            data = where(self.is_nat, astyarray(np.iinfo(np.int64).min), self.data)

            if power > 0:
                data = data * np.pow(10, power)
            if power < 0:
                data = data / np.pow(10, abs(power))

            return dtype._build(is_nat=self.is_nat, data=data, unit=dtype.unit)  # type: ignore
        return NotImplemented

    def _coerce_to_time_delta(
        self, other: TyArrayBase | PyScalar
    ) -> TyArrayTimeDelta | NotImplementedType:
        """Coerce to a time delta array of identical unit as ``self``.

        This is useful for comparisons and ``__add__``.
        """
        from .funcs import astyarray

        if isinstance(other, int):
            other = astyarray(other, dtype=onnx.int64)
        if isinstance(other, onnx.TyArrayInt64):
            other = other.astype(TimeDelta(unit=self.dtype.unit))
        if not isinstance(other, TyArrayTimeDelta):
            return NotImplemented

        if self.dtype.unit != other.dtype.unit:
            raise TypeError("inter operation between time units is not implemented")
        return other

    def _coerce_to_date_time(
        self, other: TyArrayBase | PyScalar
    ) -> TyArrayDateTime | NotImplementedType:
        """Coerce `other` to ``TyArrayDateTime``.

        This is encapsulates the promotion rules for comparison operations.
        """
        if isinstance(other, int):
            other = astyarray(other, dtype=onnx.int64)
        if isinstance(other, onnx.TyArrayInt64):
            other = other.astype(DateTime(unit=self.dtype.unit))
        if not isinstance(other, TyArrayDateTime):
            return NotImplemented

        if self.dtype.unit != other.dtype.unit:
            raise TypeError("inter operation between time units is not implemented")
        return other

    def __add__(self, rhs: TyArrayBase | PyScalar) -> Self:
        rhs = self._coerce_to_time_delta(rhs)
        if rhs is NotImplemented:
            return NotImplemented

        data = safe_cast(onnx.TyArrayInt64, self.data + rhs.data)
        is_nat = safe_cast(onnx.TyArrayBool, self.is_nat | rhs.is_nat)
        return type(self)(is_nat=is_nat, data=data, unit=self.dtype.unit)

    def __radd__(
        self, lhs: TyArrayBase | PyScalar
    ) -> TyArrayDateTime | TyArrayTimeDelta:
        return self.__add__(lhs)

    def _sub(self, other, forward: bool):
        op = operator.sub
        if isinstance(other, int):
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

    def __sub__(
        self, rhs: TyArrayBase | PyScalar
    ) -> TyArrayDateTime | TyArrayTimeDelta:
        return self._sub(rhs, True)

    def __rsub__(
        self, lhs: TyArrayBase | PyScalar
    ) -> TyArrayDateTime | TyArrayTimeDelta:
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

    def __le__(self, other: TyArrayBase | PyScalar) -> onnx.TyArrayBool:
        other = self._coerce_to_date_time(other)
        return self._apply_comp(operator.le, other)

    def __lt__(self, other: TyArrayBase | PyScalar) -> onnx.TyArrayBool:
        other = self._coerce_to_date_time(other)
        return self._apply_comp(operator.lt, other)

    def __ge__(self, other: TyArrayBase | PyScalar) -> onnx.TyArrayBool:
        other = self._coerce_to_date_time(other)
        return self._apply_comp(operator.ge, other)

    def __gt__(self, other: TyArrayBase | PyScalar) -> onnx.TyArrayBool:
        other = self._coerce_to_date_time(other)
        return self._apply_comp(operator.gt, other)

    def __ndx_equal__(self, other) -> onnx.TyArrayBool:
        if not isinstance(other, TyArrayBase):
            return NotImplemented

        if not isinstance(other, TyArrayDateTime):
            return NotImplemented
        if self.dtype.unit != other.dtype.unit:
            raise TypeError(
                "comparison between different units is not implemented, yet"
            )

        res = self.data == other.data
        is_nat = self.is_nat | other.is_nat

        return safe_cast(onnx.TyArrayBool, res & ~is_nat)


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
    is_nat = this.is_nat
    if isinstance(other, TyArrayTimeDelta):
        is_nat = safe_cast(onnx.TyArrayBool, is_nat | other.is_nat)
    return type(this)(is_nat=is_nat, data=data, unit=this.dtype.unit)


def _coerce_other(
    this: TyArrayTimeDelta, other
) -> onnx.TyArrayInt64 | NotImplementedType:
    from .funcs import astyarray

    if isinstance(other, int):
        return astyarray(other, dtype=onnx.int64)
    elif isinstance(other, TyArrayTimeDelta):
        if this.dtype.unit != other.dtype.unit:
            raise ValueError("inter-operation between time units is not implemented")
        return other.data
    return NotImplemented


def validate_unit(unit: str) -> Literal["ns", "s"]:
    if unit in ["ns", "s"]:
        return unit  # type: ignore
    raise ValueError(f"unsupported datetime unit `{unit}`")
