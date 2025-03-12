# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
"""Implementation of "custom" datetime-related data types."""

from __future__ import annotations

import operator
from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeVar, get_args

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
TIMEARRAY_co = TypeVar("TIMEARRAY_co", bound="TimeBaseArray", covariant=True)


class BaseTimeDType(DType[TIMEARRAY_co]):
    unit: Unit

    def __init__(self, unit: Unit):
        if unit not in get_args(Unit):
            raise TypeError(f"unsupported time unit `{unit}`")
        self.unit = unit

    @abstractmethod
    def _build(self, data: onnx.TyArrayInt64) -> TIMEARRAY_co: ...

    def __ndx_cast_from__(self, arr: TyArrayBase) -> TIMEARRAY_co:
        if isinstance(arr, onnx.TyArrayInteger):
            data = arr.astype(onnx.int64)
        elif isinstance(arr, onnx.TyArrayFloating):
            data = where(arr.isnan(), _NAT_SENTINEL, arr).astype(onnx.int64)
        else:
            return NotImplemented
        return self._build(data=data)

    def __ndx_result_type__(self, other: DType | PyScalar) -> DType:
        if isinstance(other, int):
            return self
        return NotImplemented

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.unit}]"

    def __ndx_argument__(self, shape: OnnxShape) -> TIMEARRAY_co:
        data = onnx.int64.__ndx_argument__(shape)
        return self._build(data=data)

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
    ) -> TIMEARRAY_co:
        data = onnx.int64.__ndx_arange__(start, stop, step)
        return self._build(data=data)

    def __ndx_eye__(
        self,
        n_rows: int,
        n_cols: int | None = None,
        /,
        *,
        k: int = 0,
    ) -> TIMEARRAY_co:
        data = onnx.int64.__ndx_eye__(n_rows, n_cols, k=k)
        return self._build(data=data)

    def __ndx_ones__(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> TIMEARRAY_co:
        data = onnx.int64.__ndx_ones__(shape)
        return self._build(data=data)

    def __ndx_zeros__(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> TIMEARRAY_co:
        data = onnx.int64.__ndx_zeros__(shape)
        return self._build(data=data)


class DateTime(BaseTimeDType["TyArrayDateTime"]):
    def _build(self, data: onnx.TyArrayInt64) -> TyArrayDateTime:
        return TyArrayDateTime(data=data, unit=self.unit)

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

    def _build(self, data: onnx.TyArrayInt64) -> TyArrayTimeDelta:
        return TyArrayTimeDelta(data=data, unit=self.unit)


class TimeBaseArray(TyArrayBase):
    # TimeDelta and DateTime share the same memory layout. We can
    # therefore share some functionality.

    _data: onnx.TyArrayInt64

    def __init__(self, data: onnx.TyArrayInt64, unit: Unit):
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self) -> BaseTimeDType: ...

    @property
    def is_nat(self) -> onnx.TyArrayBool:
        return self._data == _NAT_SENTINEL

    def copy(self) -> Self:
        # We want to copy the component arrays, too.
        # TODO: This should be more efficient...
        return type(self)(data=self._data.copy(), unit=self.dtype.unit)

    def disassemble(self) -> Var:
        return self._data.disassemble()

    def __ndx_value_repr__(self):
        if self._data._var._value is not None:
            return {"data": str(self.unwrap_numpy().astype(str).tolist())}
        return {"data": "*lazy*"}

    def __getitem__(self, index: GetitemIndex) -> Self:
        data = self._data[index]
        return self.dtype._build(data)

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
        self._data[key] = value._data

    def put(
        self,
        key: onnx.TyArrayInt64,
        value: Self,
        /,
    ) -> None:
        if self.dtype != value.dtype:
            TypeError(f"data type of 'value' must much array's, found `{value.dtype}`")
        self._data.put(key, value._data)

    @property
    def dynamic_shape(self) -> onnx.TyArrayInt64:
        return self._data.dynamic_shape

    @property
    def mT(self) -> Self:  # noqa: N802
        data = self._data.mT

        return type(self)(data=data, unit=self.dtype.unit)

    @property
    def shape(self) -> OnnxShape:
        return self._data.shape

    def broadcast_to(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> Self:
        data = self._data.broadcast_to(shape)
        return self.dtype._build(data)

    def concat(self, others: list[Self], axis: None | int) -> Self:
        data = self._data.concat([arr._data for arr in others], axis=axis)
        return self.dtype._build(data)

    def reshape(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> Self:
        data = self._data.reshape(shape)
        return self.dtype._build(data)

    def squeeze(self, /, axis: int | tuple[int, ...]) -> Self:
        data = self._data.squeeze(axis)
        return self.dtype._build(data)

    def permute_dims(self, axes: tuple[int, ...]) -> Self:
        data = self._data.permute_dims(axes)
        return self.dtype._build(data)

    def isnan(self) -> onnx.TyArrayBool:
        return self.is_nat

    def _apply_comp(
        self,
        op: Callable[[onnx.TyArrayInt64, onnx.TyArrayInt64], onnx.TyArrayBool],
        other: TyArrayBase | PyScalar,
    ) -> onnx.TyArrayBool | NotImplementedType:
        if isinstance(other, TyArrayTimeDelta | TyArrayDateTime):
            other_arr = other
        elif isinstance(other, int):
            other_arr = astyarray(other, self.dtype)
        else:
            return NotImplemented

        if type(self) is not type(other):
            return NotImplemented

        self, other = _coerce_units(self, other_arr)

        data = op(self._data, other._data)
        is_nat = self.is_nat | other.is_nat
        return data & ~is_nat

    def __le__(self, other: TyArrayBase | PyScalar) -> onnx.TyArrayBool:
        return self._apply_comp(operator.le, other)

    def __lt__(self, other: TyArrayBase | PyScalar) -> onnx.TyArrayBool:
        return self._apply_comp(operator.lt, other)

    def __ge__(self, other: TyArrayBase | PyScalar) -> onnx.TyArrayBool:
        return self._apply_comp(operator.ge, other)

    def __gt__(self, other: TyArrayBase | PyScalar) -> onnx.TyArrayBool:
        return self._apply_comp(operator.gt, other)

    def __ndx_maximum__(self, rhs: TyArrayBase | PyScalar) -> TyArrayBase:
        from .funcs import maximum

        if not isinstance(rhs, type(self)):
            return NotImplemented
        if self.dtype.unit != rhs.dtype.unit:
            raise ValueError("inter-operation between time units is not implemented")

        is_nat = self.is_nat | rhs.is_nat
        return self.dtype._build(
            where(is_nat, _NAT_SENTINEL, maximum(self._data, rhs._data))
        )

    def __ndx_minimum__(self, rhs: TyArrayBase | PyScalar) -> TyArrayBase:
        from .funcs import minimum

        if not isinstance(rhs, type(self)):
            return NotImplemented
        if self.dtype.unit != rhs.dtype.unit:
            raise ValueError("inter-operation between time units is not implemented")

        is_nat = self.is_nat | rhs.is_nat
        return self.dtype._build(
            where(is_nat, _NAT_SENTINEL, minimum(self._data, rhs._data))
        )

    def __ndx_where__(
        self, cond: onnx.TyArrayBool, other: TyArrayBase, /
    ) -> TyArrayBase:
        if self.dtype != other.dtype or not isinstance(other, type(self)):
            return NotImplemented

        return self.dtype._build(where(cond, self._data, other._data))

    def clip(
        self, /, min: TyArrayBase | None = None, max: TyArrayBase | None = None
    ) -> Self:
        if min is not None:
            if not isinstance(min, type(self)) or self.dtype != min.dtype:
                raise TypeError(
                    f"'min' must be of identical data type as self; found {min.dtype}"
                )
        if max is not None:
            if not isinstance(max, type(self)) or self.dtype != max.dtype:
                raise TypeError(
                    f"'max' must be of identical data type as self; found {max.dtype}"
                )

        is_nat = self.is_nat
        if min is not None:
            is_nat = is_nat | min.is_nat

        if max is not None:
            is_nat = is_nat | max.is_nat
        data = self._data.clip(
            min=None if min is None else min._data,
            max=None if max is None else max._data,
        )
        return self.dtype._build(where(is_nat, _NAT_SENTINEL, data))


class TyArrayTimeDelta(TimeBaseArray):
    _dtype: TimeDelta

    def __init__(self, data: onnx.TyArrayInt64, unit: Unit):
        self._data = safe_cast(onnx.TyArrayInt64, data)
        self._dtype = TimeDelta(unit)

    @property
    def dtype(self) -> TimeDelta:
        return self._dtype

    def __ndx_cast_to__(
        self, dtype: DType[TY_ARRAY_BASE]
    ) -> TY_ARRAY_BASE | NotImplementedType:
        if isinstance(dtype, onnx.Int64):
            # Disallow casting to smaller integer types since that
            # would cause an overflow for NaT values
            return self._data.astype(dtype)
        if isinstance(dtype, TimeDelta):
            # TODO: Figure out why pyright does not like the below
            return _convert_unit(self, dtype)  # type: ignore

        return NotImplemented

    def unwrap_numpy(self) -> np.ndarray:
        is_nat = self.is_nat.unwrap_numpy()
        data = self._data.unwrap_numpy()

        out = data.astype(f"timedelta64[{self.dtype.unit}]")
        out[is_nat] = np.array("NaT", out.dtype)
        return out

    def __add__(self, rhs: TyArrayBase | PyScalar) -> TyArrayTimeDelta:
        if isinstance(rhs, TyArrayTimeDelta | int):
            return _apply_op(self, rhs, operator.add, True)
        return NotImplemented

    def __radd__(self, lhs: TyArrayBase | PyScalar) -> TyArrayTimeDelta:
        if isinstance(lhs, TyArrayTimeDelta | int):
            return _apply_op(self, lhs, operator.add, False)
        return NotImplemented

    def __mul__(self, rhs: TyArrayBase | PyScalar) -> TyArrayTimeDelta:
        if isinstance(rhs, TyArrayTimeDelta | int):
            return _apply_op(self, rhs, operator.mul, True)
        return NotImplemented

    def __rmul__(self, lhs: TyArrayBase | PyScalar) -> TyArrayTimeDelta:
        if isinstance(lhs, TyArrayTimeDelta | int):
            return _apply_op(self, lhs, operator.mul, False)
        return NotImplemented

    def __sub__(self, rhs: TyArrayBase | PyScalar) -> TyArrayTimeDelta:
        if isinstance(rhs, TyArrayTimeDelta | int):
            return _apply_op(self, rhs, operator.sub, True)
        return NotImplemented

    def __rsub__(self, lhs: TyArrayBase | PyScalar) -> TyArrayTimeDelta:
        if isinstance(lhs, TyArrayTimeDelta | int):
            return _apply_op(self, lhs, operator.sub, False)
        return NotImplemented

    def __truediv__(self, rhs: TyArrayBase | PyScalar) -> TyArrayBase:
        if isinstance(rhs, onnx.TyArrayNumber | float | int):
            data = (self._data / astyarray(rhs)).astype(onnx.int64)
            return self.dtype._build(where(self.is_nat, _NAT_SENTINEL, data))

        if isinstance(rhs, TyArrayTimeDelta) and self.dtype == rhs.dtype:
            is_nat = self.is_nat | rhs.is_nat
            res = self._data.astype(onnx.float64) / rhs._data.astype(onnx.float64)
            return where(is_nat, astyarray(np.nan, dtype=onnx.float64), res)

        return NotImplemented

    def __rtruediv__(self, lhs: TyArrayBase | PyScalar) -> TyArrayBase:
        if isinstance(lhs, onnx.TyArrayNumber | float | int):
            lhs_arr = astyarray(lhs)
            data = (astyarray(lhs) / self._data).astype(onnx.float64)
            data_int64 = where(
                safe_cast(
                    onnx.TyArrayBool,
                    self.is_nat | (lhs_arr == _NAT_SENTINEL) | data.isnan(),
                ),
                _NAT_SENTINEL,
                data,
            ).astype(onnx.int64)
            return self.dtype._build(data_int64)
        return NotImplemented

    def __ndx_equal__(self, other) -> onnx.TyArrayBool:
        # TODO: Figure out what is missing to be able to narrow `other` via the data type
        if isinstance(other, TyArrayTimeDelta):
            if self.dtype.unit != other.dtype.unit:
                raise TypeError(
                    "comparison between different units is not implemented, yet"
                )

            res = self._data == other._data
            is_nat = self.is_nat | other.is_nat
            return res & ~is_nat
        return NotImplemented


class TyArrayDateTime(TimeBaseArray):
    _dtype: DateTime

    def __init__(self, data: onnx.TyArrayInt64, unit: Unit):
        self._data = safe_cast(onnx.TyArrayInt64, data)
        self._dtype = DateTime(unit)

    @property
    def dtype(self) -> DateTime:
        return self._dtype

    def unwrap_numpy(self) -> np.ndarray:
        is_nat = self.is_nat.unwrap_numpy()
        data = self._data.unwrap_numpy()

        out = data.astype(f"datetime64[{self.dtype.unit}]")
        out[is_nat] = np.array("NaT", "datetime64")
        return out

    def __ndx_cast_to__(
        self, dtype: DType[TY_ARRAY_BASE]
    ) -> TY_ARRAY_BASE | NotImplementedType:
        # TODO: Figure out why mypy/pyright don't like this method
        if isinstance(dtype, onnx.Int64):
            # Disallow casting to smaller integer types since that
            # would cause an overflow for NaT values
            data = where(self.is_nat, astyarray(np.iinfo(np.int64).min), self._data)
            return data.astype(dtype)  # type: ignore
        if isinstance(dtype, DateTime):
            return _convert_unit(self, dtype)
        return NotImplemented

    def _coerce_to_time_delta(
        self, other: TyArrayBase | PyScalar
    ) -> TyArrayTimeDelta | NotImplementedType:
        """Coerce to a time delta array.

        The units may not align after this operation.

        This is useful for comparisons and ``__add__``.
        """
        from .funcs import astyarray

        if isinstance(other, int):
            other = astyarray(other, dtype=onnx.int64)
        if isinstance(other, onnx.TyArrayInt64):
            other = other.astype(TimeDelta(unit=self.dtype.unit))
        if isinstance(other, TyArrayTimeDelta):
            return other
        return NotImplemented

    def _coerce_to_date_time(
        self, other: TyArrayBase | PyScalar
    ) -> TyArrayDateTime | NotImplementedType:
        """Coerce `other` to ``TyArrayDateTime`` but the units may not align
        afterwards."""
        if isinstance(other, int):
            other = astyarray(other, dtype=onnx.int64)
        if isinstance(other, onnx.TyArrayInt64):
            other = other.astype(DateTime(unit=self.dtype.unit))
        if isinstance(other, TyArrayDateTime):
            return other
        return NotImplemented

    def __add__(self, rhs: TyArrayBase | PyScalar) -> Self:
        rhs = self._coerce_to_time_delta(rhs)

        if rhs is NotImplemented:
            return NotImplemented

        lhs, rhs = _coerce_units(self, rhs)

        data = lhs._data + rhs._data
        is_nat = lhs.is_nat | rhs.is_nat

        return type(lhs)(where(is_nat, _NAT_SENTINEL, data), unit=lhs.dtype.unit)

    def __radd__(
        self, lhs: TyArrayBase | PyScalar
    ) -> TyArrayDateTime | TyArrayTimeDelta:
        return self.__add__(lhs)

    def _sub(self, other, forward: bool):
        if isinstance(other, int):
            other_ = TimeDelta(self.dtype.unit)._build(
                astyarray(other, dtype=onnx.int64)
            )
            return self - other_ if forward else other_ - self

        if isinstance(other, TyArrayDateTime):
            a, b = _coerce_units(self, other)
            is_nat = a.is_nat | b.is_nat
            data = safe_cast(
                onnx.TyArrayInt64, a._data - b._data if forward else b._data - a._data
            )
            return TyArrayTimeDelta(
                data=where(is_nat, _NAT_SENTINEL, data), unit=a.dtype.unit
            )

        elif isinstance(other, TyArrayTimeDelta) and forward:
            # *_ due to types of various locals set in the previous if statement
            a_, b_ = _coerce_units(self, other)
            is_nat = a_.is_nat | b_.is_nat
            data = safe_cast(
                onnx.TyArrayInt64,
                a_._data - b_._data if forward else b_._data - a_._data,
            )
            return type(self)(
                data=where(is_nat, _NAT_SENTINEL, data), unit=a_.dtype.unit
            )

        return NotImplemented

    def __sub__(
        self, rhs: TyArrayBase | PyScalar
    ) -> TyArrayDateTime | TyArrayTimeDelta:
        return self._sub(rhs, True)

    def __rsub__(
        self, lhs: TyArrayBase | PyScalar
    ) -> TyArrayDateTime | TyArrayTimeDelta:
        return self._sub(lhs, False)

    def __ndx_equal__(self, other) -> onnx.TyArrayBool:
        if not isinstance(other, TyArrayBase):
            return NotImplemented

        if not isinstance(other, TyArrayDateTime):
            return NotImplemented
        if self.dtype.unit != other.dtype.unit:
            raise TypeError(
                "comparison between different units is not implemented, yet"
            )

        res = self._data == other._data
        is_nat = self.is_nat | other.is_nat

        return safe_cast(onnx.TyArrayBool, res & ~is_nat)


def _apply_op(
    this: TyArrayTimeDelta,
    other: TyArrayTimeDelta | int,
    op: Callable[[Any, Any], Any],
    forward: bool,
) -> TyArrayTimeDelta:
    other_data, other_is_nat = _coerce_other(this, other)
    if other_data is NotImplemented:
        return NotImplemented

    if forward:
        data_ = op(this._data, other_data)
    else:
        data_ = op(other_data, this._data)
    data = safe_cast(onnx.TyArrayInt64, data_)
    is_nat = this.is_nat | other_is_nat

    return this.dtype._build(where(is_nat, _NAT_SENTINEL, data))


T1 = TypeVar("T1", bound="TimeBaseArray")
T2 = TypeVar("T2", bound="TimeBaseArray")


def _coerce_other(
    this: T1, other: T1 | int
) -> tuple[onnx.TyArrayInt64, onnx.TyArrayBool] | NotImplementedType:
    """Validate that dtypes are compatible and get ``data`` and ``is_nat`` mask from
    other."""
    from .funcs import astyarray

    if isinstance(other, int):
        return (astyarray(other, dtype=onnx.int64), _NAT_SENTINEL == other)
    elif type(this) is type(other):
        if this.dtype.unit != other.dtype.unit:
            raise ValueError("inter-operation between time units is not implemented")
        return (other._data, other.is_nat)
    return NotImplemented


def _coerce_units(a: T1, b: T2) -> tuple[T1, T2]:
    table: dict[tuple[Unit, Unit], Unit] = {
        ("ns", "s"): "ns",
        ("s", "ns"): "ns",
        ("s", "s"): "s",
        ("ns", "ns"): "ns",
    }
    target = table[(a.dtype.unit, b.dtype.unit)]
    dtype_a = type(a.dtype)(unit=target)
    dtype_b = type(b.dtype)(unit=target)
    return (a.astype(dtype_a), b.astype(dtype_b))


def validate_unit(unit: str) -> Literal["ns", "s"]:
    if unit in ["ns", "s"]:
        return unit  # type: ignore
    raise ValueError(f"unsupported datetime unit `{unit}`")


def _convert_unit(arr: TimeBaseArray, new: BaseTimeDType[TIMEARRAY_co]) -> TIMEARRAY_co:
    new_unit = new.unit
    old_unit = arr.dtype.unit
    powers = {
        "s": 0,
        "ms": 3,
        "us": 6,
        "ns": 9,
    }
    power = powers[new_unit] - powers[old_unit]  # type: ignore
    if powers == 0:
        return arr.copy()  # type: ignore
    elif power > 0:
        data = arr._data * np.pow(10, power)
    else:
        data = arr._data // np.pow(10, abs(power))

    data = where(arr.is_nat, astyarray(np.iinfo(np.int64).min), data)
    data = safe_cast(onnx.TyArrayInt64, data)

    return new._build(data=data)
