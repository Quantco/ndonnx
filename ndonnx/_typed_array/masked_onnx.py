# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import operator
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from types import NotImplementedType
from typing import TYPE_CHECKING, TypeVar, overload

import numpy as np
from typing_extensions import Self

from ndonnx import DateTime64DType, DType, TimeDelta64DType
from ndonnx._experimental import (
    DTypeInfoV1,
    GetitemIndex,
    SetitemIndex,
    TyArrayBase,
    onnx,
    safe_cast,
)
from ndonnx.types import NestedSequence, OnnxShape, PyScalar

if TYPE_CHECKING:
    from spox import Var


DTYPE = TypeVar("DTYPE", bound=DType)

TY_MA_ARRAY_ONNX = TypeVar("TY_MA_ARRAY_ONNX", bound="TyMaArray", covariant=True)
TY_ARRAY_BASE_co = TypeVar("TY_ARRAY_BASE_co", bound="TyArrayBase", covariant=True)
KEY = TypeVar("KEY", int, float, str)
VALUE = TypeVar("VALUE", int, float, str)


class _MaOnnxDType(DType[TY_MA_ARRAY_ONNX]):
    _unmasked_dtype: onnx._OnnxDType

    def __ndx_create__(
        self, val: PyScalar | np.ndarray | TyArrayBase | Var | NestedSequence
    ) -> TY_MA_ARRAY_ONNX:
        if isinstance(val, np.ma.MaskedArray):
            data = onnx.const(val.data)
            if val.mask is np.ma.nomask:
                mask = None
            else:
                mask = safe_cast(onnx.TyArrayBool, onnx.const(val.mask))
            return make_nullable(data, mask).astype(self)
        if isinstance(val, TyMaArray):
            return val.astype(self)
        return make_nullable(self._unmasked_dtype.__ndx_create__(val), None).astype(
            self
        )

    def __str__(self) -> str:
        return type(self).__name__.lower()

    @abstractmethod
    def _build(
        self, data: onnx.TyArray, mask: onnx.TyArrayBool | None
    ) -> TY_MA_ARRAY_ONNX: ...

    def __ndx_cast_from__(self, arr: TyArrayBase) -> TY_MA_ARRAY_ONNX:
        if isinstance(arr, onnx.TyArray):
            return make_nullable(arr, None).astype(self)
        if isinstance(arr, TyMaArray):
            mask = arr.mask
            data_ = arr.data.astype(self._unmasked_dtype)
            return self._build(data=data_, mask=mask)
        return NotImplemented

    @property
    def __ndx_infov1__(self) -> DTypeInfoV1:
        return DTypeInfoV1(
            author="ndonnx", type_name=self.__class__.__name__, meta=None
        )

    def __ndx_argument__(self, shape: OnnxShape) -> TY_MA_ARRAY_ONNX:
        data = self._unmasked_dtype.__ndx_argument__(shape)
        mask = onnx.bool_.__ndx_argument__(shape)
        return self._build(data=data, mask=mask)

    def __ndx_arange__(
        self,
        start: int | float,
        stop: int | float,
        step: int | float = 1,
    ) -> TY_MA_ARRAY_ONNX:
        # Get everything onto the same type
        data = self._unmasked_dtype.__ndx_arange__(start, stop, step)
        return self._build(data=data, mask=None)

    def __ndx_eye__(
        self,
        n_rows: int,
        n_cols: int | None = None,
        /,
        *,
        k: int = 0,
    ) -> TY_MA_ARRAY_ONNX:
        data = self._unmasked_dtype.__ndx_eye__(n_rows, n_cols, k=k)

        return self._build(data=data, mask=None)

    def __ndx_ones__(
        self, shape: tuple[int, ...] | onnx.TyArrayInt64
    ) -> TY_MA_ARRAY_ONNX:
        data = self._unmasked_dtype.__ndx_ones__(shape)
        return self._build(data=data, mask=None)

    def __ndx_zeros__(
        self, shape: tuple[int, ...] | onnx.TyArrayInt64
    ) -> TY_MA_ARRAY_ONNX:
        data = self._unmasked_dtype.__ndx_zeros__(shape)
        return self._build(data=data, mask=None)


class _NNumber(_MaOnnxDType):
    def __ndx_result_type__(self, rhs: DType | PyScalar) -> DType | NotImplementedType:
        if isinstance(rhs, onnx.NumericDTypes | int | float):
            onnx_result = onnx.result_type(self._unmasked_dtype, rhs)
        elif isinstance(rhs, NumericDTypes):
            onnx_result = onnx.result_type(self._unmasked_dtype, rhs._unmasked_dtype)

        else:
            # No implicit promotion for bools and strings
            return NotImplemented

        return to_nullable_dtype(onnx_result)


class NUtf8(_MaOnnxDType):
    _unmasked_dtype = onnx.utf8

    def __ndx_result_type__(self, rhs: DType | PyScalar) -> DType | NotImplementedType:
        if isinstance(rhs, onnx.Utf8 | str):
            return self
        return NotImplemented

    def _build(
        self, data: onnx.TyArray, mask: onnx.TyArrayBool | None
    ) -> TyMaArrayString:
        return TyMaArrayString(data, mask)


class NBool(_MaOnnxDType):
    _unmasked_dtype = onnx.bool_

    def __ndx_result_type__(self, rhs: DType | PyScalar) -> DType | NotImplementedType:
        return NotImplemented

    def _build(
        self, data: onnx.TyArray, mask: onnx.TyArrayBool | None
    ) -> TyMaArrayBool:
        return TyMaArrayBool(data, mask)

    @property
    def __ndx_infov1__(self) -> DTypeInfoV1:
        # Override to be compatible with the existing schema
        return DTypeInfoV1(author="ndonnx", type_name="NBoolean", meta=None)


class NInt8(_NNumber):
    _unmasked_dtype = onnx.int8

    def _build(
        self, data: onnx.TyArray, mask: onnx.TyArrayBool | None
    ) -> TyMaArrayInt8:
        return TyMaArrayInt8(data, mask)


class NInt16(_NNumber):
    _unmasked_dtype = onnx.int16

    def _build(
        self, data: onnx.TyArray, mask: onnx.TyArrayBool | None
    ) -> TyMaArrayInt16:
        return TyMaArrayInt16(data, mask)


class NInt32(_NNumber):
    _unmasked_dtype = onnx.int32

    def _build(
        self, data: onnx.TyArray, mask: onnx.TyArrayBool | None
    ) -> TyMaArrayInt32:
        return TyMaArrayInt32(data, mask)


class NInt64(_NNumber):
    _unmasked_dtype = onnx.int64

    def _build(
        self, data: onnx.TyArray, mask: onnx.TyArrayBool | None
    ) -> TyMaArrayInt64:
        return TyMaArrayInt64(data, mask)


class NUInt8(_NNumber):
    _unmasked_dtype = onnx.uint8

    def _build(
        self, data: onnx.TyArray, mask: onnx.TyArrayBool | None
    ) -> TyMaArrayUInt8:
        return TyMaArrayUInt8(data, mask)


class NUInt16(_NNumber):
    _unmasked_dtype = onnx.uint16

    def _build(
        self, data: onnx.TyArray, mask: onnx.TyArrayBool | None
    ) -> TyMaArrayUInt16:
        return TyMaArrayUInt16(data, mask)


class NUInt32(_NNumber):
    _unmasked_dtype = onnx.uint32

    def _build(
        self, data: onnx.TyArray, mask: onnx.TyArrayBool | None
    ) -> TyMaArrayUInt32:
        return TyMaArrayUInt32(data, mask)


class NUInt64(_NNumber):
    _unmasked_dtype = onnx.uint64

    def _build(
        self, data: onnx.TyArray, mask: onnx.TyArrayBool | None
    ) -> TyMaArrayUInt64:
        return TyMaArrayUInt64(data, mask)


class NFloat16(_NNumber):
    _unmasked_dtype = onnx.float16

    def _build(
        self, data: onnx.TyArray, mask: onnx.TyArrayBool | None
    ) -> TyMaArrayFloat16:
        return TyMaArrayFloat16(data, mask)


class NFloat32(_NNumber):
    _unmasked_dtype = onnx.float32

    def _build(
        self, data: onnx.TyArray, mask: onnx.TyArrayBool | None
    ) -> TyMaArrayFloat32:
        return TyMaArrayFloat32(data, mask)


class NFloat64(_NNumber):
    _unmasked_dtype = onnx.float64

    def _build(
        self, data: onnx.TyArray, mask: onnx.TyArrayBool | None
    ) -> TyMaArrayFloat64:
        return TyMaArrayFloat64(data, mask)


# Nullable Singleton instances
nbool = NBool()

nfloat16 = NFloat16()
nfloat32 = NFloat32()
nfloat64 = NFloat64()

nint8 = NInt8()
nint16 = NInt16()
nint32 = NInt32()
nint64 = NInt64()

nuint8 = NUInt8()
nuint16 = NUInt16()
nuint32 = NUInt32()
nuint64 = NUInt64()

nutf8 = NUtf8()

# Union types
#
# Use `ndonnx.is_*_dtype` instead if possible
IntegerDTypes = NInt8 | NInt16 | NInt32 | NInt64 | NUInt8 | NUInt16 | NUInt32 | NUInt64
FloatDTypes = NFloat16 | NFloat32 | NFloat64

NumericDTypes = FloatDTypes | IntegerDTypes

DTypes = NBool | NumericDTypes | NUtf8


def _make_binary_pair(
    fun: Callable[[onnx.TyArray, onnx.TyArray | PyScalar], onnx.TyArray],
):
    """Helper to define dunder methods.

    Does not work with proper type hints, though.
    """

    def forward(self, rhs: TyArrayBase | PyScalar) -> TyArrayBase:
        return _apply_op(self, rhs, fun, True)

    def backward(self, lhs) -> TyArrayBase:
        return _apply_op(self, lhs, fun, False)

    return forward, backward


def _make_unary_member_promoted_type(fun_name: str):
    def impl(self) -> TyMaArray:
        data = getattr(self.data, fun_name)()
        return make_nullable(data, self.mask)

    return impl


def _make_unary_member_same_type(fun_name: str):
    """Helper to define dunder methods.

    Does not work with proper type hints, though.
    """

    def impl(self: TyMaArray) -> TyArrayBase:
        data = getattr(self.data, fun_name)()
        return type(self)(data=data, mask=self.mask)

    return impl


class TyMaArrayBase(TyArrayBase):
    """Typed masked array object."""

    mask: onnx.TyArrayBool | None

    @property
    @abstractmethod
    def data(self) -> TyArrayBase: ...

    def __ndx_value_repr__(self) -> dict[str, str]:
        reps = {}
        reps["data"] = self.data.__ndx_value_repr__()["data"]
        reps["mask"] = (
            "None" if self.mask is None else self.mask.__ndx_value_repr__()["data"]
        )
        return reps


class TyMaArray(TyMaArrayBase):
    """Masked version of core types."""

    _dtype: _MaOnnxDType
    # Specialization of data from `Data` to `_ArrayCoreType`
    _data: onnx.TyArray

    def __init__(self, data: onnx.TyArray, mask: onnx.TyArrayBool | None):
        self.data = data
        self.mask = mask

    @property
    def dtype(self) -> _MaOnnxDType:
        # Implemented in child class
        raise NotImplementedError

    @property
    def data(self) -> onnx.TyArray:
        return self._data

    @data.setter
    def data(self, data: onnx.TyArray):
        if data.dtype != self.dtype._unmasked_dtype:
            raise ValueError(
                f"expected data of type `{self.dtype._unmasked_dtype}` found `{data.dtype}`"
            )
        self._data = data

    def disassemble(self) -> dict[str, Var]:
        # Ensure that we always return a mask for backwards compatibility.
        # TODO: remove this
        if self.mask is None:
            mask = onnx.const(False, dtype=onnx.bool_).broadcast_to(self.dynamic_shape)
        else:
            mask = self.mask
        return {
            # Maintain compatibility with existing schema
            "values": self.data.disassemble(),
            "null": mask.disassemble(),
        }

    @property
    def mT(self) -> Self:  # noqa: N802
        data = self.data.mT
        mask = self.mask.mT if self.mask is not None else None
        return type(self)(data=data, mask=mask)

    @property
    def dynamic_shape(self) -> onnx.TyArrayInt64:
        return self.data.dynamic_shape

    @property
    def shape(self) -> OnnxShape:
        return self.data.shape

    @property
    def is_constant(self) -> bool:
        if self.mask is None:
            return self.data.is_constant
        return self.data.is_constant and self.data.is_constant

    def _pass_through_same_type(self, fun_name: str, *args, **kwargs) -> Self:
        data = getattr(self.data, fun_name)(*args, **kwargs)
        mask = (
            getattr(self.mask, fun_name)(*args, **kwargs)
            if self.mask is not None
            else None
        )
        return type(self)(data=data, mask=mask)

    def fill_null(self, value: int | float | bool | str) -> onnx.TyArray:
        dtype = onnx.result_type(self.data.dtype, value)
        if self.mask is None:
            return self.data.astype(dtype)
        value_arr = onnx.const(value, dtype)
        return onnx.where(self.mask, value_arr, self.data)

    def permute_dims(self, axes: tuple[int, ...]) -> Self:
        return self._pass_through_same_type("permute_dims", axes=axes)

    def reshape(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> Self:
        return self._pass_through_same_type("reshape", shape=shape)

    def squeeze(self, /, axis: int | tuple[int, ...]) -> Self:
        return self._pass_through_same_type("squeeze", axis=axis)

    def take(self, indices: onnx.TyArrayInt64, /, *, axis: int | None = None) -> Self:
        return self._pass_through_same_type("take", indices, axis=axis)

    def tril(self, /, *, k: int = 0) -> Self:
        return self._pass_through_same_type("tril", k=k)

    def triu(self, /, *, k: int = 0) -> Self:
        return self._pass_through_same_type("triu", k=k)

    def broadcast_to(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> Self:
        return self._pass_through_same_type("broadcast_to", shape=shape)

    def copy(self) -> Self:
        # We want to copy the component arrays, too.
        mask = None if self.mask is None else self.mask.copy()
        return type(self)(data=self.data.copy(), mask=mask)

    def __copy__(self) -> Self:
        return self.copy()

    def unwrap_numpy(self) -> np.ndarray:
        return np.ma.MaskedArray(
            data=self.data.unwrap_numpy(),
            mask=None if self.mask is None else self.mask.unwrap_numpy(),
        )

    def __getitem__(self, key: GetitemIndex) -> Self:
        return self._pass_through_same_type("__getitem__", key=key)

    def __setitem__(self, index: SetitemIndex, value: Self) -> None:
        self.data[index] = value.data
        if self.mask is None and value.mask is None:
            return
        if self.mask is None:
            # Create a new mask for self
            self.mask = onnx.const(False, onnx.bool_).broadcast_to(self.dynamic_shape)
        if value.mask is None:
            self.mask[index] = onnx.const(False, onnx.bool_)
        else:
            self.mask[index] = value.mask

    def put(
        self,
        key: onnx.TyArrayInt64,
        value: Self,
        /,
    ) -> None:
        self.dtype.__ndx_ones__
        self.data.put(key, value.data)
        if value.mask is not None:
            if self.mask is None:
                self.mask = onnx.const(False, onnx.bool_).broadcast_to(
                    self.dynamic_shape
                )
            self.mask.put(key, value.mask)

    def __ndx_cast_to__(self, dtype: DType[TY_ARRAY_BASE_co]) -> TY_ARRAY_BASE_co:
        if isinstance(dtype, onnx._OnnxDType):
            # Not clear what the behavior should be if we have a mask
            return NotImplemented
        elif isinstance(dtype, _MaOnnxDType):
            new_data = self.data.astype(dtype._unmasked_dtype)
            dtype._build(data=new_data, mask=self.mask)
        elif isinstance(dtype, DateTime64DType | TimeDelta64DType):
            is_nat = self.mask
            if isinstance(self._data, onnx.TyArrayFloating):
                if is_nat is None:
                    is_nat = self._data.isnan()
                else:
                    is_nat |= self._data.isnan()
            dt_data = self._data.astype(onnx.int64)
            if is_nat is not None:
                nat = onnx.const(int(np.iinfo(np.int64).min), dtype=onnx.int64)
                dt_data = onnx.where(is_nat, nat, dt_data)
            return dtype._build(dt_data)  # type: ignore
        return NotImplemented

    def concat(self, others: list[Self], axis: None | int) -> Self:
        data = self.data.concat([el.data for el in others], axis)
        if all(el.mask is None for el in [self] + others):
            mask = None
        else:
            masks = []
            for el in [self] + others:
                masks.append(
                    onnx.const(False, onnx.bool_).broadcast_to(self.data.dynamic_shape)
                    if el.mask is None
                    else el.mask
                )
            mask = safe_cast(onnx.TyArrayBool, masks[0].concat(masks[1:], axis))
        return safe_cast(type(self), make_nullable(data, mask))

    def apply_mapping(self, mapping: Mapping[KEY, VALUE], default: VALUE) -> TyMaArray:
        data = self._data.apply_mapping(mapping, default=default)
        return make_nullable(data, mask=self.mask)

    def __ndx_equal__(self, other) -> TyArrayBase | NotImplementedType:
        return _apply_op(self, other, operator.eq, True)

    def __ndx_where__(
        self, cond: onnx.TyArrayBool, y: TyArrayBase | PyScalar, /
    ) -> TyArrayBase:
        if isinstance(y, onnx.TyArray | PyScalar):
            _, y = onnx.promote(self._data, y)
            return self.__ndx_where__(cond, make_nullable(y, None))
        if isinstance(y, TyMaArray):
            x_ = self._data
            y_ = y._data
            new_data = x_.__ndx_where__(cond, y_)
            if self.mask is not None and y.mask is not None:
                new_mask = cond & self.mask | ~cond & y.mask
            elif self.mask is not None:
                new_mask = cond & self.mask
            elif y.mask is not None:
                new_mask = ~cond & y.mask
            else:
                new_mask = None

            if new_mask is not None and not isinstance(new_mask, onnx.TyArrayBool):
                # Should never happen. Might be worth while adding
                # overloads to the BoolData dunder methods to
                # propagate the types more precisely.
                raise TypeError(f"expected boolean mask, found `{new_mask.dtype}`")

            return make_nullable(new_data, new_mask)

        return NotImplemented

    def __ndx_rwhere__(
        self, cond: onnx.TyArrayBool, x: TyArrayBase | PyScalar, /
    ) -> TyArrayBase | NotImplementedType:
        if isinstance(x, onnx.TyArray | PyScalar):
            _, x = onnx.promote(self._data, x)
            return make_nullable(x, None).__ndx_where__(cond, self)
        return NotImplemented

    def isin(self, items: Sequence[onnx.VALUE]) -> onnx.TyArrayBool:
        data = self.data.isin(items)
        # Masked values always return False
        if self.mask is None:
            return data
        return safe_cast(onnx.TyArrayBool, data & ~self.mask)


class TyMaArrayString(TyMaArray):
    @property
    def dtype(self) -> NUtf8:
        return nutf8

    __add__, __radd__ = _make_binary_pair(operator.add)  # type: ignore


class TyMaArrayNumber(TyMaArray):
    @property
    def dtype(self) -> NumericDTypes:
        raise NotImplementedError

    def clip(
        self, min: TyArrayBase | None = None, max: TyArrayBase | None = None
    ) -> Self:
        allowed_types = int | float | onnx.TyArray | TyMaArrayNumber
        if min is not None and not isinstance(min, allowed_types):
            raise TypeError(
                f"'clip' is not implemented for argument 'min' with data type `{min.dtype}`"
            )
        if max is not None and not isinstance(max, allowed_types):
            raise TypeError(
                f"'clip' is not implemented for argument 'max' with data type `{max.dtype}`"
            )
        mask = self.mask
        if isinstance(min, TyMaArrayNumber):
            mask = merge_masks(mask, min.mask)
            min = min.data
        if isinstance(max, TyMaArrayNumber):
            mask = merge_masks(mask, max.mask)
            max = max.data

        data = self.data.clip(min, max)

        return type(self)(data=data, mask=mask)

    @overload
    def prod(
        self,
        /,
        *,
        dtype: DType[TY_ARRAY_BASE_co],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> TY_ARRAY_BASE_co: ...

    @overload
    def prod(
        self,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
    ) -> TyArrayBase: ...

    def prod(
        self,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
    ) -> TyArrayBase:
        return self.fill_null(1).prod(axis=axis, dtype=dtype, keepdims=keepdims)

    def __ndx_maximum__(self, rhs: TyArrayBase | PyScalar, /) -> TyArrayBase:
        if isinstance(rhs, PyScalar):
            rhs = onnx.const(rhs)
        if isinstance(rhs, onnx.TyArray):
            return self.__ndx_maximum__(make_nullable(rhs, None))
        if isinstance(rhs, TyMaArray):
            lhs_data = self._data
            rhs_data = rhs._data
            new_data = safe_cast(onnx.TyArray, lhs_data.__ndx_maximum__(rhs_data))
            if new_data is NotImplemented:
                return new_data
            new_mask = merge_masks(self.mask, rhs.mask)
            return make_nullable(new_data, new_mask)

        return NotImplemented

    def __ndx_minimum__(self, rhs: TyArrayBase | PyScalar, /) -> TyArrayBase:
        if isinstance(rhs, PyScalar):
            rhs = onnx.const(rhs)
        if isinstance(rhs, onnx.TyArray):
            return self.__ndx_minimum__(make_nullable(rhs, None))
        if isinstance(rhs, TyMaArray):
            lhs_data = self._data
            rhs_data = rhs._data
            new_data = safe_cast(onnx.TyArray, lhs_data.__ndx_minimum__(rhs_data))
            if new_data is NotImplemented:
                return new_data
            new_mask = merge_masks(self.mask, rhs.mask)
            return make_nullable(new_data, new_mask)

        return NotImplemented

    # Dunder implementations
    #
    # We don't differentiate between the different subclasses since we
    # always just pass through to the underlying non-masked typed. We
    # will get an error from there if appropriate.
    __add__, __radd__ = _make_binary_pair(operator.add)  # type: ignore
    __sub__, __rsub__ = _make_binary_pair(operator.sub)  # type: ignore
    __mod__, __rmod__ = _make_binary_pair(operator.mod)  # type: ignore
    __mul__, __rmul__ = _make_binary_pair(operator.mul)  # type: ignore
    __truediv__, __rtruedive__ = _make_binary_pair(operator.truediv)  # type: ignore
    __ge__, _ = _make_binary_pair(operator.ge)  # type: ignore
    __le__, _ = _make_binary_pair(operator.le)  # type: ignore
    __gt__, _ = _make_binary_pair(operator.gt)  # type: ignore
    __lt__, _ = _make_binary_pair(operator.lt)  # type: ignore
    __pow__, __rpow__ = _make_binary_pair(operator.pow)  # type: ignore
    __floordiv__, __rfloordiv__ = _make_binary_pair(operator.floordiv)  # type: ignore

    __neg__ = _make_unary_member_promoted_type("__neg__")  # type: ignore
    __invert__ = _make_unary_member_same_type("__invert__")  # type: ignore

    __abs__ = _make_unary_member_same_type("__abs__")  # type: ignore
    __pos__ = _make_unary_member_same_type("__pos__")  # type: ignore
    ceil = _make_unary_member_same_type("ceil")  # type: ignore
    floor = _make_unary_member_same_type("floor")  # type: ignore

    def isfinite(self) -> TyMaArrayBool:
        return TyMaArrayBool(
            data=safe_cast(onnx.TyArrayBool, self._data.isfinite()), mask=self.mask
        )

    def isinf(self) -> TyMaArrayBool:
        return TyMaArrayBool(
            data=safe_cast(onnx.TyArrayBool, self._data.isinf()), mask=self.mask
        )

    def isnan(self) -> TyMaArrayBool:
        return TyMaArrayBool(
            data=safe_cast(onnx.TyArrayBool, self._data.isnan()), mask=self.mask
        )

    round = _make_unary_member_same_type("round")  # type: ignore
    sign = _make_unary_member_same_type("sign")  # type: ignore
    sqrt = _make_unary_member_same_type("sqrt")  # type: ignore
    square = _make_unary_member_same_type("square")  # type: ignore
    trunc = _make_unary_member_same_type("trunc")  # type: ignore


class TyMaArrayInteger(TyMaArrayNumber):
    @property
    def dtype(self) -> IntegerDTypes:
        raise NotImplementedError

    __and__, __rand__ = _make_binary_pair(operator.and_)  # type: ignore
    __or__, __ror__ = _make_binary_pair(operator.or_)  # type: ignore
    __xor__, __rxor__ = _make_binary_pair(operator.xor)  # type: ignore
    __lshift__, __rlshift__ = _make_binary_pair(operator.lshift)  # type: ignore
    __rshift__, __rrshift__ = _make_binary_pair(operator.rshift)  # type: ignore
    __invert__ = _make_unary_member_same_type("__invert__")  # type: ignore


class TyMaArrayFloating(TyMaArrayNumber):
    @property
    def dtype(self) -> FloatDTypes:
        raise NotImplementedError

    acos = _make_unary_member_same_type("acos")  # type: ignore
    acosh = _make_unary_member_same_type("acosh")  # type: ignore
    asin = _make_unary_member_same_type("asin")  # type: ignore
    asinh = _make_unary_member_same_type("asinh")  # type: ignore
    atan = _make_unary_member_same_type("atan")  # type: ignore
    atanh = _make_unary_member_same_type("atanh")  # type: ignore
    cos = _make_unary_member_same_type("cos")  # type: ignore
    cosh = _make_unary_member_same_type("cosh")  # type: ignore
    exp = _make_unary_member_same_type("exp")  # type: ignore
    log = _make_unary_member_same_type("log")  # type: ignore
    log2 = _make_unary_member_same_type("log2")  # type: ignore
    log10 = _make_unary_member_same_type("log10")  # type: ignore
    sin = _make_unary_member_same_type("sin")  # type: ignore
    sinh = _make_unary_member_same_type("sinh")  # type: ignore
    tan = _make_unary_member_same_type("tan")  # type: ignore
    tanh = _make_unary_member_same_type("tanh")  # type: ignore


class TyMaArrayBool(TyMaArray):
    @property
    def dtype(self) -> NBool:
        return nbool

    __invert__ = _make_unary_member_same_type("__invert__")  # type: ignore

    __and__, __rand__ = _make_binary_pair(operator.and_)  # type: ignore
    __or__, __ror__ = _make_binary_pair(operator.or_)  # type: ignore
    __xor__, __rxor__ = _make_binary_pair(operator.xor)  # type: ignore
    __lshift__, __rlshift__ = _make_binary_pair(operator.lshift)  # type: ignore
    __rshift__, __rrshift__ = _make_binary_pair(operator.rshift)  # type: ignore


class TyMaArrayInt8(TyMaArrayInteger):
    @property
    def dtype(self) -> NInt8:
        return nint8


class TyMaArrayInt16(TyMaArrayInteger):
    @property
    def dtype(self) -> NInt16:
        return nint16


class TyMaArrayInt32(TyMaArrayInteger):
    @property
    def dtype(self) -> NInt32:
        return nint32


class TyMaArrayInt64(TyMaArrayInteger):
    @property
    def dtype(self) -> NInt64:
        return nint64


class TyMaArrayUInt8(TyMaArrayInteger):
    @property
    def dtype(self) -> NUInt8:
        return nuint8


class TyMaArrayUInt16(TyMaArrayInteger):
    @property
    def dtype(self) -> NUInt16:
        return nuint16


class TyMaArrayUInt32(TyMaArrayInteger):
    @property
    def dtype(self) -> NUInt32:
        return nuint32


class TyMaArrayUInt64(TyMaArrayInteger):
    @property
    def dtype(self) -> NUInt64:
        return nuint64


class TyMaArrayFloat16(TyMaArrayFloating):
    @property
    def dtype(self) -> NFloat16:
        return nfloat16


class TyMaArrayFloat32(TyMaArrayFloating):
    @property
    def dtype(self) -> NFloat32:
        return nfloat32


class TyMaArrayFloat64(TyMaArrayFloating):
    @property
    def dtype(self) -> NFloat64:
        return nfloat64


def merge_masks(
    a: onnx.TyArrayBool | None, b: onnx.TyArrayBool | None
) -> onnx.TyArrayBool | None:
    """Merge two masks."""
    if a is None:
        return b
    if b is None:
        return a
    out = a | b
    if isinstance(out, onnx.TyArrayBool):
        return out
    # Should never happen
    raise TypeError("Unexpected array type")


def make_nullable(core_array: onnx.TyArray, mask: onnx.TyArrayBool | None) -> TyMaArray:
    return to_nullable_dtype(core_array.dtype)._build(core_array, mask)


def _apply_op(
    this: TyMaArray,
    other: TyArrayBase | PyScalar,
    op: Callable[[onnx.TyArray, onnx.TyArray | PyScalar], onnx.TyArray],
    forward: bool,
) -> TyMaArray:
    """Apply an operation by passing it through to the data member."""
    if isinstance(other, PyScalar):
        dtype = onnx.result_type(this.dtype._unmasked_dtype, other)
        other = onnx.const(other, dtype)
    if isinstance(other, onnx.TyArray):
        this_, other = onnx.promote(this.data, other)
        data = op(this_, other) if forward else op(other, this_)
        mask = this.mask
    elif isinstance(other, TyMaArray):
        data = op(this.data, other.data) if forward else op(other.data, this.data)
        mask = merge_masks(this.mask, other.mask)
    else:
        return NotImplemented

    return make_nullable(data, mask)


####################
# Conversion Table #
####################

_primitive_to_nullable: dict[onnx._OnnxDType, DTypes] = {
    onnx.bool_: nbool,
    onnx.int8: nint8,
    onnx.int16: nint16,
    onnx.int32: nint32,
    onnx.int64: nint64,
    onnx.uint8: nuint8,
    onnx.uint16: nuint16,
    onnx.uint32: nuint32,
    onnx.uint64: nuint64,
    onnx.float16: nfloat16,
    onnx.float32: nfloat32,
    onnx.float64: nfloat64,
    onnx.utf8: nutf8,
}


def to_nullable_dtype(dtype: onnx._OnnxDType | DTypes) -> DTypes:
    if isinstance(dtype, DTypes):
        return dtype
    return _primitive_to_nullable[dtype]
