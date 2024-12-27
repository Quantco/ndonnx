# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from types import NotImplementedType
from typing import TYPE_CHECKING, TypeVar, overload

import numpy as np
from typing_extensions import Self

from .._dtypes import TY_ARRAY_BASE, DType
from .._schema import DTypeInfoV1
from . import (
    TyArrayBase,
    astyarray,
    maximum,
    minimum,
    onnx,
    py_scalars,
    result_type,
    safe_cast,
    where,
)

if TYPE_CHECKING:
    from spox import Var

    from .._types import OnnxShape
    from .indexing import GetitemIndex, SetitemIndex
    from .onnx import VALUE


DTYPE = TypeVar("DTYPE", bound=DType)
CORE_DTYPES = TypeVar("CORE_DTYPES", bound=onnx.DTypes)
NCORE_DTYPES = TypeVar("NCORE_DTYPES", bound="NCoreDTypes")

NCORE_NUMERIC_DTYPES = TypeVar("NCORE_NUMERIC_DTYPES", bound="NCoreNumericDTypes")
NCORE_FLOATING_DTYPES = TypeVar("NCORE_FLOATING_DTYPES", bound="NCoreFloatingDTypes")
NCORE_INTEGER_DTYPES = TypeVar("NCORE_INTEGER_DTYPES", bound="NCoreIntegerDTypes")

TY_MA_ARRAY_ONNX = TypeVar("TY_MA_ARRAY_ONNX", bound="TyMaArray")


class _MaOnnxDType(DType[TY_MA_ARRAY_ONNX]):
    _unmasked_dtype: onnx._OnnxDType

    def __ndx_cast_from__(self, arr: TyArrayBase) -> TY_MA_ARRAY_ONNX:
        if isinstance(arr, onnx.TyArray):
            return asncoredata(arr, None).astype(self)
        if isinstance(arr, TyMaArray):
            mask = arr.mask
            data_ = arr.data.astype(self._unmasked_dtype)
            return self._tyarr_class(data=data_, mask=mask)
        raise NotImplementedError

    @property
    def _infov1(self) -> DTypeInfoV1:
        return DTypeInfoV1(
            author="ndonnx", type_name=self.__class__.__name__, meta=None
        )

    def _argument(self, shape: OnnxShape) -> TY_MA_ARRAY_ONNX:
        data = as_non_nullable(self)._argument(shape)
        mask = onnx.bool_._argument(shape)
        return self._tyarr_class(data=data, mask=mask)

    def _arange(
        self,
        start: int | float,
        stop: int | float,
        step: int | float = 1,
    ) -> TY_MA_ARRAY_ONNX:
        # Get everything onto the same type
        data = self._unmasked_dtype._arange(start, stop, step)
        return self._tyarr_class(data=data, mask=None)

    def _eye(
        self,
        n_rows: int,
        n_cols: int | None = None,
        /,
        *,
        k: int = 0,
    ) -> TY_MA_ARRAY_ONNX:
        data = self._unmasked_dtype._eye(n_rows, n_cols, k=k)

        return self._tyarr_class(data=data, mask=None)

    def _ones(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> TY_MA_ARRAY_ONNX:
        data = self._unmasked_dtype._ones(shape)
        return self._tyarr_class(data=data, mask=None)

    def _zeros(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> TY_MA_ARRAY_ONNX:
        data = self._unmasked_dtype._zeros(shape)
        return self._tyarr_class(data=data, mask=None)


class _NNumber(_MaOnnxDType):
    _unmasked_dtype: onnx.NumericDTypes

    def __ndx_result_type__(self, rhs: DType) -> DType | NotImplementedType:
        if isinstance(self, NCoreNumericDTypes) and isinstance(rhs, onnx.NumericDTypes):
            core_result = onnx._result_type_core_numeric(self._unmasked_dtype, rhs)
        elif isinstance(rhs, NCoreNumericDTypes):
            core_result = onnx._result_type_core_numeric(
                self._unmasked_dtype, rhs._unmasked_dtype
            )

        else:
            # No implicit promotion for bools and strings
            return NotImplemented

        return as_nullable(core_result)


class NUtf8(_MaOnnxDType):
    _unmasked_dtype = onnx.utf8

    def __ndx_result_type__(self, rhs: DType) -> DType | NotImplementedType:
        return NotImplemented

    @property
    def _tyarr_class(self) -> type[TyMaArrayString]:
        return TyMaArrayString


class NBoolean(_MaOnnxDType):
    _unmasked_dtype = onnx.bool_

    def __ndx_result_type__(self, rhs: DType) -> DType | NotImplementedType:
        return NotImplemented

    @property
    def _tyarr_class(self) -> type[TyMaArrayBool]:
        return TyMaArrayBool


class NInt8(_NNumber):
    _unmasked_dtype = onnx.int8

    @property
    def _tyarr_class(self) -> type[TyMaArrayInt8]:
        return TyMaArrayInt8


class NInt16(_NNumber):
    _unmasked_dtype = onnx.int16

    @property
    def _tyarr_class(self) -> type[TyMaArrayInt16]:
        return TyMaArrayInt16


class NInt32(_NNumber):
    _unmasked_dtype = onnx.int32

    @property
    def _tyarr_class(self) -> type[TyMaArrayInt32]:
        return TyMaArrayInt32


class NInt64(_NNumber):
    _unmasked_dtype = onnx.int64

    @property
    def _tyarr_class(self) -> type[TyMaArrayInt64]:
        return TyMaArrayInt64


class NUInt8(_NNumber):
    _unmasked_dtype = onnx.uint8

    @property
    def _tyarr_class(self) -> type[TyMaArrayUInt8]:
        return TyMaArrayUInt8


class NUInt16(_NNumber):
    _unmasked_dtype = onnx.uint16

    @property
    def _tyarr_class(self) -> type[TyMaArrayUInt16]:
        return TyMaArrayUInt16


class NUInt32(_NNumber):
    _unmasked_dtype = onnx.uint32

    @property
    def _tyarr_class(self) -> type[TyMaArrayUInt32]:
        return TyMaArrayUInt32


class NUInt64(_NNumber):
    _unmasked_dtype = onnx.uint64

    @property
    def _tyarr_class(self) -> type[TyMaArrayUInt64]:
        return TyMaArrayUInt64


class NFloat16(_NNumber):
    _unmasked_dtype = onnx.float16

    @property
    def _tyarr_class(self) -> type[TyMaArrayFloat16]:
        return TyMaArrayFloat16


class NFloat32(_NNumber):
    _unmasked_dtype = onnx.float32

    @property
    def _tyarr_class(self) -> type[TyMaArrayFloat32]:
        return TyMaArrayFloat32


class NFloat64(_NNumber):
    _unmasked_dtype = onnx.float64

    @property
    def _tyarr_class(self) -> type[TyMaArrayFloat64]:
        return TyMaArrayFloat64


# Nullable Singleton instances
nbool = NBoolean()

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
# Union types are exhaustive and don't create ambiguities with respect to user-defined subtypes.
# TODO: Rename
NCoreIntegerDTypes = (
    NInt8 | NInt16 | NInt32 | NInt64 | NUInt8 | NUInt16 | NUInt32 | NUInt64
)
NCoreFloatingDTypes = NFloat16 | NFloat32 | NFloat64

NCoreNumericDTypes = NCoreFloatingDTypes | NCoreIntegerDTypes

NCoreDTypes = NBoolean | NCoreNumericDTypes | NUtf8


def _make_binary_pair(fun: Callable[[onnx.TyArray, onnx.TyArray], onnx.TyArray]):
    """Helper to define dunder methods.

    Does not work with proper type hints, though.
    """

    def forward(self, rhs) -> TyArrayBase:
        return _apply_op(self, rhs, fun, True)

    def backward(self, lhs) -> TyArrayBase:
        return _apply_op(self, lhs, fun, False)

    return forward, backward


def _make_unary_member_promoted_type(fun_name: str):
    def impl(self) -> TyMaArray:
        data = getattr(self.data, fun_name)()
        return asncoredata(data, self.mask)

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

    data: TyArrayBase
    mask: onnx.TyArrayBool | None

    def __ndx_value_repr__(self) -> dict[str, str]:
        reps = {}
        reps["data"] = self.data.__ndx_value_repr__()["data"]
        reps["mask"] = (
            "None" if self.mask is None else self.mask.__ndx_value_repr__()["data"]
        )
        return reps


class TyMaArray(TyMaArrayBase):
    """Masked version of core types.

    The (subclasses) of this class are implemented such that they only
    know about `_ArrayCoreType`s, but not `_PyScalar`s.
    """

    # TODO: Things should be easier if they were to know about PyScalars

    dtype: _MaOnnxDType
    # Specialization of data from `Data` to `_ArrayCoreType`
    data: onnx.TyArray

    def __init__(self, data: onnx.TyArray, mask: onnx.TyArrayBool | None):
        self.dtype = as_nullable(data.dtype)
        self.data = data
        self.mask = mask

    def disassemble(self) -> dict[str, Var]:
        return (
            {
                # Maintain compatibility with existing schema
                "values": self.data.disassemble(),
            }
            | {"null": self.mask.disassemble()}
            if self.mask is not None
            else {}
        )

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

    def _pass_through_same_type(self, fun_name: str, *args, **kwargs) -> Self:
        data = getattr(self.data, fun_name)(*args, **kwargs)
        mask = (
            getattr(self.mask, fun_name)(*args, **kwargs)
            if self.mask is not None
            else None
        )
        return type(self)(data=data, mask=mask)

    def fill_null(self, value: int | float | bool | str) -> onnx.TyArray:
        value_arr = astyarray(value, use_py_scalars=True)
        if self.mask is None:
            dtype = result_type(self.data.dtype, value_arr.dtype)
            return self.data.astype(dtype)
        return safe_cast(onnx.TyArray, where(self.mask, value_arr, self.data))

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
        new_mask = _merge_masks(
            None if self.mask is None else self.mask[index], value.mask
        )
        if new_mask is None:
            return
        if self.mask is None:
            shape = self.dynamic_shape
            self.mask = safe_cast(
                onnx.TyArrayBool, astyarray(False).broadcast_to(shape)
            )
            self.mask[index] = new_mask
        else:
            self.mask[index] = new_mask

    def __ndx_cast_to__(self, dtype: DType[TY_ARRAY_BASE]) -> TY_ARRAY_BASE:
        # Implemented under the assumption that we know about `onnx`, but not py_scalars
        if isinstance(dtype, onnx._OnnxDType):
            # Not clear what the behavior should be if we have a mask
            raise NotImplementedError
        elif isinstance(dtype, _MaOnnxDType):
            new_data = self.data.astype(dtype._unmasked_dtype)
            dtype._tyarr_class(data=new_data, mask=self.mask)
        return NotImplemented

    def concat(self, others: list[Self], axis: None | int) -> Self:
        data = self.data.concat([el.data for el in others], axis)
        if all(el.mask is None for el in [self] + others):
            mask = None
        else:
            masks = []
            for el in [self] + others:
                masks.append(
                    astyarray(False).broadcast_to(self.data.dynamic_shape)
                    if el.mask is None
                    else el.mask
                )
            mask = safe_cast(onnx.TyArrayBool, masks[0].concat(masks[1:], axis))
        return safe_cast(type(self), asncoredata(data, mask))

    def _eqcomp(self, other) -> TyArrayBase | NotImplementedType:
        return _apply_op(self, other, operator.eq, True)

    def __ndx_where__(self, cond: onnx.TyArrayBool, y: TyArrayBase, /) -> TyArrayBase:
        if isinstance(y, onnx.TyArray):
            return self.__ndx_where__(cond, asncoredata(y, None))
        if isinstance(y, TyMaArray):
            x_ = unmask_core(self)
            y_ = unmask_core(y)
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

            return asncoredata(new_data, new_mask)

        return NotImplemented

    def __ndx_rwhere__(
        self, cond: onnx.TyArrayBool, x: TyArrayBase, /
    ) -> TyArrayBase | NotImplementedType:
        if isinstance(x, onnx.TyArray):
            return asncoredata(x, None).__ndx_where__(cond, self)
        return NotImplemented

    def isin(self, items: Sequence[VALUE]) -> onnx.TyArrayBool:
        data = self.data.isin(items)
        # Masked values always return False
        if self.mask is None:
            return data
        return safe_cast(onnx.TyArrayBool, data & ~self.mask)


class TyMaArrayString(TyMaArray):
    dtype = nutf8

    __add__, __radd__ = _make_binary_pair(operator.add)  # type: ignore


class TyMaArrayNumber(TyMaArray):
    dtype: NCoreNumericDTypes

    def clip(
        self, min: TyArrayBase | None = None, max: TyArrayBase | None = None
    ) -> Self:
        allowed_types = (
            py_scalars.TyArrayPyInt
            | py_scalars.TyArrayPyFloat
            | onnx.TyArray
            | TyMaArrayNumber
            | None
        )
        if not isinstance(min, allowed_types):
            raise TypeError(
                f"'clip' is not implemented for argument 'min' with data type `{min.dtype}`"
            )
        if not isinstance(max, allowed_types):
            raise TypeError(
                f"'clip' is not implemented for argument 'max' with data type `{max.dtype}`"
            )
        mask = self.mask
        if isinstance(min, TyMaArrayNumber):
            mask = _merge_masks(mask, min.mask)
            min = min.data
        if isinstance(max, TyMaArrayNumber):
            mask = _merge_masks(mask, max.mask)
            max = max.data

        data = self.data.clip(min, max)

        return type(self)(data=data, mask=mask)

    @overload
    def prod(
        self,
        /,
        *,
        dtype: DType[TY_ARRAY_BASE],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> TY_ARRAY_BASE: ...

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

    def __ndx_maximum__(self, rhs: TyArrayBase, /) -> TyArrayBase | NotImplementedType:
        if isinstance(rhs, onnx.TyArray):
            rhs = asncoredata(rhs, None)
        if isinstance(rhs, TyMaArray):
            lhs_ = unmask_core(self)
            rhs_ = unmask_core(rhs)
            new_data = safe_cast(onnx.TyArray, maximum(lhs_, rhs_))
            new_mask = _merge_masks(self.mask, rhs.mask)
            return asncoredata(new_data, new_mask)

        return NotImplemented

    def __ndx_minimum__(self, rhs: TyArrayBase, /) -> TyArrayBase | NotImplementedType:
        if isinstance(rhs, onnx.TyArray):
            rhs = asncoredata(rhs, None)
        if isinstance(rhs, TyMaArray):
            lhs_ = unmask_core(self)
            rhs_ = unmask_core(rhs)
            new_data = safe_cast(onnx.TyArray, minimum(lhs_, rhs_))
            new_mask = _merge_masks(self.mask, rhs.mask)
            return asncoredata(new_data, new_mask)

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
    isfinite = _make_unary_member_same_type("isfinite")  # type: ignore
    isinf = _make_unary_member_same_type("isinf")  # type: ignore
    isnan = _make_unary_member_same_type("isnan")  # type: ignore
    round = _make_unary_member_same_type("round")  # type: ignore
    sign = _make_unary_member_same_type("sign")  # type: ignore
    sqrt = _make_unary_member_same_type("sqrt")  # type: ignore
    square = _make_unary_member_same_type("square")  # type: ignore
    trunc = _make_unary_member_same_type("trunc")  # type: ignore


class TyMaArrayInteger(TyMaArrayNumber):
    dtype: NCoreIntegerDTypes

    __and__, __rand__ = _make_binary_pair(operator.and_)  # type: ignore
    __or__, __ror__ = _make_binary_pair(operator.or_)  # type: ignore
    __xor__, __rxor__ = _make_binary_pair(operator.xor)  # type: ignore
    __lshift__, __rlshift__ = _make_binary_pair(operator.lshift)  # type: ignore
    __rshift__, __rrshift__ = _make_binary_pair(operator.rshift)  # type: ignore
    __invert__ = _make_unary_member_same_type("__invert__")  # type: ignore


class TyMaArrayFloating(TyMaArrayNumber):
    dtype: NCoreFloatingDTypes

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
    dtype = nbool

    __invert__ = _make_unary_member_same_type("__invert__")  # type: ignore

    __and__, __rand__ = _make_binary_pair(operator.and_)  # type: ignore
    __or__, __ror__ = _make_binary_pair(operator.or_)  # type: ignore
    __xor__, __rxor__ = _make_binary_pair(operator.xor)  # type: ignore
    __lshift__, __rlshift__ = _make_binary_pair(operator.lshift)  # type: ignore
    __rshift__, __rrshift__ = _make_binary_pair(operator.rshift)  # type: ignore


class TyMaArrayInt8(TyMaArrayInteger):
    dtype = nint8


class TyMaArrayInt16(TyMaArrayInteger):
    dtype = nint16


class TyMaArrayInt32(TyMaArrayInteger):
    dtype = nint32


class TyMaArrayInt64(TyMaArrayInteger):
    dtype = nint64


class TyMaArrayUInt8(TyMaArrayInteger):
    dtype = nuint8


class TyMaArrayUInt16(TyMaArrayInteger):
    dtype = nuint16


class TyMaArrayUInt32(TyMaArrayInteger):
    dtype = nuint32


class TyMaArrayUInt64(TyMaArrayInteger):
    dtype = nuint64


class TyMaArrayFloat16(TyMaArrayFloating):
    dtype = nfloat16


class TyMaArrayFloat32(TyMaArrayFloating):
    dtype = nfloat32


class TyMaArrayFloat64(TyMaArrayFloating):
    dtype = nfloat64


def _merge_masks(
    a: onnx.TyArrayBool | None, b: onnx.TyArrayBool | None
) -> onnx.TyArrayBool | None:
    if a is None:
        return b
    if b is None:
        return a
    out = a | b
    if isinstance(out, onnx.TyArrayBool):
        return out
    # Should never happen
    raise TypeError("Unexpected array type")


def asncoredata(core_array: onnx.TyArray, mask: onnx.TyArrayBool | None) -> TyMaArray:
    return as_nullable(core_array.dtype)._tyarr_class(core_array, mask)


def unmask_core(arr: onnx.TyArray | TyMaArray) -> onnx.TyArray:
    if isinstance(arr, onnx.TyArray):
        return arr
    return arr.data


def _apply_op(
    this: TyMaArray,
    other: TyArrayBase,
    op: Callable[[onnx.TyArray, onnx.TyArray], onnx.TyArray],
    forward: bool,
) -> TyMaArray:
    """Apply an operation by passing it through to the data member."""
    if isinstance(other, onnx.TyArray):
        data = op(this.data, other) if forward else op(other, this.data)
        mask = this.mask
    elif isinstance(other, TyMaArray):
        data = op(this.data, other.data) if forward else op(other.data, this.data)
        mask = _merge_masks(this.mask, other.mask)
    else:
        return NotImplemented

    return asncoredata(data, mask)


####################
# Conversion Table #
####################

_core_to_nullable_core: dict[onnx._OnnxDType, NCoreDTypes] = {
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


def as_nullable(dtype: onnx._OnnxDType) -> NCoreDTypes:
    return _core_to_nullable_core[dtype]


def as_non_nullable(dtype: _MaOnnxDType) -> onnx._OnnxDType:
    mapping: dict[_MaOnnxDType, onnx._OnnxDType] = {
        v: k for k, v in _core_to_nullable_core.items()
    }
    return mapping[dtype]
