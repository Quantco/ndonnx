# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import operator
from collections.abc import Callable
from types import NotImplementedType
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from typing_extensions import Self

from .._dtypes import TY_ARRAY, DType
from .._schema import DTypeInfoV1
from . import onnx
from .funcs import astyarray, result_type, where
from .typed_array import TyArrayBase
from .utils import safe_cast

if TYPE_CHECKING:
    from spox import Var

    from .._array import OnnxShape
    from .indexing import GetitemIndex, SetitemIndex


DTYPE = TypeVar("DTYPE", bound=DType)
CORE_DTYPES = TypeVar("CORE_DTYPES", bound=onnx.DTypes)
NCORE_DTYPES = TypeVar("NCORE_DTYPES", bound="NCoreDTypes")

NCORE_NUMERIC_DTYPES = TypeVar("NCORE_NUMERIC_DTYPES", bound="NCoreNumericDTypes")
NCORE_FLOATING_DTYPES = TypeVar("NCORE_FLOATING_DTYPES", bound="NCoreFloatingDTypes")
NCORE_INTEGER_DTYPES = TypeVar("NCORE_INTEGER_DTYPES", bound="NCoreIntegerDTypes")

TY_MA_ARRAY_ONNX = TypeVar("TY_MA_ARRAY_ONNX", bound="TyMaArray")


class _MaOnnxDType(DType[TY_MA_ARRAY_ONNX]):
    _unmasked_dtype: onnx.DTypes

    def __ndx_convert_tyarray__(self, arr: TyArrayBase) -> TY_MA_ARRAY_ONNX:
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


class _NNumber(_MaOnnxDType):
    _unmasked_dtype: onnx.NumericDTypes

    def _result_type(self, rhs: DType) -> DType | NotImplementedType:
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


class NString(_MaOnnxDType):
    _unmasked_dtype = onnx.string

    def _result_type(self, rhs: DType) -> DType | NotImplementedType:
        return NotImplemented

    @property
    def _tyarr_class(self) -> type[TyMaArrayString]:
        return TyMaArrayString


class NBoolean(_MaOnnxDType):
    _unmasked_dtype = onnx.bool_

    def _result_type(self, rhs: DType) -> DType | NotImplementedType:
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


class NUint8(_NNumber):
    _unmasked_dtype = onnx.uint8

    @property
    def _tyarr_class(self) -> type[TyMaArrayUint8]:
        return TyMaArrayUint8


class NUint16(_NNumber):
    _unmasked_dtype = onnx.uint16

    @property
    def _tyarr_class(self) -> type[TyMaArrayUint16]:
        return TyMaArrayUint16


class NUint32(_NNumber):
    _unmasked_dtype = onnx.uint32

    @property
    def _tyarr_class(self) -> type[TyMaArrayUint32]:
        return TyMaArrayUint32


class NUint64(_NNumber):
    _unmasked_dtype = onnx.uint64

    @property
    def _tyarr_class(self) -> type[TyMaArrayUint64]:
        return TyMaArrayUint64


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

nuint8 = NUint8()
nuint16 = NUint16()
nuint32 = NUint32()
nuint64 = NUint64()

nstring = NString()

# Union types
#
# Union types are exhaustive and don't create ambiguities with respect to user-defined subtypes.

NCoreIntegerDTypes = (
    NInt8 | NInt16 | NInt32 | NInt64 | NUint8 | NUint16 | NUint32 | NUint64
)
NCoreFloatingDTypes = NFloat16 | NFloat32 | NFloat64

NCoreNumericDTypes = NCoreFloatingDTypes | NCoreIntegerDTypes

NCoreDTypes = NBoolean | NCoreNumericDTypes | NString


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

    dtype: NCoreDTypes
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

    def fill_null(self, value: int | float | bool | str) -> onnx.TyArray:
        value_arr = astyarray(value, use_py_scalars=True)
        if self.mask is None:
            dtype = result_type(self.data.dtype, value_arr.dtype)
            return self.data.astype(dtype)
        return safe_cast(onnx.TyArray, where(self.mask, value_arr, self.data))

    def reshape(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> Self:
        data = self.data.reshape(shape)
        mask = self.mask.reshape(shape) if self.mask is not None else None
        return type(self)(data=data, mask=mask)

    def broadcast_to(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> Self:
        data = self.data.broadcast_to(shape)
        mask = self.mask.broadcast_to(shape) if self.mask else None
        return type(self)(data=data, mask=mask)

    def unwrap_numpy(self) -> np.ndarray:
        return np.ma.MaskedArray(
            data=self.data.unwrap_numpy(),
            mask=None if self.mask is None else self.mask.unwrap_numpy(),
        )

    def __getitem__(self, index: GetitemIndex) -> Self:
        new_data = self.data[index]
        new_mask = self.mask[index] if self.mask is not None else None

        return type(self)(data=new_data, mask=new_mask)

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

    def __ndx_astype__(self, dtype: DType[TY_ARRAY]) -> TY_ARRAY:
        # Implemented under the assumption that we know about core, but not py_scalars
        if isinstance(dtype, onnx.DTypes):
            # TODO: Not clear what the behavior should be if we have a mask
            # TODO: There is currently no way to get the mask through the public `Array` class!
            raise NotImplementedError
        elif isinstance(dtype, NCoreDTypes):
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

    def _eqcomp(self, other: TyArrayBase) -> TyArrayBase | NotImplementedType:
        raise NotImplementedError()

    def __ndx_where__(
        self, cond: onnx.TyArrayBool, y: TyArrayBase, /
    ) -> TyArrayBase | NotImplementedType:
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


class TyMaArrayString(TyMaArray):
    dtype = nstring

    def __add__(self, rhs: TyArrayBase) -> TyMaArray:
        return _apply_op(self, rhs, operator.add, True)

    def __radd__(self, lhs: TyArrayBase) -> TyMaArray:
        return _apply_op(self, lhs, operator.add, False)


class TyMaArrayNumber(TyMaArray):
    dtype: NCoreNumericDTypes

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
            return self.__ndx_maximum__(asncoredata(rhs, None))
        if isinstance(rhs, TyMaArray):
            lhs_ = unmask_core(self)
            rhs_ = unmask_core(rhs)
            new_data = lhs_.__ndx_maximum__(rhs_)
            new_mask = _merge_masks(self.mask, rhs.mask)
            return asncoredata(new_data, new_mask)

        return NotImplemented

    def __ndx_rmaximum__(self, lhs: TyArrayBase, /) -> TyArrayBase | NotImplementedType:
        if isinstance(lhs, onnx.TyArray):
            return asncoredata(lhs, None).__ndx_maximum__(self)
        return NotImplemented

    # Dunder implementations
    #
    # We don't differentiate between the different subclasses since we
    # always just pass through to the underlying non-masked typed. We
    # will get an error from there if appropriate.

    def __add__(self, rhs: TyArrayBase) -> TyMaArray:
        return _apply_op(self, rhs, operator.add, True)

    def __radd__(self, lhs: TyArrayBase) -> TyMaArray:
        return _apply_op(self, lhs, operator.add, False)

    def __sub__(self, rhs: TyArrayBase) -> TyMaArray:
        return _apply_op(self, rhs, operator.sub, True)

    def __rsub__(self, lhs: TyArrayBase) -> TyMaArray:
        return _apply_op(self, lhs, operator.sub, False)

    def __mul__(self, rhs: TyArrayBase) -> TyMaArray:
        return _apply_op(self, rhs, operator.mul, True)

    def __rmul__(self, lhs: TyArrayBase) -> TyMaArray:
        return _apply_op(self, lhs, operator.mul, False)


class TyMaArrayInteger(TyMaArrayNumber):
    dtype: NCoreIntegerDTypes


class TyMaArrayFloating(TyMaArrayNumber):
    dtype: NCoreFloatingDTypes


class TyMaArrayBool(TyMaArray):
    dtype = nbool


class TyMaArrayInt8(TyMaArrayInteger):
    dtype = nint8


class TyMaArrayInt16(TyMaArrayInteger):
    dtype = nint16


class TyMaArrayInt32(TyMaArrayInteger):
    dtype = nint32


class TyMaArrayInt64(TyMaArrayInteger):
    dtype = nint64


class TyMaArrayUint8(TyMaArrayInteger):
    dtype = nuint8


class TyMaArrayUint16(TyMaArrayInteger):
    dtype = nuint16


class TyMaArrayUint32(TyMaArrayInteger):
    dtype = nuint32


class TyMaArrayUint64(TyMaArrayInteger):
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

_core_to_nullable_core: dict[onnx.DTypes, NCoreDTypes] = {
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
    onnx.string: nstring,
}


def as_nullable(dtype: onnx.DTypes) -> NCoreDTypes:
    return _core_to_nullable_core[dtype]


def as_non_nullable(dtype: _MaOnnxDType) -> onnx.DTypes:
    mapping: dict[_MaOnnxDType, onnx.DTypes] = {
        v: k for k, v in _core_to_nullable_core.items()
    }
    return mapping[dtype]
