# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import operator
from collections.abc import Callable
from types import NotImplementedType
from typing import TYPE_CHECKING, TypeVar

from typing_extensions import Self

from .. import dtypes
from ..dtypes import CoreDTypes, DType, NCoreDTypes
from ..schema import Schema, flatten_components
from .core import TyArray, TyArrayBool, TyArrayInt64
from .typed_array import TyArrayBase

if TYPE_CHECKING:
    from ..array import OnnxShape
    from ..schema import Components, StructComponent
    from .indexing import StaticGetIndex


DTYPE = TypeVar("DTYPE", bound=DType)
CORE_DTYPES = TypeVar("CORE_DTYPES", bound=CoreDTypes)
NCORE_DTYPES = TypeVar("NCORE_DTYPES", bound=NCoreDTypes)

NCORE_NUMERIC_DTYPES = TypeVar("NCORE_NUMERIC_DTYPES", bound=dtypes.NCoreNumericDTypes)
NCORE_FLOATING_DTYPES = TypeVar(
    "NCORE_FLOATING_DTYPES", bound=dtypes.NCoreFloatingDTypes
)
NCORE_INTEGER_DTYPES = TypeVar("NCORE_INTEGER_DTYPES", bound=dtypes.NCoreIntegerDTypes)

ALL_NUM_DTYPES = TypeVar(
    "ALL_NUM_DTYPES", bound=dtypes.CoreNumericDTypes | dtypes.NCoreNumericDTypes
)


class TyMaArrayBase(TyArrayBase):
    """Typed masked array object."""

    data: TyArrayBase
    mask: TyArrayBool | None

    def disassemble(self) -> tuple[Components, Schema]:
        dtype_info = self.dtype._info
        component_schema: StructComponent = (
            {
                "data": self.data.disassemble()[1],
            }
            | {"mask": self.mask.disassemble()[1]}
            if self.mask is not None
            else {}
        )
        schema = Schema(dtype_info=dtype_info, components=component_schema)
        components = flatten_components(
            {
                "data": self.data.disassemble()[0],
            }
            | {"mask": self.mask.disassemble()[0]}
            if self.mask is not None
            else {}
        )
        return components, schema


class TyMaArray(TyMaArrayBase):
    """Masked version of core types.

    The (subclasses) of this class are implemented such that they only
    know about `_ArrayCoreType`s, but not `_PyScalar`s.
    """

    dtype: dtypes.NCoreDTypes
    # Specialization of data from `Data` to `_ArrayCoreType`
    data: TyArray

    def __init__(self, data: TyArray, mask: TyArrayBool | None):
        self.dtype = dtypes.as_nullable(data.dtype)
        self.data = data
        self.mask = mask

    @property
    def shape(self) -> OnnxShape:
        return self.data.shape

    @property
    def dynamic_shape(self) -> TyArrayInt64:
        return self.data.dynamic_shape

    def reshape(self, shape: tuple[int, ...]) -> Self:
        data = self.data.reshape(shape)
        mask = self.mask.reshape(shape) if self.mask is not None else None
        return type(self)(data=data, mask=mask)

    def broadcast_to(self, shape: tuple[int, ...] | TyArrayInt64) -> Self:
        data = self.data.broadcast_to(shape)
        mask = self.mask.broadcast_to(shape) if self.mask else None
        return type(self)(data=data, mask=mask)

    def __getitem__(self, index: StaticGetIndex | TyArrayInt64) -> Self:
        new_data = self.data[index]
        new_mask = self.mask[index] if self.mask is not None else None

        return type(self)(data=new_data, mask=new_mask)

    def _astype(self, dtype: DType) -> TyArrayBase:
        # Implemented under the assumption that we know about core, but not py_scalars
        if isinstance(dtype, dtypes.CoreDTypes):
            # TODO: Not clear what the behavior should be if we have a mask
            # TODO: There is currently no way to get the mask through the public `Array` class!
            raise NotImplementedError
        elif isinstance(dtype, dtypes.NCoreDTypes):
            new_data = self.data.astype(dtype._unmasked_dtype)
            dtype._tyarr_class(data=new_data, mask=self.mask)
        return NotImplemented

    def _where(
        self, cond: TyArrayBool, y: TyArrayBase
    ) -> TyArrayBase | NotImplementedType:
        if isinstance(y, TyArray):
            return self._where(cond, asncoredata(y, None))
        if isinstance(y, TyMaArray):
            x_ = unmask_core(self)
            y_ = unmask_core(y)
            new_data = x_._where(cond, y_)
            if self.mask is not None and y.mask is not None:
                new_mask = cond & self.mask | ~cond & y.mask
            elif self.mask is not None:
                new_mask = cond & self.mask
            elif y.mask is not None:
                new_mask = ~cond & y.mask
            else:
                new_mask = None

            if new_mask is not None and not isinstance(new_mask, TyArrayBool):
                # Should never happen. Might be worth while adding
                # overloads to the BoolData dunder methods to
                # propagate the types more precisely.
                raise TypeError(f"expected boolean mask, found `{new_mask.dtype}`")

            return asncoredata(new_data, new_mask)

        return NotImplemented

    def _rwhere(
        self, cond: TyArrayBool, x: TyArrayBase
    ) -> TyArrayBase | NotImplementedType:
        if isinstance(x, TyArray):
            return asncoredata(x, None)._where(cond, self)
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

    def _eqcomp(self, other: TyArrayBase) -> TyArrayBase | NotImplementedType:
        raise NotImplementedError()
        ...


class TyMaArrayNumber(TyMaArray):
    dtype: dtypes.NCoreNumericDTypes


class TyMaArrayInteger(TyMaArrayNumber):
    dtype: dtypes.NCoreIntegerDTypes


class TyMaArrayFloating(TyMaArrayNumber):
    dtype: dtypes.NCoreFloatingDTypes


class TyMaArrayBool(TyMaArray):
    dtype = dtypes.nbool


class TyMaArrayInt8(TyMaArrayInteger):
    dtype = dtypes.nint8


class TyMaArrayInt16(TyMaArrayInteger):
    dtype = dtypes.nint16


class TyMaArrayInt32(TyMaArrayInteger):
    dtype = dtypes.nint32


class TyMaArrayInt64(TyMaArrayInteger):
    dtype = dtypes.nint64


class TyMaArrayUint8(TyMaArrayInteger):
    dtype = dtypes.nuint8


class TyMaArrayUint16(TyMaArrayInteger):
    dtype = dtypes.nuint16


class TyMaArrayUint32(TyMaArrayInteger):
    dtype = dtypes.nuint32


class TyMaArrayUint64(TyMaArrayInteger):
    dtype = dtypes.nuint64


class TyMaArrayFloat16(TyMaArrayFloating):
    dtype = dtypes.nfloat16


class TyMaArrayFloat32(TyMaArrayFloating):
    dtype = dtypes.nfloat32


class TyMaArrayFloat64(TyMaArrayFloating):
    dtype = dtypes.nfloat64


def _merge_masks(a: TyArrayBool | None, b: TyArrayBool | None) -> TyArrayBool | None:
    if a is None:
        return b
    if b is None:
        return a
    out = a | b
    if isinstance(out, TyArrayBool):
        return out
    # Should never happen
    raise TypeError("Unexpected array type")


def asncoredata(core_array: TyArrayBase, mask: TyArrayBool | None) -> TyMaArray:
    from . import core

    if isinstance(core_array, core.TyArrayInt8):
        return TyMaArrayInt8(core_array, mask)
    if isinstance(core_array, core.TyArrayInt16):
        return TyMaArrayInt16(core_array, mask)
    if isinstance(core_array, core.TyArrayInt32):
        return TyMaArrayInt32(core_array, mask)
    if isinstance(core_array, core.TyArrayInt64):
        return TyMaArrayInt64(core_array, mask)

    if isinstance(core_array, core.TyArrayUint8):
        return TyMaArrayUint8(core_array, mask)
    if isinstance(core_array, core.TyArrayUint16):
        return TyMaArrayUint16(core_array, mask)
    if isinstance(core_array, core.TyArrayUint32):
        return TyMaArrayUint32(core_array, mask)
    if isinstance(core_array, core.TyArrayUint64):
        return TyMaArrayUint64(core_array, mask)

    if isinstance(core_array, core.Float16Data):
        return TyMaArrayFloat16(core_array, mask)
    if isinstance(core_array, core.Float32Data):
        return TyMaArrayFloat32(core_array, mask)
    if isinstance(core_array, core.Float64Data):
        return TyMaArrayFloat64(core_array, mask)

    if isinstance(core_array, core.TyArrayBool):
        return TyMaArrayBool(core_array, mask)

    raise TypeError("expected '_ArrayCoreType' found `{type(core_array)}`")


def unmask_core(arr: TyArray | TyMaArray) -> TyArray:
    if isinstance(arr, TyArray):
        return arr
    return arr.data


def _apply_op(
    this: TyMaArray,
    other: TyArrayBase,
    op: Callable[[TyArray, TyArray], TyArray],
    forward: bool,
) -> TyMaArray:
    """Apply an operation by passing it through to the data member."""
    if isinstance(other, TyArray):
        data = op(this.data, other) if forward else op(other, this.data)
        mask = this.mask
    elif isinstance(other, TyMaArray):
        data = op(this.data, other.data) if forward else op(other.data, this.data)
        mask = _merge_masks(this.mask, other.mask)
    else:
        return NotImplemented

    return asncoredata(data, mask)
