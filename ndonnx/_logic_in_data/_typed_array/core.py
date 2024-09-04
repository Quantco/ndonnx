# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import NotImplementedType
from typing import TYPE_CHECKING, TypeGuard, TypeVar

import numpy as np
import spox.opset.ai.onnx.v21 as op
from spox import Tensor, Var, argument
from typing_extensions import Self

from .. import dtypes
from ..dtypes import (
    CoreDTypes,
    DType,
    from_numpy,
)
from .typed_array import TyArrayBase
from .utils import promote

if TYPE_CHECKING:
    from ..array import Index, OnnxShape


CORE_DTYPES = TypeVar("CORE_DTYPES", bound=CoreDTypes)
ALL_NUM_DTYPES = TypeVar(
    "ALL_NUM_DTYPES", bound=dtypes.CoreNumericDTypes | dtypes.NCoreNumericDTypes
)


class TyArray(TyArrayBase[CORE_DTYPES]):
    var: Var

    def __init__(self, var: Var):
        if from_numpy(var.unwrap_tensor().dtype) != self.dtype:
            raise ValueError(f"data type of 'var' is incompatible with `{type(self)}`")
        self.var = var

    @classmethod
    def from_typed_array(cls, tyarr: TyArrayBase):
        if isinstance(tyarr, TyArray):
            var = op.cast(tyarr.var, to=dtypes.as_numpy(cls.dtype))
            return cls(var)
        raise NotImplementedError

    @classmethod
    def as_argument(cls, shape: OnnxShape, dtype: DType):
        if isinstance(dtype, CoreDTypes):
            var = argument(Tensor(dtypes.as_numpy(dtype), shape))
            return cls(var)
        raise ValueError("unexpected 'dtype' `{dtype}`")

    def __getitem__(self, index: Index) -> Self:
        if isinstance(index, int):
            var = op.slice(
                self.var,
                starts=op.const([index]),
                ends=op.const([index + 1]),
                axes=op.const([0]),
            )
            var = op.squeeze(var, axes=op.const(0))
            return type(self)(var)
        raise NotImplementedError

    @property
    def shape(self) -> OnnxShape:
        shape = self.var.unwrap_tensor().shape
        if shape is None:
            raise ValueError("Missing shape information")
        return shape

    def to_numpy(self) -> np.ndarray:
        if self.var._value is not None:
            np_arr = np.asarray(self.var._value.value)
            if np_arr.dtype != dtypes.as_numpy(self.dtype):
                # Should not happen
                raise ValueError("unexpected value data type")
            return np_arr
        raise ValueError("no propagated value available")

    def reshape(self, shape: tuple[int, ...]) -> Self:
        var = op.reshape(self.var, op.const(shape))
        return type(self)(var)

    def as_core_dtype(self, dtype: CoreDTypes) -> TyArray:
        raise ValueError(f"Casting between `{self.dtype}` and `{dtype}` is undefined")

    def _astype(self, dtype: DType) -> TyArrayBase:
        return NotImplemented

    def _where(
        self, cond: TyArrayBool, y: TyArrayBase
    ) -> TyArrayBase | NotImplementedType:
        if isinstance(y, TyArray):
            x, y = promote(self, y)
            var = op.where(cond.var, x.var, y.var)
            return type(x)(var)

        return NotImplemented


class TyArrayNumber(TyArray[CORE_DTYPES]):
    def __add__(self, rhs: TyArrayBase) -> TyArrayBase:
        if isinstance(rhs, TyArrayNumber):
            # NOTE: Can't always promote for all data types (c.f. datetime / timedelta)
            if type(self) != type(rhs):
                a, b = promote(self, rhs)
                return a + b
            var = op.add(self.var, rhs.var)
            return ascoredata(var)
        return NotImplemented

    def __sub__(self, rhs: TyArrayBase) -> TyArrayBase:
        if isinstance(rhs, TyArrayNumber):
            # NOTE: Can't always promote for all data types (c.f. datetime / timedelta)
            if type(self) != type(rhs):
                a, b = promote(self, rhs)
                return a - b
            var = op.add(self.var, rhs.var)
            return ascoredata(var)
        return NotImplemented


class TyArrayInteger(TyArrayNumber[CORE_DTYPES]):
    def __or__(self, rhs: TyArrayBase) -> TyArrayBase:
        if isinstance(rhs, TyArray):
            if self.dtype != rhs.dtype:
                a, b = promote(self, rhs)
                return a | b

            # Data is core & integer
            var = op.bitwise_or(self.var, rhs.var)
            return ascoredata(var)
        return NotImplemented


class TyArrayFloating(TyArrayNumber[CORE_DTYPES]): ...


class TyArrayBool(TyArray[dtypes.Bool]):
    dtype = dtypes.bool_

    def __or__(self, rhs: TyArrayBase) -> TyArrayBase:
        if self.dtype != rhs.dtype:
            a, b = promote(self, rhs)
            return a | b

        if isinstance(rhs, TyArrayBool):
            var = op.or_(self.var, rhs.var)
            return ascoredata(var)
        return NotImplemented

    def __and__(self, rhs: TyArrayBase) -> TyArrayBase:
        if self.dtype != rhs.dtype:
            a, b = promote(self, rhs)
            return a & b

        if isinstance(rhs, TyArrayBool):
            var = op.and_(self.var, rhs.var)
            return ascoredata(var)
        return NotImplemented

    def __invert__(self) -> TyArrayBool:
        var = op.not_(self.var)
        return type(self)(var)


class TyArrayInt8(TyArrayInteger[dtypes.Int8]):
    dtype = dtypes.int8


class TyArrayInt16(TyArrayInteger[dtypes.Int16]):
    dtype = dtypes.int16


class TyArrayInt32(TyArrayInteger[dtypes.Int32]):
    dtype = dtypes.int32


class TyArrayInt64(TyArrayInteger[dtypes.Int64]):
    dtype = dtypes.int64


class TyArrayUint8(TyArrayInteger[dtypes.Uint8]):
    dtype = dtypes.uint8


class TyArrayUint16(TyArrayInteger[dtypes.Uint16]):
    dtype = dtypes.uint16


class TyArrayUint32(TyArrayInteger[dtypes.Uint32]):
    dtype = dtypes.uint32


class TyArrayUint64(TyArrayInteger[dtypes.Uint64]):
    dtype = dtypes.uint64


class Float16Data(TyArrayFloating[dtypes.Float16]):
    dtype = dtypes.float16


class Float32Data(TyArrayFloating[dtypes.Float32]):
    dtype = dtypes.float32


class Float64Data(TyArrayFloating[dtypes.Float64]):
    dtype = dtypes.float64


def ascoredata(var: Var) -> TyArray:
    dtype = from_numpy(var.unwrap_tensor().dtype)

    return dtype._tyarr_class(var)


def is_sequence_of_core_data(
    seq: Sequence[TyArrayBase],
) -> TypeGuard[Sequence[TyArray]]:
    return all(isinstance(d, TyArray) for d in seq)


def _promote_and_apply_op(
    lhs: TyArray,
    rhs: TyArrayBase,
    arr_op: Callable[[TyArray, TyArray], TyArray],
    spox_op: Callable[[Var, Var], Var],
) -> TyArray:
    """Promote and apply an operation by passing it through to the data member."""
    if isinstance(rhs, TyArray):
        if lhs.dtype != rhs.dtype:
            a, b = promote(lhs, rhs)
            return arr_op(a, b)

        # Data is core & integer
        var = spox_op(lhs.var, rhs.var)
        return ascoredata(var)
    return NotImplemented
