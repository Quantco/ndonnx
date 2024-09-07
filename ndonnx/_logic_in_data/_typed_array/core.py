# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import operator
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
from ..schema import Schema, var_to_primitive
from .typed_array import TyArrayBase
from .utils import promote

if TYPE_CHECKING:
    from ..array import Index, OnnxShape
    from ..schema import Components


CORE_DTYPES = TypeVar("CORE_DTYPES", bound=CoreDTypes)
ALL_NUM_DTYPES = TypeVar(
    "ALL_NUM_DTYPES", bound=dtypes.CoreNumericDTypes | dtypes.NCoreNumericDTypes
)


class TyArray(TyArrayBase):
    dtype: dtypes.CoreDTypes
    var: Var

    def __init__(self, var: Var):
        if from_numpy(var.unwrap_tensor().dtype) != self.dtype:
            raise ValueError(f"data type of 'var' is incompatible with `{type(self)}`")
        self.var = var

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

    def disassemble(self) -> tuple[Components, Schema]:
        dtype_info = self.dtype._info
        component_schema = var_to_primitive(self.var)
        schema = Schema(dtype_info=dtype_info, components=component_schema)
        components = {"var": self.var}
        return components, schema

    def reshape(self, shape: tuple[int, ...]) -> Self:
        var = op.reshape(self.var, op.const(shape))
        return type(self)(var)

    def as_core_dtype(self, dtype: CoreDTypes) -> TyArray:
        raise ValueError(f"Casting between `{self.dtype}` and `{dtype}` is undefined")

    def _astype(self, dtype: DType) -> TyArrayBase:
        return NotImplemented

    def _eqcomp(self, other) -> TyArrayBase:
        return _promote_and_apply_op(self, other, operator.eq, op.equal, True)

    def _where(
        self, cond: TyArrayBool, y: TyArrayBase
    ) -> TyArrayBase | NotImplementedType:
        if isinstance(y, TyArray):
            x, y = promote(self, y)
            var = op.where(cond.var, x.var, y.var)
            return type(x)(var)

        return NotImplemented


class TyArrayNumber(TyArray):
    dtype: dtypes.CoreNumericDTypes

    def __add__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.add, op.add, True)

    def __radd__(self, lhs) -> TyArrayBase:
        return _promote_and_apply_op(self, lhs, operator.add, op.add, False)

    def __mul__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.mul, op.mul, True)

    def __rmul__(self, lhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, lhs, operator.mul, op.mul, False)

    def __sub__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.sub, op.sub, True)

    def __rsub__(self, lhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, lhs, operator.sub, op.sub, False)


class TyArrayInteger(TyArrayNumber):
    dtype: dtypes.CoreIntegerDTypes

    def __or__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.or_, op.bitwise_or, True)


class TyArrayFloating(TyArrayNumber):
    dtype: dtypes.CoreFloatingDTypes


class TyArrayBool(TyArray):
    dtype = dtypes.bool_

    def __or__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.or_, op.or_, True)

    def __and__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.and_, op.and_, True)

    def __invert__(self) -> TyArrayBool:
        var = op.not_(self.var)
        return type(self)(var)


class TyArrayInt8(TyArrayInteger):
    dtype = dtypes.int8


class TyArrayInt16(TyArrayInteger):
    dtype = dtypes.int16


class TyArrayInt32(TyArrayInteger):
    dtype = dtypes.int32


class TyArrayInt64(TyArrayInteger):
    dtype = dtypes.int64


class TyArrayUint8(TyArrayInteger):
    dtype = dtypes.uint8


class TyArrayUint16(TyArrayInteger):
    dtype = dtypes.uint16


class TyArrayUint32(TyArrayInteger):
    dtype = dtypes.uint32


class TyArrayUint64(TyArrayInteger):
    dtype = dtypes.uint64


class Float16Data(TyArrayFloating):
    dtype = dtypes.float16


class Float32Data(TyArrayFloating):
    dtype = dtypes.float32


class Float64Data(TyArrayFloating):
    dtype = dtypes.float64


def ascoredata(var: Var) -> TyArray:
    dtype = from_numpy(var.unwrap_tensor().dtype)

    return dtype._tyarr_class(var)


def is_sequence_of_core_data(
    seq: Sequence[TyArrayBase],
) -> TypeGuard[Sequence[TyArray]]:
    return all(isinstance(d, TyArray) for d in seq)


def _promote_and_apply_op(
    this: TyArray,
    other: TyArrayBase,
    arr_op: Callable[[TyArray, TyArray], TyArray],
    spox_op: Callable[[Var, Var], Var],
    forward: bool,
) -> TyArray:
    """Promote and apply an operation by passing it through to the data member."""
    if isinstance(other, TyArray):
        if this.dtype != other.dtype:
            a, b = promote(this, other)
            return arr_op(a, b) if forward else arr_op(b, a)

        # Data is core & integer
        var = spox_op(this.var, other.var) if forward else spox_op(other.var, this.var)
        return ascoredata(var)
    return NotImplemented
