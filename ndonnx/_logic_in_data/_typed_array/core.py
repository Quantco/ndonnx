# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import NotImplementedType
from typing import TYPE_CHECKING, Any, TypeGuard, TypeVar

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
from .typed_array import _TypedArray
from .utils import promote

if TYPE_CHECKING:
    from ..array import Index, OnnxShape


CORE_DTYPES = TypeVar("CORE_DTYPES", bound=CoreDTypes)
ALL_NUM_DTYPES = TypeVar(
    "ALL_NUM_DTYPES", bound=dtypes.CoreNumericDTypes | dtypes.NCoreNumericDTypes
)


class _ArrayCoreType(_TypedArray[CORE_DTYPES]):
    var: Var

    def __init__(self, var: Var):
        if from_numpy(var.unwrap_tensor().dtype) != self.dtype:
            raise ValueError(f"data type of 'var' is incompatible with `{type(self)}`")
        self.var = var

    @classmethod
    def from_typed_array(cls, tyarr: _TypedArray):
        if isinstance(tyarr, _ArrayCoreType):
            var = op.cast(tyarr.var, to=dtypes.as_numpy(cls.dtype))
            return cls(var)
        raise NotImplementedError

    @classmethod
    def as_argument(cls, shape: OnnxShape):
        var = argument(Tensor(dtypes.as_numpy(cls.dtype), shape))
        return cls(var)

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

    @classmethod
    def from_np_schema(cls, schema: dict[str, Any], /) -> _ArrayCoreType:
        if "var" in schema and len(schema) == 1:
            (var,) = schema.values()
            return ascoredata(op.const(var))
        raise ValueError("'schema' has unexpected layout")

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

    def as_core_dtype(self, dtype: CoreDTypes) -> _ArrayCoreType:
        raise ValueError(f"Casting between `{self.dtype}` and `{dtype}` is undefined")

    def _astype(self, dtype: DType) -> _TypedArray:
        return NotImplemented

    def _where(
        self, cond: BoolData, y: _TypedArray
    ) -> _TypedArray | NotImplementedType:
        if isinstance(y, _ArrayCoreType):
            x, y = promote(self, y)
            var = op.where(cond.var, x.var, y.var)
            return type(x)(var)

        return NotImplemented


class _ArrayCoreNum(_ArrayCoreType[CORE_DTYPES]):
    def __add__(self, rhs: _TypedArray) -> _TypedArray:
        if isinstance(rhs, _ArrayCoreNum):
            # NOTE: Can't always promote for all data types (c.f. datetime / timedelta)
            if type(self) != type(rhs):
                a, b = promote(self, rhs)
                return a + b
            var = op.add(self.var, rhs.var)
            return ascoredata(var)
        return NotImplemented


class _ArrayCoreInteger(_ArrayCoreNum[CORE_DTYPES]):
    def __or__(self, rhs: _TypedArray) -> _TypedArray:
        if isinstance(rhs, _ArrayCoreType):
            if self.dtype != rhs.dtype:
                a, b = promote(self, rhs)
                return a | b

            # Data is core & integer
            var = op.bitwise_or(self.var, rhs.var)
            return ascoredata(var)
        return NotImplemented


class _ArrayCoreFloating(_ArrayCoreNum[CORE_DTYPES]): ...


class BoolData(_ArrayCoreType[dtypes.Bool]):
    dtype = dtypes.bool_

    def __or__(self, rhs: _TypedArray) -> _TypedArray:
        if self.dtype != rhs.dtype:
            a, b = promote(self, rhs)
            return a | b

        if isinstance(rhs, BoolData):
            var = op.or_(self.var, rhs.var)
            return ascoredata(var)
        return NotImplemented

    def __and__(self, rhs: _TypedArray) -> _TypedArray:
        if self.dtype != rhs.dtype:
            a, b = promote(self, rhs)
            return a & b

        if isinstance(rhs, BoolData):
            var = op.and_(self.var, rhs.var)
            return ascoredata(var)
        return NotImplemented

    def __invert__(self) -> BoolData:
        var = op.not_(self.var)
        return type(self)(var)


class Int8Data(_ArrayCoreInteger[dtypes.Int8]):
    dtype = dtypes.int8


class Int16Data(_ArrayCoreInteger[dtypes.Int16]):
    dtype = dtypes.int16


class Int32Data(_ArrayCoreInteger[dtypes.Int32]):
    dtype = dtypes.int32


class Int64Data(_ArrayCoreInteger[dtypes.Int64]):
    dtype = dtypes.int64


class Uint8Data(_ArrayCoreInteger[dtypes.Uint8]):
    dtype = dtypes.uint8


class Uint16Data(_ArrayCoreInteger[dtypes.Uint16]):
    dtype = dtypes.uint16


class Uint32Data(_ArrayCoreInteger[dtypes.Uint32]):
    dtype = dtypes.uint32


class Uint64Data(_ArrayCoreInteger[dtypes.Uint64]):
    dtype = dtypes.uint64


class Float16Data(_ArrayCoreFloating[dtypes.Float16]):
    dtype = dtypes.float16


class Float32Data(_ArrayCoreFloating[dtypes.Float32]):
    dtype = dtypes.float32


class Float64Data(_ArrayCoreFloating[dtypes.Float64]):
    dtype = dtypes.float64


def ascoredata(var: Var) -> _ArrayCoreType:
    dtype = from_numpy(var.unwrap_tensor().dtype)

    return dtype._tyarr_class(var)


def is_sequence_of_core_data(
    seq: Sequence[_TypedArray],
) -> TypeGuard[Sequence[_ArrayCoreType]]:
    return all(isinstance(d, _ArrayCoreType) for d in seq)


def _promote_and_apply_op(
    lhs: _ArrayCoreType,
    rhs: _TypedArray,
    arr_op: Callable[[_ArrayCoreType, _ArrayCoreType], _ArrayCoreType],
    spox_op: Callable[[Var, Var], Var],
) -> _ArrayCoreType:
    """Promote and apply an operation by passing it through to the data member."""
    if isinstance(rhs, _ArrayCoreType):
        if lhs.dtype != rhs.dtype:
            a, b = promote(lhs, rhs)
            return arr_op(a, b)

        # Data is core & integer
        var = spox_op(lhs.var, rhs.var)
        return ascoredata(var)
    return NotImplemented
