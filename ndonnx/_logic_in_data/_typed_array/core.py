# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeGuard, TypeVar, overload

import numpy as np
import spox.opset.ai.onnx.v21 as op
from spox import Tensor, Var, argument
from typing_extensions import Self

from .. import dtypes
from ..dtypes import (
    CoreDTypes,
    DType,
    result_type,
)
from .typed_array import _TypedArray

if TYPE_CHECKING:
    from ..array import OnnxShape


CORE_DTYPES = TypeVar("CORE_DTYPES", bound=CoreDTypes)
ALL_NUM_DTYPES = TypeVar(
    "ALL_NUM_DTYPES", bound=dtypes.CoreNumericDTypes | dtypes.NCoreNumericDTypes
)


class _ArrayCoreType(_TypedArray[CORE_DTYPES]):
    var: Var

    def __init__(self, var: Var):
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

    def __getitem__(self, index) -> Self:
        # TODO
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        return len(self.shape)

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

    def reshape(self, shape: tuple[int, ...]) -> _ArrayCoreType:
        var = op.reshape(self.var, op.const(shape))
        return ascoredata(var)

    @overload
    def promote(self, *others: _ArrayCoreType) -> Sequence[_ArrayCoreType]: ...

    @overload
    def promote(self, *others: _TypedArray) -> Sequence[_TypedArray]: ...

    def promote(self, *others: _TypedArray) -> Sequence[_TypedArray]:
        # TODO
        raise NotImplementedError

    def as_core_dtype(self, dtype: CoreDTypes) -> _ArrayCoreType:
        raise ValueError(f"Casting between `{self.dtype}` and `{dtype}` is undefine")

    def __add__(self, lhs: _TypedArray) -> _ArrayCoreType:
        return NotImplemented

    def _promote(self, *others: _TypedArray) -> list[_ArrayCoreType]:
        """Promote with other `_ArrayCoreType` objects or return `NotImplemented`."""
        if is_sequence_of_core_data(others):
            res_type = result_type(self.dtype, *[d.dtype for d in others])
            return [self.as_core_dtype(res_type)] + [
                d.as_core_dtype(res_type) for d in others
            ]
        return NotImplemented

    def __or__(self, rhs: _TypedArray) -> _ArrayCoreType:
        return NotImplemented

    def _astype(self, dtype: DType) -> _TypedArray:
        return NotImplemented


class _ArrayCoreNum(_ArrayCoreType[CORE_DTYPES]):
    def __add__(self, rhs: _TypedArray) -> _ArrayCoreType:
        if isinstance(rhs, _ArrayCoreNum):
            # NOTE: Can't always promote for all data types (c.f. datetime / timedelta)
            if type(self) != type(rhs):
                a, b = self.promote(self, rhs)
                return a + b
            var = op.add(self.var, rhs.var)
            return ascoredata(var)
        return NotImplemented


class _ArrayCoreInteger(_ArrayCoreNum[CORE_DTYPES]):
    def __or__(self, rhs: _TypedArray) -> _ArrayCoreType:
        if isinstance(rhs, _ArrayCoreType):
            lhs: _ArrayCoreType = self
            if lhs.dtype != rhs.dtype:
                lhs, rhs = lhs.promote(rhs)
                return lhs.__or__(rhs)

            # Data is core & integer
            var = op.bitwise_or(lhs.var, rhs.var)
            return ascoredata(var)
        return NotImplemented


class _ArrayCoreFloating(_ArrayCoreNum[CORE_DTYPES]): ...


class BoolData(_ArrayCoreType[dtypes.Bool]):
    dtype = dtypes.bool_

    def __or__(self, rhs: _TypedArray) -> _ArrayCoreType:
        if isinstance(rhs, _ArrayCoreType):
            lhs: _ArrayCoreType = self
            if lhs.dtype != rhs.dtype:
                lhs, rhs = lhs.promote(rhs)
                return lhs.__or__(rhs)

            # Data is core & bool
            var = op.or_(lhs.var, rhs.var)
            return ascoredata(var)
        return NotImplemented


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
    np_dtype = var.unwrap_tensor().dtype

    mapping: dict[np.dtype, _ArrayCoreType] = {
        np.dtype("bool"): BoolData(var),
        np.dtype("int8"): Int8Data(var),
        np.dtype("int16"): Int16Data(var),
        np.dtype("int32"): Int32Data(var),
        np.dtype("int64"): Int64Data(var),
        np.dtype("uint8"): Uint8Data(var),
        np.dtype("uint16"): Uint16Data(var),
        np.dtype("uint32"): Uint32Data(var),
        np.dtype("uint64"): Uint64Data(var),
        np.dtype("float16"): Float16Data(var),
        np.dtype("float32"): Float32Data(var),
        np.dtype("float64"): Float64Data(var),
    }
    try:
        return mapping[np_dtype]
    except KeyError:
        raise NotImplementedError


def is_sequence_of_core_data(
    seq: Sequence[_TypedArray],
) -> TypeGuard[Sequence[_ArrayCoreType]]:
    return all(isinstance(d, _ArrayCoreType) for d in seq)
