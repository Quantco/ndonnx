# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from types import EllipsisType, NotImplementedType
from typing import TYPE_CHECKING, TypeGuard, TypeVar

import numpy as np
import spox.opset.ai.onnx.v21 as op
from spox import Var
from typing_extensions import Self

from .. import dtypes
from ..dtypes import CoreDTypes, DType, float32, float64, from_numpy
from ..schema import Schema, var_to_primitive
from .typed_array import TyArrayBase
from .utils import promote, safe_cast

if TYPE_CHECKING:
    from ..array import Index, OnnxShape, SetitemIndex
    from ..schema import Components


CORE_DTYPES = TypeVar("CORE_DTYPES", bound=CoreDTypes)
ALL_NUM_DTYPES = TypeVar(
    "ALL_NUM_DTYPES", bound=dtypes.CoreNumericDTypes | dtypes.NCoreNumericDTypes
)


class _Index:
    starts: list[int]
    ends: list[int]
    steps: list[int]
    axes: list[int]
    squeeze_axes: list[int]

    def __init__(self, index: Index):
        if isinstance(index, tuple):
            index_ = index
        elif isinstance(index, int | slice):
            index_ = (index,)
        else:
            raise NotImplementedError
        self.starts = []
        self.ends = []
        self.steps = []
        self.axes = []
        self.squeeze_axes = []

        def compute_end_slice(stop: int | None, step: int | None) -> int:
            if isinstance(stop, int):
                return stop
            step = step or 1
            # Iterate "to the end"
            if step < 1:
                return int(np.iinfo(np.int64).min)
            return int(np.iinfo(np.int64).max)

        def compute_end_single_idx(start: int):
            end = start + 1
            if end == 0:
                return np.iinfo(np.int64).max
            return end

        has_ellipsis = False
        for i, el in enumerate(index_):
            if isinstance(el, slice):
                self.starts.append(el.start or 0)
                self.ends.append(compute_end_slice(el.stop, el.step))
                self.axes.append(i)
                self.steps.append(el.step or 1)
            elif isinstance(el, int):
                self.starts.append(el)
                self.ends.append(compute_end_single_idx(el))
                self.axes.append(i)
                self.steps.append(1)
                self.squeeze_axes.append(i)
            elif isinstance(el, type(...)):
                # Continue from the back until we hit the ellipsis again
                has_ellipsis = True
                break
            else:
                raise NotImplementedError
        if not has_ellipsis:
            return
        for i, el in enumerate(index_[::-1], start=1):
            if isinstance(el, slice):
                self.starts.append(el.start or 0)
                self.ends.append(compute_end_slice(el.stop, el.step))
                self.axes.append(-i)
                self.steps.append(el.step or 1)
            elif isinstance(el, int):
                self.starts.append(el)
                self.ends.append(compute_end_single_idx(el))
                self.axes.append(-i)
                self.steps.append(1)
                self.squeeze_axes.append(-i)
            elif isinstance(el, type(...)):
                break
            else:
                raise NotImplementedError


class TyArray(TyArrayBase):
    dtype: dtypes.CoreDTypes
    var: Var

    def __init__(self, var: Var):
        if from_numpy(var.unwrap_tensor().dtype) != self.dtype:
            raise ValueError(f"data type of 'var' is incompatible with `{type(self)}`")
        self.var = var

    def __getitem__(self, index: Index) -> Self:
        if index == ():
            return self
        parsed = _Index(index)
        var = op.slice(
            self.var,
            starts=op.const(parsed.starts),
            ends=op.const(parsed.ends),
            axes=op.const(parsed.axes),
            steps=op.const(parsed.steps),
        )
        var = op.squeeze(var, axes=op.const(parsed.squeeze_axes, np.int64))
        return type(self)(var)

    def __setitem__(
        self,
        key: SetitemIndex,
        value: Self,
        /,
    ) -> None:
        from ..array import Array

        if isinstance(key, Array):
            raise NotImplementedError
        if value.ndim == 0:
            value = type(self)(op.unsqueeze(value.var, op.const([0])))
        if isinstance(key, int | EllipsisType | slice):
            keys: tuple = (key,)
        else:
            keys = key
        if keys[-1] == ...:
            # remove trailing ellipsis
            keys = keys[:-1]
        if isinstance(keys, tuple) and all_items_are_int(keys):
            var = op.scatter_nd(self.var, op.const([keys]), value.var)

        else:
            # - Roll axes until all ellipsis axes are in the end
            # - Update
            # - Roll back
            raise NotImplementedError

        # Validate that var has the right type
        self.var = type(self)(var).var

    @property
    def shape(self) -> OnnxShape:
        shape = self.var.unwrap_tensor().shape
        if shape is None:
            raise ValueError("Missing shape information")
        return shape

    @property
    def dynamic_shape(self) -> TyArrayInt64:
        var = op.shape(self.var)
        return TyArrayInt64(var)

    def to_numpy(self) -> np.ndarray:
        if self.var._value is not None:
            np_arr = np.asarray(self.var._value.value)
            if np_arr.dtype != dtypes.as_numpy(self.dtype):
                # Should not happen
                raise ValueError("unexpected value data type")
            return np_arr
        raise ValueError("no propagated value available")

    def all(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> TyArrayBool:
        if isinstance(axis, int):
            axis = (axis,)

        bools = self.astype(dtypes.bool_)

        if bools.ndim == 0:
            if axis:
                ValueError("'axis' were provided but 'self' is a scalar")
            # Nothing left to reduce
            return safe_cast(TyArrayBool, bools)

        axes = op.const(list(axis)) if axis else None

        # max int8 is returned if dimensions are empty
        var = op.reduce_min(bools.astype(dtypes.int8).var, axes=axes, keepdims=keepdims)
        return safe_cast(TyArrayBool, TyArrayInt8(var).astype(dtypes.bool_))

    def disassemble(self) -> tuple[Components, Schema]:
        dtype_info = self.dtype._info
        component_schema = var_to_primitive(self.var)
        schema = Schema(dtype_info=dtype_info, components=component_schema)
        components = {"var": self.var}
        return components, schema

    def reshape(self, shape: tuple[int, ...]) -> Self:
        var = op.reshape(self.var, op.const(shape), allowzero=True)
        return type(self)(var)

    def broadcast_to(self, shape: tuple[int, ...] | TyArrayInt64) -> Self:
        if isinstance(shape, tuple):
            shape_var = op.const(shape, dtype=np.int64)
        else:
            shape_var = shape.var
        var = op.expand(self.var, shape_var)
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

    def __radd__(self, lhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, lhs, operator.add, op.add, False)

    def __truediv__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.truediv, op.div, True)

    def __rtruediv__(self, lhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, lhs, operator.truediv, op.div, False)

    def __mul__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.mul, op.mul, True)

    def __rmul__(self, lhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, lhs, operator.mul, op.mul, False)

    def __sub__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.sub, op.sub, True)

    def __rsub__(self, lhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, lhs, operator.sub, op.sub, False)

    # Element-wise functions
    def __abs__(self) -> Self:
        # ORT supports all data types
        return type(self)(op.abs(self.var))


class TyArrayInteger(TyArrayNumber):
    dtype: dtypes.CoreIntegerDTypes

    def __or__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.or_, op.bitwise_or, True)

    def isnan(self) -> TyArrayBool:
        var = op.constant_of_shape(op.shape(self.var), value=np.array(False))
        return TyArrayBool(var)


T = TypeVar("T", bound=TyArray)


def _element_wise(
    x: T, op: Callable[[Var], Var], via: dtypes._CoreDType | None = None
) -> T:
    if via is not None:
        return safe_cast(type(x), type(x)(op(x.astype(via).var)).astype(x.dtype))
    return type(x)(op(x.var))


class TyArrayFloating(TyArrayNumber):
    dtype: dtypes.CoreFloatingDTypes

    # Element-wise for floating point
    def acos(self) -> Self:
        return _element_wise(self, op.acos, float32)

    def acosh(self) -> Self:
        return _element_wise(self, op.acosh, float32)

    def asin(self) -> Self:
        return _element_wise(self, op.asin, float32)

    def asinh(self) -> Self:
        return _element_wise(self, op.asinh, float32)

    def atan(self) -> Self:
        return _element_wise(self, op.atan, float32)

    def atanh(self) -> Self:
        return _element_wise(self, op.atanh, float32)

    def ceil(self) -> Self:
        return _element_wise(self, op.ceil, float64)

    def exp(self) -> Self:
        return _element_wise(self, op.exp)

    def expm1(self) -> Self:
        from .py_scalars import _ArrayPyFloat

        return safe_cast(type(self), self - _ArrayPyFloat(1.0)).exp()

    def floor(self) -> Self:
        return _element_wise(self, op.floor, float64)

    def isnan(self) -> TyArrayBool:
        return TyArrayBool(op.isnan(self.var))

    def log(self) -> Self:
        return _element_wise(self, op.log, float64)

    def log1p(self) -> Self:
        from .py_scalars import _ArrayPyFloat

        x = safe_cast(type(self), self + _ArrayPyFloat(1.0))
        return _element_wise(x, op.log, float64)

    def log2(self) -> Self:
        from .py_scalars import _ArrayPyFloat

        res = self.log() / _ArrayPyFloat(float(np.log(2)))
        return safe_cast(type(self), res)

    # TODO: remaining element-wise functions


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


def all_items_are_int(
    seq: Sequence,
) -> TypeGuard[Sequence[int]]:
    return all(isinstance(d, int) for d in seq)


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
