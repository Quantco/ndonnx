# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from types import EllipsisType, NotImplementedType
from typing import TYPE_CHECKING, TypeGuard, TypeVar

import numpy as np
import spox
import spox._future
import spox.opset.ai.onnx.v21 as op
from spox import Var
from typing_extensions import Self

from ndonnx._corearray import _CoreArray

from .. import dtypes
from ..dtypes import CoreDTypes, DType, float32, float64, from_numpy
from ..schema import Schema, var_to_primitive
from .indexing import GetitemIndex, GetitemIndexStatic, SetitemIndex, SetitemIndexStatic
from .typed_array import TyArrayBase
from .utils import promote, safe_cast

if TYPE_CHECKING:
    from ..array import OnnxShape
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

    def __getitem__(self, key: GetitemIndex) -> Self:
        if isinstance(key, TyArrayInt64):
            raise NotImplementedError

        ca = _as_old_corarray(self)
        if isinstance(key, TyArrayBool):
            # Let's be defensive here: Bool arrays may become a
            # subclass of int array
            key_: GetitemIndexStatic | _CoreArray = _as_old_corarray(key)
        elif isinstance(key, TyArrayInteger):
            key_ = _as_old_corarray(key.astype(dtypes.int64))
        else:
            key_ = key

        return type(self)(ca[key_].var)

    def __setitem__(
        self,
        key: SetitemIndex,
        value: Self,
        /,
    ) -> None:
        ca = _as_old_corarray(self)
        if isinstance(key, TyArrayBool):
            key_: SetitemIndexStatic | _CoreArray = _as_old_corarray(key)
        elif isinstance(key, TyArrayInteger):
            key_ = _as_old_corarray(key.astype(dtypes.int64))
        else:
            key_ = key
        ca[key_] = _as_old_corarray(value)

        # Check that the type was not changed by going through the constructor
        self.var = type(self)(ca.var).var

        return

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

    def unwrap_numpy(self) -> np.ndarray:
        if self.var._value is not None:
            np_arr = np.asarray(self.var._value.value)
            np_arr.astype(np.dtypes.StringDType)
            if np_arr.dtype == dtypes.as_numpy(self.dtype):
                return np_arr
            if dtypes.as_numpy(self.dtype).kind == "U" and np_arr.dtype.kind in [
                "U",
                "O",
            ]:
                # String-case
                # TODO: Where does the "object" kind come from?
                # Probably spox; we should have a more predictable
                # upstream string support.
                return np_arr

            # Should not happen
            raise ValueError("unexpected value data type")

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

        axes = op.const(list(axis), np.int64) if axis else None

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
        var = op.reshape(self.var, op.const(shape, np.int64), allowzero=True)
        return type(self)(var)

    def broadcast_to(self, shape: tuple[int, ...] | TyArrayInt64) -> Self:
        if isinstance(shape, tuple):
            shape_var = op.const(shape, dtype=np.int64)
        else:
            shape_var = shape.var
        var = op.expand(self.var, shape_var)
        return type(self)(var)

    def permute_dims(self, axes: tuple[int, ...]) -> Self:
        var = op.transpose(self.var, perm=axes)
        return type(self)(var)

    def as_core_dtype(self, dtype: CoreDTypes) -> TyArray:
        raise ValueError(f"Casting between `{self.dtype}` and `{dtype}` is undefined")

    def _astype(self, dtype: DType) -> TyArrayBase:
        return NotImplemented

    def _eqcomp(self, other) -> TyArrayBase:
        return _promote_and_apply_op(self, other, operator.eq, op.equal, True)

    def __ndx_where__(
        self, cond: TyArrayBool, y: TyArrayBase, /
    ) -> TyArrayBase | NotImplementedType:
        if isinstance(y, TyArray):
            x, y = promote(self, y)
            var = op.where(cond.var, x.var, y.var)
            return type(x)(var)

        return NotImplemented

    def __ndx_maximum__(self, rhs: TyArrayBase, /) -> TyArrayBase | NotImplementedType:
        if isinstance(rhs, TyArray):
            lhs, rhs = promote(self, rhs)
            var = op.max([lhs.var, rhs.var])
            return type(lhs)(var)

        return NotImplemented


class TyArrayString(TyArray):
    dtype = dtypes.string

    def __add__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.add, op.string_concat, True)

    def __radd__(self, lhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, lhs, operator.add, op.string_concat, False)


class TyArrayNumber(TyArray):
    dtype: dtypes.CoreNumericDTypes

    def sum(
        self,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
    ) -> TyArrayBase:
        if dtype is not None and not isinstance(dtype, dtypes.CoreNumericDTypes):
            return self.astype(dtype).sum(axis=axis, dtype=dtype, keepdims=keepdims)
        elif dtype is None and isinstance(self, TyArrayInteger):
            # Input is signed
            if self.dtype in (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64):
                dtype_: dtypes.CoreNumericDTypes = dtypes.int64
            else:
                dtype_ = dtypes.uint64
        elif dtype is None:
            dtype_ = self.dtype
        else:
            dtype_ = dtype
        if axis is None:
            axis_ = None
        elif isinstance(axis, int):
            axis_ = op.const([axis], dtype=np.int64)
        else:
            axis_ = op.const(axis, dtype=np.int64)
        var = op.reduce_sum(
            self.astype(dtype_).var,
            axis_,
            keepdims=keepdims,
            noop_with_empty_axes=axis is not None,
        )
        return ascoredata(var)

    def __add__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.add, op.add, True)

    def __radd__(self, lhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, lhs, operator.add, op.add, False)

    def __ge__(self, rhs: TyArrayBase, /) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.ge, op.greater_or_equal, False)

    def __gt__(self, rhs: TyArrayBase, /) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.gt, op.greater, False)

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

    def isinf(self) -> TyArrayBool:
        from ..array import Array
        from ..funcs import full_like

        return safe_cast(TyArrayBool, full_like(Array._from_data(self), False)._data)


T = TypeVar("T", bound=TyArray)


def _element_wise(
    x: T, op: Callable[[Var], Var], via: dtypes._CoreDType | None = None
) -> T:
    """Apply an element wise operations possible with an onnxruntime workaround ``via``
    a different data type.

    The workaround is only applied if
    ``spox._value_prop._VALUE_PROP_BACKEND==spox._future.ValuePropBackend.ONNXRUNTIME``.
    """
    target_ort = (
        spox._value_prop._VALUE_PROP_BACKEND
        == spox._future.ValuePropBackend.ONNXRUNTIME
    )
    if via is not None and target_ort:
        res = ascoredata(op(x.astype(via).var))
        return safe_cast(type(x), res.astype(x.dtype))
    return type(x)(op(x.var))


class TyArrayFloating(TyArrayNumber):
    dtype: dtypes.CoreFloatingDTypes

    def isinf(self) -> TyArrayBool:
        return TyArrayBool(op.isinf(self.var))

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

    def floor(self) -> Self:
        return _element_wise(self, op.floor, float64)

    def isnan(self) -> TyArrayBool:
        return TyArrayBool(op.isnan(self.var))

    def log(self) -> Self:
        return _element_wise(self, op.log, float64)

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


class TyArrayFloat16(TyArrayFloating):
    dtype = dtypes.float16


class TyArrayFloat32(TyArrayFloating):
    dtype = dtypes.float32


class TyArrayFloat64(TyArrayFloating):
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


def _remove_trailing_ellipsis_and_none_slice(
    indices: tuple[int | slice | EllipsisType, ...],
) -> tuple[int | slice | EllipsisType, ...]:
    if indices[-1] in (..., slice(None)):
        return _remove_trailing_ellipsis_and_none_slice(indices[:-1])
    return tuple(indices)


def _as_old_corarray(tyarr: TyArray) -> _CoreArray:
    from ndonnx._corearray import _CoreArray

    ca = _CoreArray(tyarr.var)
    try:
        spox_prop_val = tyarr.var._get_value()
        if not isinstance(spox_prop_val, np.ndarray):
            raise ValueError(
                "Propagated value has unexpected type `{type(spox_prop_val)}`"
            )
        ca._eager_value = spox_prop_val
    except ValueError:
        pass
    return ca
