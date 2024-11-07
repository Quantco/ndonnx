# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import operator
from abc import abstractmethod
from collections.abc import Callable, Sequence
from types import NotImplementedType
from typing import TYPE_CHECKING, Literal, TypeGuard, TypeVar, overload

import numpy as np
import spox
import spox._future
import spox.opset.ai.onnx.v21 as op
from spox import Tensor, Var, argument
from typing_extensions import Self

from ndonnx._corearray import _CoreArray

from .. import _dtypes as dtypes
from .._dtypes import TY_ARRAY, DType, as_numpy, from_numpy
from .._schema import DTypeInfoV1
from . import ort_compat
from .indexing import GetitemIndex, GetitemIndexStatic, SetitemIndex, SetitemIndexStatic
from .typed_array import TyArrayBase
from .utils import promote, safe_cast

if TYPE_CHECKING:
    from .._array import OnnxShape


TY_ARRAY_ONNX = TypeVar("TY_ARRAY_ONNX", bound="TyArray")


class _OnnxDType(DType[TY_ARRAY_ONNX]):
    """Data types with a direct representation in the ONNX standard."""

    @property
    @abstractmethod
    def _tyarr_class(self) -> type[TY_ARRAY_ONNX]: ...

    def __ndx_convert_tyarray__(self, arr: TyArrayBase) -> TY_ARRAY_ONNX:
        if isinstance(arr, TyArray):
            var = op.cast(arr.var, to=as_numpy(self))
            return safe_cast(self._tyarr_class, ascoredata(var))
        raise NotImplementedError

    def _argument(self, shape: OnnxShape) -> TY_ARRAY_ONNX:
        var = argument(Tensor(as_numpy(self), shape))
        return self._tyarr_class(var)

    @property
    def _infov1(self) -> DTypeInfoV1:
        return DTypeInfoV1(
            author="ndonnx", type_name=self.__class__.__name__, meta=None
        )


class _Number(_OnnxDType):
    def _result_type(self, rhs: DType) -> DType | NotImplementedType:
        if isinstance(self, NumericDTypes) and isinstance(rhs, NumericDTypes):
            return _result_type_core_numeric(self, rhs)

        return NotImplemented


class String(_OnnxDType):
    def _result_type(self, rhs: DType) -> DType | NotImplementedType:
        return NotImplemented

    @property
    def _tyarr_class(self) -> type[TyArrayString]:
        return TyArrayString


# TODO: Should this be a subclass of _Number?
class Boolean(_OnnxDType):
    def _result_type(self, rhs: DType) -> DType | NotImplementedType:
        return NotImplemented

    @property
    def _tyarr_class(self) -> type[TyArrayBool]:
        return TyArrayBool


class Int8(_Number):
    @property
    def _tyarr_class(self) -> type[TyArrayInt8]:
        return TyArrayInt8


class Int16(_Number):
    @property
    def _tyarr_class(self) -> type[TyArrayInt16]:
        return TyArrayInt16


class Int32(_Number):
    @property
    def _tyarr_class(self) -> type[TyArrayInt32]:
        return TyArrayInt32


class Int64(_Number):
    @property
    def _tyarr_class(self) -> type[TyArrayInt64]:
        return TyArrayInt64


class Uint8(_Number):
    @property
    def _tyarr_class(self) -> type[TyArrayUint8]:
        return TyArrayUint8


class Uint16(_Number):
    @property
    def _tyarr_class(self) -> type[TyArrayUint16]:
        return TyArrayUint16


class Uint32(_Number):
    @property
    def _tyarr_class(self) -> type[TyArrayUint32]:
        return TyArrayUint32


class Uint64(_Number):
    @property
    def _tyarr_class(self) -> type[TyArrayUint64]:
        return TyArrayUint64


class Float16(_Number):
    @property
    def _tyarr_class(self) -> type[TyArrayFloat16]:
        return TyArrayFloat16


class Float32(_Number):
    @property
    def _tyarr_class(self) -> type[TyArrayFloat32]:
        return TyArrayFloat32


class Float64(_Number):
    @property
    def _tyarr_class(self) -> type[TyArrayFloat64]:
        return TyArrayFloat64


# Non-nullable Singleton instances
bool_ = Boolean()

float16 = Float16()
float32 = Float32()
float64 = Float64()

int16 = Int16()
int32 = Int32()
int64 = Int64()
int8 = Int8()

uint8 = Uint8()
uint16 = Uint16()
uint32 = Uint32()
uint64 = Uint64()

string = String()

# Union types
#
# Union types are exhaustive and don't create ambiguities with respect to user-defined subtypes.
FloatingDTypes = Float16 | Float32 | Float64
SignedIntegerDTypes = Int8 | Int16 | Int32 | Int64
UnsignedIntegerDTypes = Uint8 | Uint16 | Uint32 | Uint64
IntegerDTypes = SignedIntegerDTypes | UnsignedIntegerDTypes
NumericDTypes = FloatingDTypes | IntegerDTypes
DTypes = Boolean | NumericDTypes | String


class TyArray(TyArrayBase):
    dtype: DTypes
    var: Var

    def __init__(self, var: Var):
        if from_numpy(var.unwrap_tensor().dtype) != self.dtype:
            raise ValueError(f"data type of 'var' is incompatible with `{type(self)}`")
        if var.unwrap_tensor().shape is None:
            raise ValueError("'var' has no shape information")
        self.var = var

    def __ndx_value_repr__(self) -> dict[str, str]:
        try:
            return {"data": str(self.unwrap_numpy().tolist())}
        except ValueError:
            return {"data": "*lazy*"}

    def __getitem__(self, key: GetitemIndex) -> Self:
        if isinstance(key, TyArrayInt64):
            raise NotImplementedError

        ca = _as_old_corarray(self)
        if isinstance(key, TyArrayBool):
            # Let's be defensive here: Bool arrays may become a
            # subclass of int array
            key_: GetitemIndexStatic | _CoreArray = _as_old_corarray(key)
        elif isinstance(key, TyArrayInteger):
            key_ = _as_old_corarray(key.astype(int64))
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
            key_ = _as_old_corarray(key.astype(int64))
        else:
            key_ = key
        ca[key_] = _as_old_corarray(value)

        # Check that the type was not changed by going through the constructor
        self.var = type(self)(ca.var).var

        return

    @property
    def dynamic_shape(self) -> TyArrayInt64:
        var = op.shape(self.var)
        return TyArrayInt64(var)

    @property
    def mT(self) -> Self:  # noqa: N802
        if self.ndim < 2:
            raise ValueError(
                "array must have two or more dimensions, found `{self.ndim}`"
            )
        dims = list(range(self.ndim))
        dims = dims[:-2] + dims[-2:][::-1]
        var = op.transpose(self.var, perm=dims)
        return type(self)(var)

    @property
    def shape(self) -> OnnxShape:
        shape = self.var.unwrap_tensor().shape
        if shape is None:
            raise ValueError("Missing shape information")
        return shape

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
                return np_arr.astype(str)

            # Should not happen
            raise ValueError("unexpected value data type")

        raise ValueError("no propagated value available")

    def all(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> TyArrayBool:
        if isinstance(axis, int):
            axis = (axis,)

        bools = self.astype(bool_)

        if bools.ndim == 0:
            if axis:
                ValueError("'axis' were provided but 'self' is a scalar")
            # Nothing left to reduce
            return safe_cast(TyArrayBool, bools)

        axes = op.const(list(axis), np.int64) if axis else None

        # Note: reduce_min, which would support uint8s, appears to be buggy on the onnxruntime
        # side. Thus we use reduce_prod for now.
        var = op.reduce_prod(bools.astype(int32).var, axes=axes, keepdims=keepdims)
        return safe_cast(TyArrayBool, TyArrayInt32(var).astype(bool_))

    def any(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> TyArrayBool:
        if isinstance(axis, int):
            axis = (axis,)

        bools = self.astype(bool_)

        if bools.ndim == 0:
            if axis:
                ValueError("'axis' were provided but 'self' is a scalar")
            # Nothing left to reduce
            return safe_cast(TyArrayBool, bools)

        axes = op.const(list(axis), np.int64) if axis else None

        # min uint8 is returned if dimensions are empty
        var = op.reduce_max(bools.astype(uint8).var, axes=axes, keepdims=keepdims)
        return safe_cast(TyArrayBool, TyArrayUint8(var).astype(bool_))

    def as_core_dtype(self, dtype: DTypes) -> TyArray:
        raise ValueError(f"Casting between `{self.dtype}` and `{dtype}` is undefined")

    def broadcast_to(self, shape: tuple[int, ...] | TyArrayInt64) -> Self:
        if isinstance(shape, tuple):
            shape_var = op.const(shape, dtype=np.int64)
        else:
            shape_var = shape.var
        var = op.expand(self.var, shape_var)
        return type(self)(var)

    def concat(self, others: list[Self], axis: None | int) -> Self:
        var = op.concat(
            [self.var] + [el.var for el in others], axis=0 if axis is None else axis
        )
        res = type(self)(var)
        if axis is None and res.ndim != 1:
            # Flatten the result. It is ok to concat first on the 0th axis.
            res = res.reshape((-1,))
        return res

    def disassemble(self) -> Var:
        return self.var

    def permute_dims(self, axes: tuple[int, ...]) -> Self:
        var = op.transpose(self.var, perm=axes)
        return type(self)(var)

    def reshape(self, shape: tuple[int, ...] | TyArrayInt64) -> Self:
        if isinstance(shape, tuple | list):
            var = op.reshape(self.var, op.const(shape, np.int64), allowzero=True)
        else:
            var = op.reshape(self.var, shape.var, allowzero=True)
        return type(self)(var)

    def searchsorted(
        self,
        x2: Self,
        /,
        *,
        side: Literal["left", "right"] = "left",
        sorter: TyArrayInteger | None = None,
    ) -> TyArrayInt64:
        x1 = self
        if x1.ndim != 1:
            raise TypeError("x1 must be a 1-dimensional array")
        if sorter is not None:
            raise NotImplementedError(
                "'sorter' argument of 'searchsorted' is not implemented, yet"
            )
            # x1 = x1.take(sorter)

        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")

        x1 = x1[:, None]
        x2 = x2[None, :]
        if side == "left":
            indices = (x1 < x2).astype(int64).sum(axis=0)
        else:
            indices = (x1 <= x2).astype(int64).sum(axis=0)

        return indices

    def unique_all(self) -> tuple[Self, TyArrayInt64, TyArrayInt64, TyArrayInt64]:
        flattened = self.reshape((-1,))
        values, indices, inverse_indices, counts = op.unique(flattened.var, sorted=True)

        return (
            type(self)(values),
            TyArrayInt64(indices),
            TyArrayInt64(inverse_indices),
            TyArrayInt64(counts),
        )

    def __ndx_astype__(self, dtype: DType[TY_ARRAY]) -> TY_ARRAY:
        # TODO: How is this never hit?
        return NotImplemented

    def _eqcomp(self, other) -> TyArrayBase:
        return _promote_and_apply_op(self, other, operator.eq, op.equal, True)

    @overload
    def __ndx_where__(
        self, cond: TyArrayBool, y: TyArray, /
    ) -> TyArray | NotImplementedType: ...

    @overload
    def __ndx_where__(
        self, cond: TyArrayBool, y: TyArrayBase, /
    ) -> TyArrayBase | NotImplementedType: ...

    def __ndx_where__(
        self, cond: TyArrayBool, y: TyArrayBase, /
    ) -> TyArrayBase | NotImplementedType:
        if isinstance(y, TyArray):
            x, y = promote(self, y)
            var = op.where(cond.var, x.var, y.var)
            return type(x)(var)

        return NotImplemented

    @overload
    def __ndx_maximum__(self, rhs: TyArray, /) -> TyArray | NotImplementedType: ...

    @overload
    def __ndx_maximum__(
        self, rhs: TyArrayBase, /
    ) -> TyArrayBase | NotImplementedType: ...

    def __ndx_maximum__(self, rhs: TyArrayBase, /) -> TyArrayBase | NotImplementedType:
        if isinstance(rhs, TyArray):
            lhs, rhs = promote(self, rhs)
            var = op.max([lhs.var, rhs.var])
            return type(lhs)(var)

        return NotImplemented


class TyArrayString(TyArray):
    dtype = string

    def __add__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.add, op.string_concat, True)

    def __radd__(self, lhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, lhs, operator.add, op.string_concat, False)


class TyArrayNumber(TyArray):
    dtype: NumericDTypes

    def cumulative_sum(
        self,
        /,
        *,
        axis: int | None = None,
        dtype: DType | None = None,
        include_initial: bool = False,
    ) -> TyArrayBase:
        from .funcs import astyarray

        if dtype is None:
            if isinstance(self.dtype, SignedIntegerDTypes):
                dtype_: DType = int64
            elif isinstance(self.dtype, UnsignedIntegerDTypes):
                dtype_ = uint64
            else:
                dtype_ = self.dtype
        else:
            dtype_ = dtype
        if self.dtype != dtype_:
            return self.astype(dtype_).cumulative_sum(
                axis=axis, include_initial=include_initial, dtype=dtype_
            )

        if axis is None:
            if self.ndim > 1:
                raise ValueError(
                    "'axis' parameter must be provided for arrays with more than one dimension."
                )
            axis_ = 0
        else:
            axis_ = axis

        out = safe_cast(
            type(self),
            ascoredata(
                ort_compat.cumsum(
                    self.var,
                    op.const(axis_, np.int64),
                    exclusive=False,
                )
            ),
        )
        if include_initial:
            out_shape = out.dynamic_shape
            out_shape[axis_] = safe_cast(TyArrayInt64, astyarray(1))
            zeros = safe_cast(
                type(self), astyarray(0, out.dtype).broadcast_to(out_shape)
            )
            out = zeros.concat([out], axis=axis_)
        return out

    def _reduce(
        self,
        pub_name: str,
        var_op: Callable[..., Var],
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
    ) -> TyArrayBase:
        if dtype is None:
            if isinstance(self.dtype, SignedIntegerDTypes):
                dtype_: DType = int64
            elif isinstance(self.dtype, UnsignedIntegerDTypes):
                dtype_ = uint64
            else:
                dtype_ = self.dtype
        else:
            dtype_ = dtype
        if self.dtype != dtype_:
            member_after_cast = getattr(self.astype(dtype_), pub_name)
            return member_after_cast(axis=axis, keepdims=keepdims, dtype=dtype)

        if axis is None:
            axis_ = None
        elif isinstance(axis, int):
            axis_ = op.const([axis], dtype=np.int64)
        else:
            axis_ = op.const(axis, dtype=np.int64)
        var = var_op(
            self.var,
            axis_,
            keepdims=keepdims,
            noop_with_empty_axes=axis is not None,
        )
        return ascoredata(var)

    def prod(
        self,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
    ) -> TyArrayBase:
        return self._reduce(
            "prod", ort_compat.reduce_prod, axis=axis, dtype=dtype, keepdims=keepdims
        )

    def sum(
        self,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
    ) -> TyArrayBase:
        return self._reduce(
            "sum", op.reduce_sum, axis=axis, dtype=dtype, keepdims=keepdims
        )

    def __add__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.add, ort_compat.add, True)

    def __radd__(self, lhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, lhs, operator.add, ort_compat.add, False)

    def __ge__(self, rhs: TyArrayBase, /) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.ge, op.greater_or_equal, True)

    def __gt__(self, rhs: TyArrayBase, /) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.gt, op.greater, True)

    def __le__(self, rhs: TyArrayBase, /) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.le, op.less_or_equal, True)

    def __lt__(self, rhs: TyArrayBase, /) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.lt, op.less, True)

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

    def __pow__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.pow, op.pow, True)

    def __rpow__(self, lhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, lhs, operator.pow, op.pow, False)

    # Element-wise functions
    def __abs__(self) -> Self:
        # ORT supports all data types
        return type(self)(op.abs(self.var))


class TyArrayInteger(TyArrayNumber):
    dtype: IntegerDTypes

    def __truediv__(self, rhs: TyArrayBase) -> TyArrayBase:
        # Casting rules are implementation defined. We default to float64 like NumPy
        lhs = self.astype(float64)
        return lhs / rhs

    def __or__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.or_, op.bitwise_or, True)

    def __mod__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(
            self, rhs, operator.mod, lambda a, b: op.mod(a, b, fmod=0), True
        )

    def __rmod__(self, lhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(
            self, lhs, operator.mod, lambda a, b: op.mod(a, b, fmod=0), False
        )

    def isnan(self) -> TyArrayBool:
        var = op.constant_of_shape(op.shape(self.var), value=np.array(False))
        return TyArrayBool(var)

    def isinf(self) -> TyArrayBool:
        from .._array import Array
        from .._funcs import full_like

        return safe_cast(TyArrayBool, full_like(Array._from_data(self), False)._data)


class TyArrayFloating(TyArrayNumber):
    dtype: FloatingDTypes

    def __mod__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(
            self, rhs, operator.mod, lambda a, b: op.mod(a, b, fmod=1), True
        )

    def __rmod__(self, lhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(
            self, lhs, operator.mod, lambda a, b: op.mod(a, b, fmod=1), False
        )

    def mean(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Self:
        if axis is None:
            axes = None
        elif isinstance(axis, int):
            axes = op.const([axis], np.int64)
        else:
            axes = op.const(axis, np.int64)

        return type(self)(op.reduce_mean(self.var, axes=axes, keepdims=keepdims))

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
        from .py_scalars import TyArrayPyFloat

        res = self.log() / TyArrayPyFloat(float(np.log(2)))
        return safe_cast(type(self), res)

    # TODO: remaining element-wise functions


class TyArrayBool(TyArray):
    dtype = bool_

    def __or__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.or_, op.or_, True)

    def __and__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.and_, op.and_, True)

    def __invert__(self) -> TyArrayBool:
        var = op.not_(self.var)
        return type(self)(var)


class TyArrayInt8(TyArrayInteger):
    dtype = int8


class TyArrayInt16(TyArrayInteger):
    dtype = int16


class TyArrayInt32(TyArrayInteger):
    dtype = int32


class TyArrayInt64(TyArrayInteger):
    dtype = int64


class TyArrayUint8(TyArrayInteger):
    dtype = uint8


class TyArrayUint16(TyArrayInteger):
    dtype = uint16


class TyArrayUint32(TyArrayInteger):
    dtype = uint32


class TyArrayUint64(TyArrayInteger):
    dtype = uint64


class TyArrayFloat16(TyArrayFloating):
    dtype = float16


class TyArrayFloat32(TyArrayFloating):
    dtype = float32


class TyArrayFloat64(TyArrayFloating):
    dtype = float64


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


T = TypeVar("T", bound=TyArray)


def _element_wise(x: T, op: Callable[[Var], Var], via: _OnnxDType | None = None) -> T:
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


#####################
# Conversion tables #
#####################

# Promotion tables taken from:
# https://data-apis.org/array-api/draft/API_specification/type_promotion.html#type-promotion
# and
# https://numpy.org/neps/nep-0050-scalar-promotion.html#motivation-and-scope
_signed_signed: dict[tuple[NumericDTypes, NumericDTypes], NumericDTypes] = {
    (int8, int8): int8,
    (int16, int8): int16,
    (int32, int8): int32,
    (int64, int8): int64,
    (int8, int16): int16,
    (int16, int16): int16,
    (int32, int16): int32,
    (int64, int16): int64,
    (int8, int32): int32,
    (int16, int32): int32,
    (int32, int32): int32,
    (int64, int32): int64,
    (int8, int64): int64,
    (int16, int64): int64,
    (int32, int64): int64,
    (int64, int64): int64,
}
_unsigned_unsigned: dict[tuple[NumericDTypes, NumericDTypes], NumericDTypes] = {
    (uint8, uint8): uint8,
    (uint16, uint8): uint16,
    (uint32, uint8): uint32,
    (uint64, uint8): uint64,
    (uint8, uint16): uint16,
    (uint16, uint16): uint16,
    (uint32, uint16): uint32,
    (uint64, uint16): uint64,
    (uint8, uint32): uint32,
    (uint16, uint32): uint32,
    (uint32, uint32): uint32,
    (uint64, uint32): uint64,
    (uint8, uint64): uint64,
    (uint16, uint64): uint64,
    (uint32, uint64): uint64,
    (uint64, uint64): uint64,
}
_mixed_integers: dict[tuple[NumericDTypes, NumericDTypes], NumericDTypes] = {
    (int8, uint8): int16,
    (int16, uint8): int16,
    (int32, uint8): int32,
    (int64, uint8): int64,
    (int8, uint16): int32,
    (int16, uint16): int32,
    (int32, uint16): int32,
    (int64, uint16): int64,
    (int8, uint32): int64,
    (int16, uint32): int64,
    (int32, uint32): int64,
    (int64, uint32): int64,
    # NOTE: Standard does not define interaction with uint64!
}

_floating_floating: dict[tuple[NumericDTypes, NumericDTypes], NumericDTypes] = {
    (float32, float32): float32,
    (float32, float64): float64,
    (float64, float32): float64,
    (float64, float64): float64,
}

# Non-standard interactions
_non_standard: dict[tuple[NumericDTypes, NumericDTypes], NumericDTypes] = {
    (int8, uint64): float64,
    (int16, uint64): float64,
    (int32, uint64): float64,
    (int64, uint64): float64,
}

# Mixed integers and floating point numbers are not
# strictly defined, but generally we want to cast the
# integer to a floating point and then try again.
_int_to_floating: dict[NumericDTypes, NumericDTypes] = {
    int8: float16,
    uint8: float16,
    int16: float32,
    uint16: float32,
    int32: float64,
    uint32: float64,
    int64: float64,
    uint64: float64,
}


def _result_type_core_numeric(a: NumericDTypes, b: NumericDTypes) -> NumericDTypes:
    # Attempt promotion between known types. The implementation is not
    # using `isinstance` to avoid subclassing issues.
    if ret := _signed_signed.get((a, b)):
        return ret
    if ret := _unsigned_unsigned.get((a, b)):
        return ret
    if ret := _mixed_integers.get((a, b)):
        return ret
    if ret := _floating_floating.get((a, b)):
        return ret
    if ret := _non_standard.get((a, b)):
        return ret
    if a_floating := _int_to_floating.get(a):
        return _result_type_core_numeric(a_floating, b)
    if b_floating := _int_to_floating.get(b):
        return _result_type_core_numeric(a, b_floating)

    # TODO: Do bools and strings

    raise ValueError(f"No promotion between `{a}` and `{b}` is defined.")
