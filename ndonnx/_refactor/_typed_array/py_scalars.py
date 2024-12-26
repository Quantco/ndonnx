# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import operator
from collections.abc import Callable
from types import NotImplementedType
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import Self

import ndonnx._refactor as ndx
from ndonnx._refactor._schema import DTypeInfoV1

from .._dtypes import TY_ARRAY_BASE, DType
from . import TyArrayBase, masked_onnx, onnx, promote, safe_cast

if TYPE_CHECKING:
    from spox import Var

    from .._array import OnnxShape
    from .indexing import SetitemIndex
    from .onnx import TyArrayInt64


class _PyScalar(DType):
    def __ndx_cast_from__(self, arr: TyArrayBase) -> Self:
        raise NotImplementedError

    def _argument(self, shape: OnnxShape):
        raise ValueError("'{type(self)}' cannot be used as a model argument")

    @property
    def _infov1(self) -> DTypeInfoV1:
        raise ValueError("'_PyString' has not public schema information")

    def _arange(
        self,
        start: int | float,
        stop: int | float,
        step: int | float = 1,
    ) -> TyArrayBase:
        raise ValueError("'arange' should never be called on Python scalar")

    def _eye(
        self,
        n_rows: int,
        n_cols: int | None = None,
        /,
        *,
        k: int = 0,
    ) -> TyArrayBase:
        raise ValueError("'eye' should never be called on Python scalar")

    def _ones(self, shape: tuple[int, ...] | TyArrayInt64) -> TyArrayBase:
        raise ValueError("'ones' should never be called on Python scalar")

    def _zeros(self, shape: tuple[int, ...] | TyArrayInt64) -> TyArrayBase:
        raise ValueError("'zeros' should never be called on Python scalar")


class PyString(_PyScalar):
    def __ndx_result_type__(self, other: DType) -> DType | NotImplementedType:
        if isinstance(other, onnx.Utf8 | masked_onnx.NUtf8):
            return other
        return NotImplemented

    @property
    def _tyarr_class(self) -> type[TyArrayPyString]:
        return TyArrayPyString


class PyInteger(_PyScalar):
    def __ndx_result_type__(self, other: DType) -> DType | NotImplementedType:
        if isinstance(other, onnx.NumericDTypes | masked_onnx.NCoreNumericDTypes):
            return other
        if isinstance(other, onnx.Boolean):
            return ndx._default_int
        if isinstance(other, masked_onnx.NBoolean):
            return masked_onnx.as_nullable(ndx._default_int)
        return NotImplemented

    @property
    def _tyarr_class(self) -> type[TyArrayPyInt]:
        return TyArrayPyInt


class PyBool(PyInteger):
    def __ndx_result_type__(self, other: DType) -> DType | NotImplementedType:
        if isinstance(other, onnx.Boolean):
            return ndx.bool
        return super().__ndx_result_type__(other)

    @property
    def _tyarr_class(self) -> type[TyArrayPyBool]:
        return TyArrayPyBool


class PyFloat(_PyScalar):
    def __ndx_result_type__(self, other: DType) -> DType | NotImplementedType:
        if isinstance(other, onnx.IntegerDTypes):
            return onnx.float64
        if isinstance(other, masked_onnx.NCoreIntegerDTypes):
            return masked_onnx.nfloat64
        if isinstance(other, onnx.FloatingDTypes | masked_onnx.NCoreFloatingDTypes):
            return other
        return NotImplemented

    @property
    def _tyarr_class(self) -> type[TyArrayPyFloat]:
        return TyArrayPyFloat


# scalar singleton instances
pyint = PyInteger()
pyfloat = PyFloat()
pystring = PyString()
pybool = PyBool()


_BinaryOp = Callable[[TyArrayBase, TyArrayBase], TyArrayBase]


def _make_binary(
    op: _BinaryOp,
) -> tuple[_BinaryOp, _BinaryOp]:
    def binary_op_forward(self, other):
        return _promote_and_apply_op(self, other, op, True)

    def binary_op_backward(self, other):
        return _promote_and_apply_op(self, other, op, False)

    return binary_op_forward, binary_op_backward


class _ArrayPyScalar(TyArrayBase):
    """Array representing a Python scalar.

    This implementation is written as if it were a "custom" typed array which knows
    about other (nullable) core types. Those core types are oblivious of this typed
    array, though. Thus, this implementation may serve as a blue print for custom types.
    """

    dtype: PyFloat | PyInteger | PyString
    value: int | float | str

    def __ndx_value_repr__(self):
        return {"value": str(self.value)}

    def __getitem__(self, index) -> Self:
        raise IndexError(f"`{type(self)}` cannot be indexed")

    def __setitem__(
        self,
        key: SetitemIndex,
        value: Self,
        /,
    ) -> None:
        raise NotImplementedError

    @property
    def mT(self) -> Self:  # noqa: N802
        raise ValueError("Python scalars cannot be transposed")

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> OnnxShape:
        return ()

    @property
    def dynamic_shape(self) -> onnx.TyArrayInt64:
        raise ValueError("'dynamic_shape' should never be called on Python scalar")

    def copy(self) -> Self:
        return self

    def disassemble(self) -> dict[str, Var] | Var:
        raise ValueError(
            "Python scalars cannot be disassembled into 'spox.Var' objects"
        )

    def reshape(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> Self:
        raise ValueError("cannot 'reshape' Python scalar")

    def squeeze(self, /, axis: int | tuple[int, ...]) -> Self:
        raise ValueError("cannot 'squeeze' Python scalar")

    def broadcast_to(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> Self:
        raise ValueError("cannot 'broadcast_to' Python scalar")

    def where(self, cond: onnx.TyArrayBool, y: TyArrayBase) -> TyArrayBase:
        raise NotImplementedError

    def permute_dims(self, axes: tuple[int, ...]) -> Self:
        raise ValueError("cannot 'permute_dims' on Python scalar")

    def __ndx_cast_to__(
        self, dtype: DType[TY_ARRAY_BASE]
    ) -> TY_ARRAY_BASE | NotImplementedType:
        from .masked_onnx import asncoredata

        # mypy gets confused with the generic return type after the
        # narrowing so we put it aside here.
        res_ty = dtype._tyarr_class

        # We implement this class under the assumption that the other
        # built-in typed arrays do not know about it. Thus, we define
        # the mapping from this class into those classes
        # **here**.
        # TODO: This was probably not the best idea in hindsight...
        if isinstance(dtype, onnx.Utf8):
            # For strings we want to avoid using astype in order to
            # get value-dependent string sizes. If we use asarray,
            # int64s will always result in a `<U21` string.
            # We cannot use this for all types though, since we
            # otherwise get a hard error from NumPy when casting
            # overflowing integers.
            # TODO: Figure out why mypy things that `dtype` is `Never`...
            np_arr = np.asarray(self.value, dtype=dtype.unwrap_numpy())  # type: ignore
            return dtype._tyarr_class(onnx.const(np_arr).var)  # type: ignore
        if isinstance(dtype, onnx._OnnxDType):
            np_arr = np.asarray(self.value).astype(dtype.unwrap_numpy())
            return dtype._tyarr_class(onnx.const(np_arr).var)
        if isinstance(dtype, masked_onnx._MaOnnxDType):
            unmasked_typed_arr = self.__ndx_cast_to__(dtype._unmasked_dtype)
            return safe_cast(res_ty, asncoredata(unmasked_typed_arr, None))
        return NotImplemented

    def concat(self, others: list[Self], axis: None | int) -> Self:
        # Python scalars should never have to be concatenated
        raise NotImplementedError

    def _eqcomp(self, other) -> TyArrayBase:
        return _promote_and_apply_op(self, other, operator.eq, True)


class _ArrayPyScalarNumber(_ArrayPyScalar):
    __add__, __radd__ = _make_binary(operator.add)
    __eq__, _ = _make_binary(operator.eq)  # type: ignore
    __floordiv__, __rfloordiv__ = _make_binary(operator.floordiv)
    __ge__, _ = _make_binary(operator.ge)
    __gt__, __rgt__ = _make_binary(operator.gt)
    __le__, _ = _make_binary(operator.le)
    __lt__, _ = _make_binary(operator.lt)
    __matmul__, __rmatmul__ = _make_binary(operator.matmul)
    __mod__, __rmod__ = _make_binary(operator.mod)
    __mul__, __rmul__ = _make_binary(operator.mul)
    __ne__, _ = _make_binary(operator.ne)  # type: ignore
    __pow__, __rpow__ = _make_binary(operator.pow)
    __sub__, __rsub__ = _make_binary(operator.sub)
    __truediv__, __rtruediv__ = _make_binary(operator.truediv)


class TyArrayPyInt(_ArrayPyScalarNumber):
    dtype = pyint

    def __init__(self, value: int):
        # int is subclass of bool
        if not isinstance(value, int | bool):
            raise TypeError(f"expected 'int' or 'bool' found `{type(value)}`")
        self.value = value

    __and__, __rand__ = _make_binary(operator.and_)
    __lshift__, __rlshift__ = _make_binary(operator.lshift)
    __or__, __ror__ = _make_binary(operator.or_)
    __rshift__, __rrshift__ = _make_binary(operator.rshift)
    __xor__, __rxor__ = _make_binary(operator.xor)


class TyArrayPyBool(TyArrayPyInt):
    dtype = pybool


class TyArrayPyFloat(_ArrayPyScalarNumber):
    dtype = pyfloat

    def __init__(self, value: float):
        if not isinstance(value, float):
            raise TypeError(f"expected 'float' found `{type(value)}`")
        self.value = value


class TyArrayPyString(_ArrayPyScalar):
    dtype = pystring

    def __init__(self, value: str):
        if not isinstance(value, str):
            raise TypeError(f"expected 'str' found `{type(value)}`")
        self.value = value

    def __add__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.add, True)

    def __radd__(self, lhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, lhs, operator.add, False)


def _promote_and_apply_op(
    this: _ArrayPyScalar,
    other: TyArrayBase,
    arr_op: Callable[[Any, Any], Any],
    forward: bool,
) -> onnx.TyArray | masked_onnx.TyMaArray | NotImplementedType:
    if isinstance(other, onnx.TyArray | masked_onnx.TyMaArray):
        this_, other = promote(this, other)
        res = arr_op(this_, other) if forward else arr_op(other, this_)
        # I am not sure how to annotate `safe_cast` such that it can handle union types.
        return safe_cast(onnx.TyArray | masked_onnx.TyMaArray, res)  # type: ignore

    # We only know about the other (nullable) built-in types &
    # these scalars should never interact with themselves.
    return NotImplemented
