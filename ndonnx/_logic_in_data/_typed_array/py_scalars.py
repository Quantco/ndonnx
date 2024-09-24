# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import operator
from collections.abc import Callable
from types import NotImplementedType
from typing import TYPE_CHECKING, Any

import numpy as np
import spox.opset.ai.onnx.v21 as op
from typing_extensions import Self

import ndonnx._logic_in_data as ndx

from .. import dtypes
from ..dtypes import TY_ARRAY, DType
from . import TyArray, TyArrayInteger, TyMaArray, masked_onnx, onnx
from .typed_array import TyArrayBase
from .utils import promote, safe_cast

if TYPE_CHECKING:
    from ..array import OnnxShape
    from ..schema import Components, DTypeInfo, Schema
    from .indexing import SetitemIndex


class _PyScalar(DType):
    def __ndx_convert_tyarray__(self, arr: TyArrayBase) -> Self:
        raise NotImplementedError

    def _argument(self, shape: OnnxShape):
        raise ValueError("'{type(self)}' cannot be used as a model argument")

    @property
    def _info(self) -> DTypeInfo:
        raise ValueError("'_PyString' has not public schema information")


class PyString(_PyScalar):
    def _result_type(self, other: DType) -> DType | NotImplementedType:
        if isinstance(other, onnx.String | masked_onnx.NString):
            return other
        return NotImplemented

    @property
    def _tyarr_class(self) -> type[TyArrayPyString]:
        return TyArrayPyString


class PyInteger(_PyScalar):
    def _result_type(self, other: DType) -> DType | NotImplementedType:
        if isinstance(other, onnx.CoreNumericDTypes | masked_onnx.NCoreNumericDTypes):
            return other
        if isinstance(other, onnx.Bool):
            return ndx._default_int
        if isinstance(other, masked_onnx.NBool):
            return masked_onnx.as_nullable(ndx._default_int)
        return NotImplemented

    @property
    def _tyarr_class(self) -> type[TyArrayPyInt]:
        return TyArrayPyInt


class PyFloat(_PyScalar):
    def _result_type(self, other: DType) -> DType | NotImplementedType:
        if isinstance(other, onnx.CoreIntegerDTypes):
            return onnx.float64
        if isinstance(other, masked_onnx.NCoreIntegerDTypes):
            return masked_onnx.nfloat64
        if isinstance(other, onnx.CoreFloatingDTypes | masked_onnx.NCoreFloatingDTypes):
            return other
        raise ValueError

    @property
    def _tyarr_class(self) -> type[TyArrayPyFloat]:
        return TyArrayPyFloat


# scalar singleton instances
pyint = PyInteger()
pyfloat = PyFloat()
pystring = PyString()


class _ArrayPyScalar(TyArrayBase):
    """Array representing a Python scalar.

    This implementation is written as if it were a "custom" typed array which knows
    about other (nullable) core types. Those core types are oblivious of this typed
    array, though. Thus, this implementation may serve as a blue print for custom types.
    """

    dtype: PyFloat | PyInteger | PyString
    value: int | float | str

    def __getitem__(self, index) -> Self:
        raise IndexError(f"`{type(self)}` cannot be indexed")

    def __setitem__(
        self,
        key: SetitemIndex,
        value: Self,
        /,
    ) -> None:
        raise NotImplementedError

    def disassemble(self) -> tuple[Components, Schema]:
        raise ValueError(
            "Python scalars cannot be disassembled into 'spox.Var' objects"
        )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> OnnxShape:
        return ()

    @property
    def dynamic_shape(self) -> onnx.TyArrayInt64:
        raise ValueError("'dynamic_shape' should never be called on Python scalar")

    def reshape(self, shape: tuple[int, ...]) -> Self:
        raise ValueError("cannot reshape Python scalar")

    def broadcast_to(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> Self:
        raise ValueError("cannot broadcast Python scalar")

    def where(self, cond: onnx.TyArrayBool, y: TyArrayBase) -> TyArrayBase:
        raise NotImplementedError

    def __ndx_astype__(self, dtype: DType[TY_ARRAY]) -> TY_ARRAY | NotImplementedType:
        from . import asncoredata

        # mypy gets confused with the generic return type after the
        # narrowing so we put it aside here.
        res_ty = dtype._tyarr_class

        # We implement this class under the assumption that the other
        # built-in typed arrays do not know about it. Thus, we define
        # the mapping from this class into those classes **here**.
        if isinstance(dtype, onnx.CoreDTypes):
            np_arr = np.array(self.value, dtypes.as_numpy(dtype))
            return safe_cast(res_ty, dtype._tyarr_class(op.const(np_arr)))
        if isinstance(dtype, masked_onnx.NCoreDTypes):
            unmasked_typed_arr = self.__ndx_astype__(dtype._unmasked_dtype)
            return safe_cast(res_ty, asncoredata(unmasked_typed_arr, None))
        return NotImplemented

    def _eqcomp(self, other) -> TyArrayBase:
        return _promote_and_apply_op(self, other, operator.eq, True)


class _ArrayPyScalarNumber(_ArrayPyScalar):
    def __add__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.add, True)

    def __radd__(self, lhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, lhs, operator.add, False)

    def __le__(self, rhs: TyArrayBase, /) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.le, False)

    def __lt__(self, rhs: TyArrayBase, /) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.lt, False)

    def __ge__(self, rhs: TyArrayBase, /) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.ge, False)

    def __gt__(self, rhs: TyArrayBase, /) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.gt, False)

    def __mul__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.mul, True)

    def __rmul__(self, lhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, lhs, operator.mul, False)

    def __sub__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.sub, True)

    def __rsub__(self, lhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, lhs, operator.sub, False)


class TyArrayPyInt(_ArrayPyScalarNumber):
    dtype = pyint

    def __init__(self, value: int):
        # int is subclass of bool
        if not isinstance(value, int | bool):
            raise TypeError(f"expected 'int' or 'bool' found `{type(value)}`")
        self.value = value

    def __or__(self, rhs: TyArrayBase) -> TyArray | TyMaArray:
        if isinstance(rhs, TyArrayInteger):
            return _promote_and_apply_op(self, rhs, operator.or_, True)
        return NotImplemented


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
) -> TyArray | TyMaArray | NotImplementedType:
    if isinstance(other, TyArray | TyMaArray):
        this_, other = promote(this, other)
        res = arr_op(this_, other) if forward else arr_op(other, this_)
        # I am not sure how to annotate `safe_cast` such that it can handle union types.
        return safe_cast(TyArray | TyMaArray, res)  # type: ignore

    # We only know about the other (nullable) built-in types &
    # these scalars should never interact with themselves.
    return NotImplemented
