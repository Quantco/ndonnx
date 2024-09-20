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

from .. import dtypes
from ..dtypes import DType
from .core import TyArray, TyArrayInteger
from .masked import TyMaArray
from .typed_array import TyArrayBase
from .utils import promote, safe_cast

if TYPE_CHECKING:
    from ..array import OnnxShape
    from ..schema import Components, Schema
    from .core import TyArrayBool, TyArrayInt64


class _ArrayPyScalar(TyArrayBase):
    """Array representing a Python scalar.

    This implementation is written as if it were a "custom" typed array which knows
    about other (nullable) core types. Those core types are oblivious of this typed
    array, though. Thus, this implementation may serve as a blue print for custom types.
    """

    dtype: dtypes._PyFloat | dtypes._PyInt
    value: int | float

    def __getitem__(self, index) -> Self:
        raise IndexError(f"`{type(self)}` cannot be indexed")

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
    def dynamic_shape(self) -> TyArrayInt64:
        raise ValueError("'dynamic_shape' should never be called on Python scalar")

    def reshape(self, shape: tuple[int, ...]) -> Self:
        raise ValueError("cannot reshape Python scalar")

    def broadcast_to(self, shape: tuple[int, ...] | TyArrayInt64) -> Self:
        raise ValueError("cannot broadcast Python scalar")

    def __add__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.add, True)

    def __radd__(self, lhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, lhs, operator.add, False)

    def __mul__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.mul, True)

    def __rmul__(self, lhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, lhs, operator.mul, False)

    def __sub__(self, rhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, rhs, operator.sub, True)

    def __rsub__(self, lhs: TyArrayBase) -> TyArrayBase:
        return _promote_and_apply_op(self, lhs, operator.sub, False)

    def _astype(self, dtype: DType) -> TyArrayBase | NotImplementedType:
        from .masked import asncoredata

        # We implement this class under the assumption that the other
        # built-in typed arrays do not know about it. Thus, we define
        # the mapping from this class into those classes **here**.
        if isinstance(dtype, dtypes.CoreNumericDTypes):
            np_arr = np.array(self.value, dtypes.as_numpy(dtype))
            return dtype._tyarr_class(op.const(np_arr))
        if isinstance(dtype, dtypes.NCoreNumericDTypes):
            unmasked_typed_arr = self._astype(dtype._unmasked_dtype)
            return asncoredata(unmasked_typed_arr, None)
        return NotImplemented

    def _eqcomp(self, other) -> TyArrayBase:
        return _promote_and_apply_op(self, other, operator.eq, True)

    def where(self, cond: TyArrayBool, y: TyArrayBase) -> TyArrayBase:
        raise NotImplementedError


class _ArrayPyInt(_ArrayPyScalar):
    dtype = dtypes._pyint

    def __init__(self, value: int):
        # int is subclass of bool
        if not isinstance(value, int | bool):
            raise TypeError(f"expected 'int' or 'bool' found `{type(value)}`")
        self.value = value

    def __or__(self, rhs: TyArrayBase) -> TyArray | TyMaArray:
        if isinstance(rhs, TyArrayInteger):
            return _promote_and_apply_op(self, rhs, operator.or_, True)
        return NotImplemented


class _ArrayPyFloat(_ArrayPyScalar):
    dtype = dtypes._pyfloat

    def __init__(self, value: float):
        if not isinstance(value, float):
            raise TypeError(f"expected 'float' found `{type(value)}`")
        self.value = value


def _promote_and_apply_op(
    this: _ArrayPyScalar,
    other: TyArrayBase,
    arr_op: Callable[[Any, Any], Any],
    forward: bool,
) -> TyArray | TyMaArray | NotImplementedType:
    if isinstance(other, TyArray | TyMaArray):
        try:
            this_, other = promote(this, other)
        except Exception:
            breakpoint()
        # I am not sure how to annotate `safe_cast` such that it can handle union types.
        res = arr_op(this_, other) if forward else arr_op(other, this_)
        return safe_cast(TyArray | TyMaArray, res)  # type: ignore

    # We only know about the other (nullable) built-in types &
    # these scalars should never interact with themselves.
    return NotImplemented
