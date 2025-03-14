# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from types import NotImplementedType
from typing import TYPE_CHECKING, Literal, TypeVar, overload

import numpy as np
from spox import Var

from .._dtypes import DType
from .._types import NestedSequence, PyScalar
from .typed_array import TyArrayBase
from .utils import promote

if TYPE_CHECKING:
    from .. import Array
    from .._dtypes import TY_ARRAY_BASE
    from . import onnx


def _infer_sequence(
    val: NestedSequence,
) -> DType:
    from . import onnx

    types = set()
    for item in val:
        if isinstance(item, Sequence) and not isinstance(item, str):
            types.add(_infer_sequence(item))
        else:
            types.add(_infer_dtype(item))
    if len(types) == 0:
        return onnx.float64
    return _result_dtype(*types)


def _infer_dtype(
    val: PyScalar | np.ndarray | TyArrayBase | Var | NestedSequence,
) -> DType:
    from . import masked_onnx, onnx

    if isinstance(val, np.ndarray):
        if val.dtype == object:
            if not all(isinstance(el, str) for el in val.flatten()):
                raise ValueError(
                    "found array with object data type but it contains non-string elements"
                )
            return onnx.utf8
        core_type = onnx.from_numpy(val.dtype)
        if isinstance(val, np.ma.MaskedArray):
            return masked_onnx.to_nullable_dtype(core_type)
        else:
            return core_type
    elif isinstance(val, TyArrayBase):
        return val.dtype
    elif isinstance(val, Var):
        return onnx.from_numpy(val.unwrap_tensor().dtype)
    elif isinstance(val, bool):
        return onnx.bool_
    elif isinstance(val, int):
        return onnx.int64
    elif isinstance(val, float):
        return onnx.float64
    elif isinstance(val, str):
        return onnx.utf8
    elif isinstance(val, Sequence):
        return _infer_sequence(val)
    elif isinstance(val, np.generic):
        return onnx.from_numpy(val.dtype)
    else:
        raise ValueError(f"Unable to infer dtype from {val}")


@overload
def astyarray(
    val: PyScalar | np.ndarray | TyArrayBase | Var | Array | NestedSequence,
    dtype: DType[TY_ARRAY_BASE],
) -> TY_ARRAY_BASE: ...


@overload
def astyarray(
    val: PyScalar | np.ndarray | TyArrayBase | Var | Array | NestedSequence,
    dtype: None | DType = None,
) -> TyArrayBase: ...


def astyarray(
    val: PyScalar | np.ndarray | TyArrayBase | Var | Array | NestedSequence,
    dtype: None | DType[TY_ARRAY_BASE] = None,
) -> TyArrayBase:
    """Conversion of values of various types into a built-in typed array.

    This function always copies
    """
    from .. import Array

    val = val if not isinstance(val, Array) else val._tyarray
    inferred_dtype = _infer_dtype(val) if dtype is None else dtype
    return inferred_dtype.__ndx_create__(val)


def concat(
    arrays: tuple[TyArrayBase, ...] | list[TyArrayBase], /, *, axis: None | int = 0
) -> TyArrayBase:
    first, *others = arrays
    dtype = reduce(
        lambda dtype, arr: result_type(dtype, arr.dtype), others, first.dtype
    )
    arrays = [arr.astype(dtype) for arr in arrays]
    return arrays[0].concat(arrays[1:], axis=axis)


@overload
def result_type(
    first: onnx.NumericDTypes, *others: onnx.NumericDTypes
) -> onnx.NumericDTypes: ...


@overload
def result_type(first: onnx.DTypes, *others: onnx.DTypes) -> onnx.DTypes: ...


@overload
def result_type(first: DType, *others: DType) -> DType: ...


@overload
def result_type(
    first: TyArrayBase | DType, *others: TyArrayBase | DType | PyScalar
) -> DType: ...


def result_type(
    first: TyArrayBase | DType, *others: TyArrayBase | DType | PyScalar
) -> DType:
    def get_dtype(obj: TyArrayBase | DType) -> DType:
        if isinstance(obj, TyArrayBase):
            return obj.dtype
        return obj

    def get_dtype_or_scalar(obj: TyArrayBase | DType | PyScalar) -> DType | PyScalar:
        if isinstance(obj, PyScalar):
            return obj
        if isinstance(obj, TyArrayBase):
            return obj.dtype
        return obj

    if len(others) == 0:
        return get_dtype(first)

    return _result_dtype(get_dtype(first), *(get_dtype_or_scalar(el) for el in others))


def _result_dtype(first: DType, *others: DType | PyScalar) -> DType:
    def result_binary(a: DType, b: DType | PyScalar) -> DType:
        if a == b:
            return a
        res1 = a.__ndx_result_type__(b)
        if res1 != NotImplemented:
            return res1
        if not isinstance(b, PyScalar):
            res2 = b.__ndx_result_type__(a)
            if res2 != NotImplemented:
                return res2
        b_ty = b if isinstance(b, DType) else type(b)
        raise TypeError(f"no common type found for `{a}` and `{b_ty}`")

    return reduce(result_binary, others, first)


def searchsorted(
    x1: TyArrayBase,
    x2: TyArrayBase,
    /,
    *,
    side: Literal["left", "right"] = "left",
    sorter: onnx.TyArrayInteger | None = None,
) -> onnx.TyArrayInt64:
    x1, x2 = promote(x1, x2)
    return x1.searchsorted(x2, side=side, sorter=sorter)


#########################################################################
# Free functions implemented via `__ndx_*__` methods on the typed array #
#########################################################################

T = TypeVar("T")
TYARRAY = TypeVar("TYARRAY", bound="onnx.TyArray")


def _validate(
    x1: TyArrayBase, x2: TyArrayBase, result: T | NotImplementedType, func_name: str
) -> T:
    if isinstance(result, NotImplementedType):
        raise TypeError(
            f"Unsupported operand data types for '{func_name}': `{x1.dtype}` and `{x2.dtype}`"
        )
    return result


@overload
def where(cond: onnx.TyArrayBool, x: TYARRAY, y: TYARRAY) -> TYARRAY: ...


@overload
def where(cond: onnx.TyArrayBool, x: TyArrayBase, y: TyArrayBase) -> TyArrayBase: ...


def where(cond: onnx.TyArrayBool, x: TyArrayBase, y: TyArrayBase) -> TyArrayBase:
    from . import onnx

    if not isinstance(cond, onnx.TyArrayBool):
        raise TypeError("'cond' must be a boolean data type.")

    res = x.__ndx_where__(cond, y)
    if res is NotImplemented:
        res = y.__ndx_rwhere__(cond, x)
    return _validate(x, y, res, "where")


def logical_and(x1: TyArrayBase, x2: TyArrayBase, /) -> TyArrayBase:
    res = x1.__ndx_logical_and__(x2)
    if res is NotImplemented:
        res = x2.__ndx_rlogical_and__(x1)
    return _validate(x1, x2, res, "logical_and")


def logical_or(x1: TyArrayBase, x2: TyArrayBase, /) -> TyArrayBase:
    res = x1.__ndx_logical_or__(x2)
    if res is NotImplemented:
        res = x2.__ndx_rlogical_or__(x1)
    return _validate(x1, x2, res, "logical_or")


def logical_xor(x1: TyArrayBase, x2: TyArrayBase, /) -> TyArrayBase:
    res = x1.__ndx_logical_xor__(x2)
    if res is NotImplemented:
        res = x2.__ndx_rlogical_xor__(x1)
    return _validate(x1, x2, res, "logical_xor")


@overload
def maximum(x1: TYARRAY, x2: TYARRAY, /) -> TYARRAY: ...
@overload
def maximum(x1: TyArrayBase, x2: TyArrayBase, /) -> TyArrayBase: ...


def maximum(x1: TyArrayBase, x2: TyArrayBase, /) -> TyArrayBase:
    res = x1.__ndx_maximum__(x2)
    if res is NotImplemented:
        res = x2.__ndx_maximum__(x1)
    return _validate(x1, x2, res, "maximum")


@overload
def minimum(x1: TYARRAY, x2: TYARRAY, /) -> TYARRAY: ...
@overload
def minimum(x1: TyArrayBase, x2: TyArrayBase, /) -> TyArrayBase: ...


def minimum(x1: TyArrayBase, x2: TyArrayBase, /) -> TyArrayBase:
    res = x1.__ndx_minimum__(x2)
    if res is NotImplemented:
        res = x2.__ndx_minimum__(x1)
    return _validate(x1, x2, res, "minimum")


def arange(
    dtype: DType[TY_ARRAY_BASE],
    start: int | float,
    stop: int | float,
    step: int | float = 1,
) -> TY_ARRAY_BASE:
    res = dtype.__ndx_arange__(start, stop, step)
    if res is NotImplemented:
        raise TypeError(f"'arange' is not implemented for `{dtype}`")
    return res


def eye(
    dtype: DType[TY_ARRAY_BASE],
    n_rows: int,
    n_cols: int | None = None,
    /,
    *,
    k: int = 0,
) -> TY_ARRAY_BASE:
    res = dtype.__ndx_eye__(n_rows, n_cols, k=k)
    if res is NotImplemented:
        raise TypeError(f"'eye' is not implemented for `{dtype}`")
    return res


def ones(
    dtype: DType[TY_ARRAY_BASE], shape: tuple[int, ...] | onnx.TyArrayInt64
) -> TY_ARRAY_BASE:
    res = dtype.__ndx_ones__(shape)
    if res is NotImplemented:
        raise TypeError(f"'ones' is not implemented for `{dtype}`")
    return res


def zeros(
    dtype: DType[TY_ARRAY_BASE], shape: tuple[int, ...] | onnx.TyArrayInt64
) -> TY_ARRAY_BASE:
    res = dtype.__ndx_zeros__(shape)
    if res is NotImplemented:
        raise TypeError(f"'zeros' is not implemented for `{dtype}`")
    return res
