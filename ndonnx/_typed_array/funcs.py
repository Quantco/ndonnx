# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from types import NotImplementedType
from typing import Literal, TypeVar, overload

import numpy as np
from spox import Var

from ndonnx import DType, from_numpy_dtype
from ndonnx.types import NestedSequence, PyScalar

from . import TyArrayBase, datetime, masked_onnx, onnx, promote
from ._utils import validate_op_result

TY_ARRAY_BASE_co = TypeVar("TY_ARRAY_BASE_co", bound="TyArrayBase", covariant=True)


def _infer_sequence(
    val: NestedSequence,
) -> DType:
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
    if isinstance(val, np.ndarray):
        if val.dtype == object:
            if not all(isinstance(el, str) for el in val.flatten()):
                raise ValueError(
                    "found array with object data type but it contains non-string elements"
                )
            return onnx.utf8
        dtype = from_numpy_dtype(val.dtype)
        if isinstance(val, np.ma.MaskedArray):
            if isinstance(dtype, datetime.BaseTimeDType):
                return dtype
            return masked_onnx.to_nullable_dtype(dtype)
        return dtype
    elif isinstance(val, TyArrayBase):
        return val.dtype
    elif isinstance(val, Var):
        return onnx.from_numpy_dtype(val.unwrap_tensor().dtype)
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
        return onnx.from_numpy_dtype(val.dtype)
    else:
        raise ValueError(f"unable to infer dtype from `{val}`")


@overload
def astyarray(
    val: PyScalar | np.ndarray | TyArrayBase | Var | NestedSequence,
    dtype: DType[TY_ARRAY_BASE_co],
) -> TY_ARRAY_BASE_co: ...


@overload
def astyarray(
    val: PyScalar | np.ndarray | TyArrayBase | Var | NestedSequence,
    dtype: None | DType = None,
) -> TyArrayBase: ...


def astyarray(
    val: PyScalar | np.ndarray | TyArrayBase | Var | NestedSequence,
    dtype: None | DType[TY_ARRAY_BASE_co] = None,
) -> TyArrayBase:
    """Conversion of values of various types into a built-in typed array.

    This function always copies.
    """
    inferred_dtype = _infer_dtype(val) if dtype is None else dtype
    res = inferred_dtype.__ndx_create__(val)
    if res is NotImplemented:
        raise ValueError(
            f"failed to instantiate array of dtype `{inferred_dtype}` from `{val}`"
        )
    return res


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


@overload
def where(cond: onnx.TyArrayBool, x: TYARRAY, y: TYARRAY) -> TYARRAY: ...


@overload
def where(
    cond: onnx.TyArrayBool, x: TyArrayBase | PyScalar, y: TyArrayBase | PyScalar
) -> TyArrayBase: ...


def where(
    cond: onnx.TyArrayBool, x: TyArrayBase | PyScalar, y: TyArrayBase | PyScalar
) -> TyArrayBase:
    if not isinstance(cond, onnx.TyArrayBool):
        raise TypeError("'cond' must be a boolean data type.")

    res: NotImplementedType | TyArrayBase = NotImplemented
    if isinstance(x, TyArrayBase):
        res = x.__ndx_where__(cond, y)
    if res is NotImplemented and isinstance(y, TyArrayBase):
        res = y.__ndx_rwhere__(cond, x)
    return validate_op_result(x, y, res, "where")


def logaddexp(
    x1: TyArrayBase | int | float, x2: TyArrayBase | int | float, /
) -> TyArrayBase:
    res: NotImplementedType | TyArrayBase = NotImplemented

    if isinstance(x1, TyArrayBase):
        res = x1.__ndx_logaddexp__(x2)
    if res is NotImplemented and isinstance(x2, TyArrayBase):
        res = x2.__ndx_logaddexp__(x1)
    return validate_op_result(x1, x2, res, "logaddexp")


def logical_and(x1: TyArrayBase | bool, x2: TyArrayBase | bool, /) -> TyArrayBase:
    res: TyArrayBase | NotImplementedType = NotImplemented
    if isinstance(x1, TyArrayBase):
        res = x1.__ndx_logical_and__(x2)
    if res is NotImplemented and isinstance(x2, TyArrayBase):
        res = x2.__ndx_rlogical_and__(x1)

    return validate_op_result(x1, x2, res, "logical_and")


def logical_or(x1: TyArrayBase | bool, x2: TyArrayBase | bool, /) -> TyArrayBase:
    res: TyArrayBase | NotImplementedType = NotImplemented
    if isinstance(x1, TyArrayBase):
        res = x1.__ndx_logical_or__(x2)
    if res is NotImplemented and isinstance(x2, TyArrayBase):
        res = x2.__ndx_rlogical_or__(x1)

    return validate_op_result(x1, x2, res, "logical_or")


def logical_xor(x1: TyArrayBase | bool, x2: TyArrayBase | bool, /) -> TyArrayBase:
    res: TyArrayBase | NotImplementedType = NotImplemented
    if isinstance(x1, TyArrayBase):
        res = x1.__ndx_logical_xor__(x2)
    if res is NotImplemented and isinstance(x2, TyArrayBase):
        res = x2.__ndx_rlogical_xor__(x1)

    return validate_op_result(x1, x2, res, "logical_xor")


@overload
def maximum(x1: TYARRAY, x2: TYARRAY, /) -> TYARRAY: ...
@overload
def maximum(
    x1: TyArrayBase | int | float, x2: TyArrayBase | int | float, /
) -> TyArrayBase: ...


def maximum(
    x1: TyArrayBase | int | float, x2: TyArrayBase | int | float, /
) -> TyArrayBase:
    res: NotImplementedType | TyArrayBase = NotImplemented
    if isinstance(x1, TyArrayBase):
        res = x1.__ndx_maximum__(x2)
    if res is NotImplemented and isinstance(x2, TyArrayBase):
        res = x2.__ndx_maximum__(x1)
    return validate_op_result(x1, x2, res, "maximum")


@overload
def minimum(x1: TYARRAY, x2: TYARRAY, /) -> TYARRAY: ...
@overload
def minimum(
    x1: TyArrayBase | int | float, x2: TyArrayBase | int | float, /
) -> TyArrayBase: ...


def minimum(
    x1: TyArrayBase | int | float, x2: TyArrayBase | int | float, /
) -> TyArrayBase:
    res: NotImplementedType | TyArrayBase = NotImplemented
    if isinstance(x1, TyArrayBase):
        res = x1.__ndx_minimum__(x2)
    if res is NotImplemented and isinstance(x2, TyArrayBase):
        res = x2.__ndx_minimum__(x1)
    return validate_op_result(x1, x2, res, "minimum")


def arange(
    dtype: DType[TY_ARRAY_BASE_co],
    start: int | float,
    stop: int | float,
    step: int | float = 1,
) -> TY_ARRAY_BASE_co:
    res = dtype.__ndx_arange__(start, stop, step)
    if res is NotImplemented:
        raise TypeError(f"'arange' is not implemented for `{dtype}`")
    return res


def eye(
    dtype: DType[TY_ARRAY_BASE_co],
    n_rows: int,
    n_cols: int | None = None,
    /,
    *,
    k: int = 0,
) -> TY_ARRAY_BASE_co:
    res = dtype.__ndx_eye__(n_rows, n_cols, k=k)
    if res is NotImplemented:
        raise TypeError(f"'eye' is not implemented for `{dtype}`")
    return res


def ones(
    dtype: DType[TY_ARRAY_BASE_co], shape: tuple[int, ...] | onnx.TyArrayInt64
) -> TY_ARRAY_BASE_co:
    res = dtype.__ndx_ones__(shape)
    if res is NotImplemented:
        raise TypeError(f"'ones' is not implemented for `{dtype}`")
    return res


def zeros(
    dtype: DType[TY_ARRAY_BASE_co], shape: tuple[int, ...] | onnx.TyArrayInt64
) -> TY_ARRAY_BASE_co:
    res = dtype.__ndx_zeros__(shape)
    if res is NotImplemented:
        raise TypeError(f"'zeros' is not implemented for `{dtype}`")
    return res
