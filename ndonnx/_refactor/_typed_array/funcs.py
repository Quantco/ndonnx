# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from functools import reduce
from types import NotImplementedType
from typing import TYPE_CHECKING, Literal, TypeVar, overload

import numpy as np
from spox import Var

import ndonnx._refactor as ndx

from .typed_array import TyArrayBase
from .utils import promote, safe_cast

if TYPE_CHECKING:
    from .. import Array
    from .._dtypes import TY_ARRAY_BASE, DType
    from . import onnx


@overload
def astyarray(
    val: bool | int | float | str | np.ndarray | TyArrayBase | Var | Array,
    dtype: DType[TY_ARRAY_BASE],
    use_py_scalars=False,
) -> TY_ARRAY_BASE: ...


@overload
def astyarray(
    val: bool | int | float | str | np.ndarray | TyArrayBase | Var | Array,
    dtype: None | DType = None,
    use_py_scalars=False,
) -> TyArrayBase: ...


def astyarray(
    val: bool | int | float | str | np.ndarray | TyArrayBase | Var | Array,
    dtype: None | DType[TY_ARRAY_BASE] = None,
    use_py_scalars=False,
) -> TyArrayBase:
    """Conversion of values of various types into a built-in typed array.

    This function always copies
    """
    from .. import Array
    from . import TyArrayBase, masked_onnx, onnx
    from .date_time import DateTime, TimeDelta, validate_unit
    from .py_scalars import TyArrayPyBool, TyArrayPyFloat, TyArrayPyInt, TyArrayPyString

    if isinstance(val, np.generic):
        val = np.array(val)

    if isinstance(val, TyArrayBase):
        return val.copy()

    if isinstance(val, Array):
        return val._tyarray.copy()

    arr: TyArrayBase
    if isinstance(val, bool):
        arr = TyArrayPyBool(val)
        arr = arr if use_py_scalars else arr.astype(ndx.bool)
    elif isinstance(val, int):
        arr = TyArrayPyInt(val)
        arr = arr if use_py_scalars else arr.astype(ndx._default_int)
    elif isinstance(val, float):
        arr = TyArrayPyFloat(val)
        arr = arr if use_py_scalars else arr.astype(ndx._default_float)
    elif isinstance(val, str):
        arr = TyArrayPyString(val)
        arr = arr if use_py_scalars else arr.astype(ndx.utf8)
    elif isinstance(val, Var):
        arr = onnx.ascoredata(val)
    elif isinstance(val, np.ma.MaskedArray):
        data = safe_cast(onnx.TyArray, astyarray(val.data))
        if val.mask is np.ma.nomask:
            mask = None
        else:
            mask = safe_cast(onnx.TyArrayBool, onnx.const(val.mask))
        arr = masked_onnx.asncoredata(data, mask)
    elif isinstance(val, np.ndarray):
        if val.dtype.kind == "O" and all(isinstance(el, str) for el in val.flat):
            arr = safe_cast(onnx.TyArrayUtf8, onnx.const(val.astype(str)))
        elif val.dtype.kind == "M":
            unit, count = np.datetime_data(val.dtype)
            unit = validate_unit(unit)
            if count != 1:
                raise ValueError(
                    "cannot create datetime with unit count other than '1'"
                )
            arr = astyarray(val.astype(np.int64)).astype(DateTime(unit=unit))
        elif val.dtype.kind == "m":
            unit, count = np.datetime_data(val.dtype)
            unit = validate_unit(unit)
            if count != 1:
                raise ValueError(
                    "cannot create timedelta with unit count other than '1'"
                )
            arr = astyarray(val.astype(np.int64)).astype(TimeDelta(unit=unit))
        else:
            arr = onnx.const(val)
    else:
        raise ValueError(f"failed to convert `{type(val)}` to typed array")

    if dtype is not None:
        return arr.astype(dtype, copy=True)
    return arr


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
def result_type(first: TyArrayBase | DType, *others: TyArrayBase | DType) -> DType: ...


def result_type(first: TyArrayBase | DType, *others: TyArrayBase | DType) -> DType:
    def get_dtype(obj: TyArrayBase | DType) -> DType:
        if isinstance(obj, TyArrayBase):
            return obj.dtype
        return obj

    return _result_dtype(get_dtype(first), *(get_dtype(el) for el in others))


def _result_dtype(first: DType, *others: DType) -> DType:
    def result_binary(a: DType, b: DType) -> DType:
        if a == b:
            return a
        res1 = a.__ndx_result_type__(b)
        if res1 != NotImplemented:
            return res1
        return b.__ndx_result_type__(a)

    res = reduce(result_binary, others, first)
    if res == NotImplemented:
        raise TypeError("No common type found")
    return res


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


def _validate(
    x1: TyArrayBase, x2: TyArrayBase, result: T | NotImplementedType, func_name: str
) -> T:
    if isinstance(result, NotImplementedType):
        raise TypeError(
            f"Unsupported operand data types for '{func_name}': `{x1.dtype}` and `{x2.dtype}`"
        )
    return result


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


def maximum(x1: TyArrayBase, x2: TyArrayBase, /) -> TyArrayBase:
    res = x1.__ndx_maximum__(x2)
    if res is NotImplemented:
        res = x2.__ndx_maximum__(x1)
    return _validate(x1, x2, res, "maximum")


def minimum(x1: TyArrayBase, x2: TyArrayBase, /) -> TyArrayBase:
    res = x1.__ndx_minimum__(x2)
    if res is NotImplemented:
        res = x2.__ndx_minimum__(x1)
    return _validate(x1, x2, res, "minimum")
