# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import spox.opset.ai.onnx.v21 as op
from spox import Var

from ..dtypes import DType, default_float, default_int, string
from .onnx import ascoredata
from .py_scalars import TyArrayPyFloat, TyArrayPyInt, TyArrayPyString
from .typed_array import TyArrayBase


def typed_where(cond: TyArrayBase, x: TyArrayBase, y: TyArrayBase) -> TyArrayBase:
    from . import TyArrayBool

    # TODO: Masked condition
    if not isinstance(cond, TyArrayBool):
        raise TypeError("'cond' must be a boolean data type.")

    ret = x.__ndx_where__(cond, y)
    if ret is NotImplemented:
        ret = y.__ndx_rwhere__(cond, x)
        if ret is NotImplemented:
            raise TypeError(
                f"Unsupported operand data types for 'where': `{x.dtype}` and `{y.dtype}`"
            )
    return ret


def astypedarray(
    val: int | float | str | np.ndarray | TyArrayBase | Var,
    dtype: None | DType = None,
    use_py_scalars=False,
) -> TyArrayBase:
    """Conversion of values of various types into a built-in typed array."""
    if isinstance(val, TyArrayBase):
        return val

    arr: TyArrayBase
    if isinstance(val, int):
        arr = TyArrayPyInt(val)
        arr = arr if use_py_scalars else arr.astype(default_int)
    elif isinstance(val, float):
        arr = TyArrayPyFloat(val)
        arr = arr if use_py_scalars else arr.astype(default_float)
    elif isinstance(val, str):
        arr = TyArrayPyString(val)
        arr = arr if use_py_scalars else arr.astype(string)
    elif isinstance(val, Var):
        arr = ascoredata(val)
    elif isinstance(val, np.ma.MaskedArray):
        raise NotImplementedError
    elif isinstance(val, np.ndarray):
        arr = ascoredata(op.const(val))
    else:
        raise ValueError

    if dtype is not None:
        return arr.astype(dtype)
    return arr


def maximum(x1: TyArrayBase, x2: TyArrayBase, /) -> TyArrayBase:
    res = x1.__ndx_maximum__(x2)
    if res is NotImplemented:
        res = x2.__ndx_rmaximum__(x1)
    if res is NotImplemented:
        raise TypeError(
            f"Unsupported operand data types for 'max': `{x1.dtype}` and `{x2.dtype}`"
        )
    return res


def sum(
    x: TyArrayBase,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: bool = False,
) -> TyArrayBase:
    return x.sum(axis=axis, dtype=dtype, keepdims=keepdims)
