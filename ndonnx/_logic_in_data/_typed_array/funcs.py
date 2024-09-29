# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
from functools import reduce

import numpy as np
import spox.opset.ai.onnx.v21 as op
from spox import Var

import ndonnx._logic_in_data as ndx

from ..dtypes import DType
from . import masked_onnx, onnx
from .py_scalars import TyArrayPyFloat, TyArrayPyInt, TyArrayPyString
from .typed_array import TyArrayBase
from .utils import safe_cast


def astyarray(
    val: int | float | str | np.ndarray | TyArrayBase | Var,
    dtype: None | DType = None,
    use_py_scalars=False,
) -> TyArrayBase:
    """Conversion of values of various types into a built-in typed array."""
    if isinstance(val, TyArrayBase):
        return val

    arr: TyArrayBase
    if isinstance(val, bool):
        arr = TyArrayPyInt(val)
        arr = arr if use_py_scalars else arr.astype(ndx.bool)
    elif isinstance(val, int):
        arr = TyArrayPyInt(val)
        arr = arr if use_py_scalars else arr.astype(ndx._default_int)
    elif isinstance(val, float):
        arr = TyArrayPyFloat(val)
        arr = arr if use_py_scalars else arr.astype(ndx._default_float)
    elif isinstance(val, str):
        arr = TyArrayPyString(val)
        arr = arr if use_py_scalars else arr.astype(ndx.string)
    elif isinstance(val, Var):
        arr = onnx.ascoredata(val)
    elif isinstance(val, np.ma.MaskedArray):
        data = onnx.ascoredata(op.const(val.data))
        if val.mask is np.ma.nomask:
            mask = None
        else:
            mask = safe_cast(onnx.TyArrayBool, onnx.ascoredata(op.const(val.mask)))
        arr = masked_onnx.asncoredata(data, mask)
    elif isinstance(val, np.ndarray):
        arr = onnx.ascoredata(op.const(val))
    else:
        breakpoint()
        raise ValueError

    if dtype is not None:
        return arr.astype(dtype)
    return arr


def concat(
    arrays: tuple[TyArrayBase, ...] | list[TyArrayBase], /, *, axis: None | int = 0
) -> TyArrayBase:
    first, *others = arrays
    dtype = reduce(
        lambda dtype, arr: dtype._result_type(arr.dtype), others, first.dtype
    )
    arrays = [arr.astype(dtype) for arr in arrays]
    return arrays[0].concat(arrays[1:], axis=axis)


#########################################################################
# Free functions implemented via `__ndx_*__` methods on the typed array #
#########################################################################


def where(cond: TyArrayBase, x: TyArrayBase, y: TyArrayBase) -> TyArrayBase:
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


def maximum(x1: TyArrayBase, x2: TyArrayBase, /) -> TyArrayBase:
    res = x1.__ndx_maximum__(x2)
    if res is NotImplemented:
        res = x2.__ndx_rmaximum__(x1)
    if res is NotImplemented:
        raise TypeError(
            f"Unsupported operand data types for 'maximum': `{x1.dtype}` and `{x2.dtype}`"
        )
    return res
