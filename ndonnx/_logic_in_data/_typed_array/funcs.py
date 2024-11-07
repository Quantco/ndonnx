# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
from functools import reduce
from itertools import chain
from typing import Literal

import numpy as np
import spox.opset.ai.onnx.v21 as op
from spox import Var

import ndonnx._logic_in_data as ndx

from .._dtypes import DType
from . import masked_onnx, onnx
from .py_scalars import TyArrayPyFloat, TyArrayPyInt, TyArrayPyString
from .typed_array import TyArrayBase
from .utils import promote, safe_cast


def astyarray(
    val: int | float | str | np.ndarray | TyArrayBase | Var,
    dtype: None | DType = None,
    use_py_scalars=False,
) -> TyArrayBase:
    """Conversion of values of various types into a built-in typed array."""
    from .onnx import TyArray

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
        data = safe_cast(TyArray, astyarray(val.data))
        if val.mask is np.ma.nomask:
            mask = None
        else:
            mask = safe_cast(onnx.TyArrayBool, onnx.ascoredata(op.const(val.mask)))
        arr = masked_onnx.asncoredata(data, mask)
    elif isinstance(val, np.ndarray):
        if val.dtype.kind == "O" and all(isinstance(el, str) for el in val.flat):
            val = val.astype(str)
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


def result_type(first: TyArrayBase | DType, *others: TyArrayBase | DType) -> DType:
    from .. import _dtypes

    def get_dtype(obj: TyArrayBase | DType) -> DType:
        if isinstance(obj, TyArrayBase):
            return obj.dtype
        return obj

    return reduce(
        lambda a, b: _dtypes.result_type(a, b),
        (get_dtype(el) for el in chain([first], others)),
    )


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


def where(cond: onnx.TyArrayBool, x: TyArrayBase, y: TyArrayBase) -> TyArrayBase:
    # TODO: Masked condition?
    if not isinstance(cond, onnx.TyArrayBool):
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
