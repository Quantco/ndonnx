# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from spox import Var

from ..dtypes import DType, default_float, default_int, from_numpy
from .py_scalars import _ArrayPyFloat, _ArrayPyInt
from .typed_array import TyArrayBase


def typed_where(cond: TyArrayBase, x: TyArrayBase, y: TyArrayBase) -> TyArrayBase:
    from .core import TyArrayBool

    # TODO: Masked condition
    if not isinstance(cond, TyArrayBool):
        raise TypeError("'cond' must be a boolean data type.")

    ret = x._where(cond, y)
    if ret == NotImplemented:
        ret = y._rwhere(cond, x)
        if ret == NotImplemented:
            raise TypeError(
                f"Unsuppoerted operand data types for 'where': `{x.dtype}` and `{y.dtype}`"
            )
    return ret


def astypedarray(
    val: int | float | np.ndarray | TyArrayBase | Var,
    dtype: None | DType = None,
    use_py_scalars=False,
) -> TyArrayBase:
    """Conversion of values of various types into a built-in typed array."""
    if isinstance(val, TyArrayBase):
        return val

    arr: TyArrayBase
    if isinstance(val, int):
        arr = _ArrayPyInt(val)
        arr = arr if use_py_scalars else arr.astype(default_int)
    elif isinstance(val, float):
        arr = _ArrayPyFloat(val)
        arr = arr if use_py_scalars else arr.astype(default_float)
    elif isinstance(val, Var):
        dtype = from_numpy(val.unwrap_tensor().dtype)
        arr = dtype._tyarr_class(val)
    elif isinstance(val, np.ma.MaskedArray):
        raise NotImplementedError
    elif isinstance(val, np.ndarray):
        dtype = from_numpy(val.dtype)
        # TODO: Having `{"var": np_array}` looks awkward...
        arr = dtype._tyarr_class.from_np_schema({"var": val})
    else:
        raise ValueError

    if dtype is not None:
        return arr.astype(dtype)
    return arr
