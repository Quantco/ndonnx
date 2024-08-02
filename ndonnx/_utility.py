# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
import numpy.typing as npt
from spox import Var

# FIXME: Remove private import from Spox! Better to just use reshape!
from spox._internal_op import unsafe_reshape

import ndonnx as ndx

if TYPE_CHECKING:
    from ndonnx import Array

    from ._corearray import _CoreArray


def promote(*args: Array | npt.ArrayLike | None) -> list[Array]:
    """Promote arguments following numpy's promotion rules.

    Constant scalars are converted to `Array` objects.

    `None` values are passed through.
    """
    arrays: list[Array] = []

    for arg in args:
        if isinstance(arg, ndx.Array):
            arrays.append(arg)
        elif isinstance(arg, (np.ndarray, np.generic)):
            arrays.append(ndx.asarray(arg))
        elif isinstance(arg, (float, int, str)):
            continue
        else:
            raise TypeError(f"Cannot promote {type(arg)}")

    if not arrays:
        raise ValueError("At least one array must be provided for type promotion")

    target_dtype = ndx.result_type(*arrays)
    string_dtypes = (ndx.utf8, ndx.nutf8)
    out: list[Array] = []
    for arg in args:
        # Deal with cross-kind scalar promotions
        if isinstance(arg, float) and not isinstance(
            target_dtype, (ndx.Floating, ndx.NullableFloating)
        ):
            target_dtype = ndx.result_type(target_dtype, ndx.float64)
        elif isinstance(arg, bool) and not target_dtype not in (ndx.bool, ndx.nbool):
            target_dtype = ndx.result_type(target_dtype, ndx.bool)
        elif isinstance(arg, int) and not isinstance(
            target_dtype, (ndx.Numerical, ndx.NullableNumerical)
        ):
            target_dtype = ndx.result_type(target_dtype, ndx.int64)

        if not isinstance(arg, ndx.Array):
            arg = ndx.asarray(np.array(arg))
        if arg.dtype in string_dtypes and target_dtype not in string_dtypes:
            raise TypeError("Cannot promote string scalar to numerical type")
        elif arg.dtype not in string_dtypes and target_dtype in string_dtypes:
            raise TypeError("Cannot promote non string scalar to string type")
        out.append(arg)

    return [arr.astype(target_dtype) for arr in out]


# We assume that rank will be static, because
# it is part if the type we are working on
def get_rank(tensor: _CoreArray | Var) -> int:
    return len(get_shape(tensor))


# We assume that type will be static, because
# it is part if the type we are working on
def get_dtype(tensor: _CoreArray | Var) -> np.dtype:
    return np.dtype(unwrap_var(tensor).unwrap_tensor().dtype)


def get_shape(x: _CoreArray | Var) -> tuple[int | str | None, ...]:
    if (shape := unwrap_var(x).unwrap_tensor().shape) is None:
        raise ValueError("Shape is not known")
    return shape


def set_shape(tensor: _CoreArray | Var, shape):
    tensor = unwrap_var(tensor)
    return unsafe_reshape(tensor, shape)


def unwrap_var(tensor: _CoreArray | Var) -> Var:
    if isinstance(tensor, Var):
        return tensor
    else:
        return tensor.var


def deprecated(msg: str):
    """Decorates a function as deprecated and raises a warning when it is called."""

    def _deprecated(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            warn(msg, DeprecationWarning)
            return fn(*args, **kwargs)

        return inner

    return _deprecated
