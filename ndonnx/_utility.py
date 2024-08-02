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
    all_arguments: list[Array] = []

    for arg in args:
        if isinstance(arg, ndx.Array):
            arrays.append(arg)
            all_arguments.append(arg)
        elif isinstance(arg, np.ndarray):
            arrays.append(ndx.asarray(arg))
            all_arguments.append(arrays[-1])
        elif isinstance(arg, (float, int, str, np.generic)):
            all_arguments.append(ndx.asarray(arg))
        else:
            raise TypeError(f"Cannot promote {type(arg)}")

    if not arrays:
        raise ValueError("At least one array must be provided for type promotion")

    target_dtype = ndx.result_type(*arrays)
    return [arr.astype(target_dtype) for arr in all_arguments]


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
