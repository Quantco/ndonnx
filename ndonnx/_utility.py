# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from spox import Var

# FIXME: Remove private import from Spox! Better to just use reshape!
from spox._internal_op import unsafe_reshape

import ndonnx._data_types as dtypes

if TYPE_CHECKING:
    from ndonnx import Array

    from ._corearray import _CoreArray


def _promote_with_none(*args: Array | npt.ArrayLike) -> list[Array | None]:
    """Promote arguments following numpy's promotion rules.

    Constant scalars are converted to `Array` objects.

    `None` values are passed through.
    """
    # FIXME: The import structure is rather FUBAR!
    from ._array import Array
    from ._funcs import asarray, astype, result_type

    arr_or_none: list[Array | None] = []
    scalars: list[Array] = []
    signed_integer = False
    for arg in args:
        if arg is None:
            arr_or_none.append(arg)
        elif isinstance(arg, Array):
            arr_or_none.append(arg)
            if arg.dtype in (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64):
                signed_integer = True
        elif isinstance(arg, np.ndarray):
            arr_or_none.append(asarray(arg))
        elif isinstance(arg, (int, float, str, np.generic)):
            np_dtype = np.min_scalar_type(arg)
            if np_dtype == np.dtype("float16"):
                np_dtype = np.dtype("float32")
            arr = asarray(arg, dtypes.from_numpy_dtype(np_dtype))
            arr_or_none.append(arr)
            scalars.append(arr)
        else:
            raise TypeError(f"Cannot promote {type(arg)}")

    for scalar in scalars:
        eager_value = scalar.to_numpy()
        if eager_value is None:
            raise ValueError("Cannot promote `None` value")
        if signed_integer and eager_value.dtype.kind == "u":
            # translate dtype to signed version
            dtype = np.dtype(f"int{eager_value.dtype.itemsize * 8}")
            if eager_value > np.iinfo(dtype).max:
                dtype = np.dtype(f"int{eager_value.dtype.itemsize * 16}")
            scalar._set(scalar.astype(dtypes.from_numpy_dtype(dtype)))

    target_dtype = result_type(*[arr for arr in arr_or_none if arr is not None])

    return [None if arr is None else astype(arr, target_dtype) for arr in arr_or_none]


def promote(*args: Array | npt.ArrayLike) -> list[Array]:
    """Promote arguments following numpy's promotion rules.

    Constant scalars are converted to `Array` objects.
    """
    promoted = _promote_with_none(*args)
    ret = []
    for el in promoted:
        if el is None:
            raise ValueError("Cannot promote `None` value")
        ret.append(el)
    return ret


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
