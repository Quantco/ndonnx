# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias, TypeVar
from warnings import warn

import numpy as np

import ndonnx._refactor as ndx
from ndonnx._refactor import _typed_array as tydx

Scalar = TypeVar("Scalar", int, float, str)

Key: TypeAlias = Scalar
Value = TypeVar("Value", int, float, str)


def shape(x: ndx.Array, /) -> ndx.Array:
    """Returns shape of an array.

    Parameters
    ----------
    x: Array
        Array to get shape of

    Returns
    -------
    out: Array
        Array of shape
    """
    warn(
        "'ndonnx.shape' is deprecated in favor of 'ndonnx.Array.dynamic_shape'",
        DeprecationWarning,
    )
    return x.dynamic_shape


def isin(x: ndx.Array, /, items: Sequence[Scalar]) -> ndx.Array:
    """Return true where the input ``Array`` contains an element in ``items``.

    ``NaN`` values do **not** compare equal.

    Parameters
    ----------
    x: Array
        The input Array to check for the presence of items.
    items: Sequence[Scalar]
        Scalar items to check for in the input Array.

    Returns
    -------
    out: Array
        Array of booleans indicating whether each element of ``x`` is in ``items``.
    """
    return ndx.Array._from_tyarray(x._tyarray.isin(items))
    # Filter out nan values since we never want to compare equal to them (NumPy semantics)
    items = [el for el in items if not isinstance(el, float) or not np.isnan(el)]
    if len(items) == 0:
        return ndx.full_like(x, False, dtype=ndx.bool)
    if len(items) == 1:
        return x == items[0]
    mapping = dict(zip(items, (1,) * len(items)))
    res = static_map(x, mapping, 0)
    if isinstance(res.dtype, ndx.Nullable):
        data = get_data(res).astype(ndx.bool)
        mask = get_mask(res)
        if mask is None:
            return data
        return data & ~mask
    return res.astype(ndx.bool)


# TODO: Bad naming: It is obvious from the type hints that this mapping is "static"
def static_map(
    x: ndx.Array,
    /,
    mapping: Mapping[Key, Value],
    default: Value | None = None,
) -> ndx.Array:
    """Map values in ``x`` based on the static ``mapping``.

    Parameters
    ----------
    x: Array
        The Array whose values will be mapped.
    mapping: Mapping[Key, Value]
        A mapping from keys to values. The keys must be of the same type as the values in ``x``.
    default: Value, optional
        The default value to use when a key is not found in the mapping.

    Returns
    -------
    out: Array
        A new Array with the values mapped according to the mapping.
    """
    if not isinstance(x._tyarray, tydx.onnx.TyArray):
        raise ndx.UnsupportedOperationError(
            f"'static_map' accepts only non-nullable arrays, found `{x.dtype}`"
        )

    np_dtype = x.dtype.unwrap_numpy()
    keys = np.array(list(mapping.keys()), dtype=np_dtype)
    values = np.array(list(mapping.values()))

    if values.dtype.kind in ("O", "U"):
        values = values.astype(str)

    if default is None:
        # Default values do not follow the ONNX standard (which is ok!)
        if np.issubdtype(values.dtype, np.floating):
            default = 0.0  # type: ignore
        elif np.issubdtype(values.dtype, np.integer):
            default = 0  # type: ignore
        elif values.dtype.kind == "U":
            default = "MISSING"  # type: ignore
        elif values.dtype == bool:
            # to be in line with the numerical 0
            default = False  # type: ignore

    tyarr = x._tyarray.static_map(dict(zip(keys, values)), default)
    return ndx.Array._from_tyarray(tyarr)


def fill_null(x: ndx.Array, /, value: ndx.Array | Scalar) -> ndx.Array:
    """Returns a new ``Array`` with the null values filled with the given value.

    Parameters
    ----------
    x: Array
        The Array to fill the null values of.
    value: Array | Scalar
        The value to fill the null values with.

    Returns
    -------
    out: Array
        A new Array with the null values filled with the given value.
    """
    value_ = tydx.astyarray(value)
    xty = x._tyarray
    if isinstance(xty, tydx.masked_onnx.TyMaArray):
        result_type = ndx.result_type(xty.data.dtype, value_.dtype)
        if xty.mask is None:
            res: tydx.TyArrayBase = xty.data
        else:
            res = tydx.funcs.where(xty.mask, value_, xty.data)
    elif isinstance(xty, tydx.onnx.TyArray):
        result_type = ndx.result_type(xty.dtype, value_.dtype)
        res = xty.astype(result_type)
    else:
        raise TypeError("'fill_null' is only implemented for built-in types.")

    return ndx.Array._from_tyarray(res).astype(result_type)


def make_nullable(
    x: ndx.Array,
    null: ndx.Array,
    /,
    *,
    merge_strategy: Literal["raise", "merge"] = "raise",
) -> ndx.Array:
    """Given an array ``x`` of values and a null mask ``null``, construct a new Array
    with a nullable data type.

    Parameters
    ----------
    x: Array
        Array of values

    null: Array
        Array of booleans indicating whether each element of ``x`` is null.

    merge_strategy: Literal["raise", "merge"]
        If `"raise"` a ``TypeError`` is raised if ``x`` is already of
        a nullable data type. If `"merge"` is provided, any mask
        existing on ``x`` is merged with ``null``.

    Returns
    -------
    out: Array
        A new Array with a nullable data type.

    Raises
    ------
    TypeError
        If the data type of ``x`` does not have a nullable counterpart.
    """
    if not isinstance(null._tyarray, tydx.TyArrayBool):
        raise TypeError(f"'null' must be of boolean data type, found `{null.dtype}`")
    if isinstance(x._tyarray, tydx.TyArray):
        return ndx.Array._from_tyarray(
            tydx.masked_onnx.asncoredata(x._tyarray, null._tyarray)
        )
    if merge_strategy == "merge" and isinstance(x._tyarray, tydx.TyMaArray):
        mask = tydx.masked_onnx._merge_masks(x._tyarray.mask, null._tyarray)
        tyarr = tydx.asncoredata(x._tyarray.data, mask)
        return ndx.Array._from_tyarray(tyarr)
    raise ndx.UnsupportedOperationError(
        f"'make_nullable' not implemented for `{x.dtype}`"
    )


def get_mask(x: ndx.Array, /) -> ndx.Array | None:
    """Get null-mask if there is any."""
    if isinstance(x._tyarray, tydx.masked_onnx.TyMaArray):
        if x._tyarray.mask is None:
            return None
        return ndx.Array._from_tyarray(x._tyarray.mask)
    return None


def get_data(x: ndx.Array, /) -> ndx.Array:
    """Get data part of a masked array.

    If the ``x`` is not masked, return ``x``.
    """
    if isinstance(x._tyarray, tydx.masked_onnx.TyMaArray):
        return ndx.Array._from_tyarray(x._tyarray.data)
    return ndx.Array._from_tyarray(x._tyarray)
