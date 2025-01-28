# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias, TypeVar
from warnings import warn

import numpy as np

import ndonnx._refactor as ndx

from . import _typed_array as tydx
from ._typed_array.masked_onnx import TyMaArray

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
    if default is None:
        raise TypeError(
            f"failed to infer default value for mapping values of data type `{values.dtype}`"
        )

    return ndx.Array._from_tyarray(x._tyarray.apply_mapping(mapping, default))


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
    if isinstance(value_.dtype, ndx.Nullable):
        raise ValueError("'fill_null' expects a none-nullable fill value data type.")
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
    null: ndx.Array | None,
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
    x = x.copy()
    null = None if null is None else null.copy()

    if null is None:
        if isinstance(x, TyMaArray):
            return x
        if isinstance(x, tydx.onnx.TyArray):
            return ndx.Array._from_tyarray(
                tydx.masked_onnx.asncoredata(x._tyarray, None)
            )
        raise TypeError(f"failed to make array of data type `{x.dtype}` nullable")
    if not isinstance(null._tyarray, tydx.onnx.TyArrayBool):
        raise TypeError(f"'null' must be of boolean data type, found `{null.dtype}`")
    if isinstance(x._tyarray, tydx.onnx.TyArray):
        return ndx.Array._from_tyarray(
            tydx.masked_onnx.asncoredata(x._tyarray, null._tyarray)
        )
    if merge_strategy == "merge" and isinstance(x._tyarray, tydx.masked_onnx.TyMaArray):
        mask = tydx.masked_onnx._merge_masks(x._tyarray.mask, null._tyarray)
        tyarr = tydx.masked_onnx.asncoredata(x._tyarray.data, mask)
        return ndx.Array._from_tyarray(tyarr)
    if isinstance(x._tyarray, tydx.date_time.TimeBaseArray):
        # TODO: The semantics of this branch are very odd!
        is_nat = tydx.masked_onnx._merge_masks(x._tyarray.is_nat, null._tyarray)
        tyarr = x.dtype._tyarr_class(
            is_nat, x._tyarray.data, unit=x._tyarray.dtype.unit
        )
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


def put(a: ndx.Array, indices: ndx.Array, updates: ndx.Array, /) -> None:
    """Replaces specified elements of an array with given values.

    This function follows the semantics of `numpy.put` with
    `mode="raises". The data types of the update array and the updates
    must match. The indices must be provided as a 1D int64 array.
    """
    from ._typed_array.onnx import TyArrayInt64

    if not isinstance(indices._tyarray, TyArrayInt64):
        raise TypeError(
            f"'indices' must be provided as an int64 tensor, found `{indices.dtype}`"
        )
    if a.dtype != updates.dtype:
        raise TypeError(
            f"data types of 'a' (`{a.dtype}`) and 'updates' (`{updates.dtype}`) must match."
        )
    a._tyarray.put(indices._tyarray, updates._tyarray)
