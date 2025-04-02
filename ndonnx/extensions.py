# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias, TypeVar, get_args
from warnings import warn

import numpy as np
from typing_extensions import TypeIs

import ndonnx as ndx

from . import _typed_array as tydx
from ._typed_array.masked_onnx import TyMaArray

SCALAR = TypeVar("SCALAR", int, float, str)

KEY: TypeAlias = SCALAR
VALUE = TypeVar("VALUE", int, float, str)


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


def isin(x: ndx.Array, /, items: Sequence[SCALAR]) -> ndx.Array:
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


# TODO: Bad naming: It is obvious from the type hints that this
# mapping is "static". Furthermore, we don't make a similar
# distinction in other functions.
def static_map(
    x: ndx.Array,
    /,
    mapping: Mapping[KEY, VALUE],
    default: VALUE | None = None,
) -> ndx.Array:
    """Map values in ``x`` based on the static ``mapping``.

    Parameters
    ----------
    x: Array
        The Array whose values will be mapped.
    mapping: Mapping[Key, Value]
        A mapping from keys to values. The keys must be of the same type as the values in ``x``.
    default: Value, optional
        The default value to use when a key is not found in the
        mapping. If `None` the value depends on the type of `Value` in
        the `mapping`:
            - `float`: ``0.0``
            - `int`: ``0``
            - `bool`: ``False``
            - `str`: `"MISSING"`

    Returns
    -------
    out: Array
        A new Array with the values mapped according to the mapping.

    Raises
    ------

    ValueError
        If `mapping` is empty and `default` is ``None``.
    """
    if not mapping and default is None:
        raise ValueError(
            "a 'default' value must be supplied to 'static_map' if 'mapping' is empty"
        )
    if not mapping and default is not None:
        return ndx.broadcast_to(ndx.asarray(default), x.dynamic_shape)

    values = np.array(list(mapping.values()))

    if values.dtype.kind in ("O", "U"):
        values = values.astype(str)

    if default is None:
        # Default values do not follow the ONNX standard (which is ok!)
        if np.issubdtype(values.dtype, np.floating):
            default = 0.0  # type: ignore
        elif values.dtype == bool:
            # to be in line with the numerical 0
            default = False  # type: ignore
        elif np.issubdtype(values.dtype, np.integer):
            default = 0  # type: ignore
        elif values.dtype.kind == "U":
            default = "MISSING"  # type: ignore
    if default is None:
        raise TypeError(
            f"failed to infer default value for mapping values of data type `{values.dtype}`"
        )

    return ndx.Array._from_tyarray(x._tyarray.apply_mapping(mapping, default))


def fill_null(x: ndx.Array, /, value: ndx.Array | SCALAR) -> ndx.Array:
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
    value_ = ndx.asarray(value)._tyarray
    if is_nullable_dtype(value_.dtype):
        raise ValueError("'fill_null' expects a none-nullable fill value data type")
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
        raise TypeError("'fill_null' is only implemented for built-in types")

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
        If `"raise"`, a ``TypeError`` is raised if ``x`` is already of
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
        if isinstance(x._tyarray, TyMaArray):
            return x
        if isinstance(x._tyarray, tydx.onnx.TyArray):
            return ndx.Array._from_tyarray(
                tydx.masked_onnx.make_nullable(x._tyarray, None)
            )
        raise TypeError(f"failed to make array of data type `{x.dtype}` nullable")
    if not isinstance(null._tyarray, tydx.onnx.TyArrayBool):
        raise TypeError(f"'null' must be of boolean data type, found `{null.dtype}`")
    if isinstance(x._tyarray, tydx.onnx.TyArray):
        return ndx.Array._from_tyarray(
            tydx.masked_onnx.make_nullable(x._tyarray, null._tyarray)
        )
    if merge_strategy == "merge" and isinstance(x._tyarray, tydx.masked_onnx.TyMaArray):
        mask = tydx.masked_onnx.merge_masks(x._tyarray.mask, null._tyarray)
        tyarr = tydx.masked_onnx.make_nullable(x._tyarray.data, mask)
        return ndx.Array._from_tyarray(tyarr)
    if isinstance(x._tyarray, tydx.datetime.TimeBaseArray):
        # TODO: The semantics of this branch are very odd!
        is_nat = x._tyarray.is_nat
        merged = tydx.masked_onnx.merge_masks(is_nat, null._tyarray)
        if merged is not None:
            is_nat = merged
        return ndx.Array._from_tyarray(x._tyarray.dtype._build(data=x._tyarray._data))

    raise TypeError(f"'make_nullable' not implemented for `{x.dtype}`")


def get_mask(x: ndx.Array, /) -> ndx.Array | None:
    """Get null-mask if there is any."""
    if isinstance(x._tyarray, tydx.masked_onnx.TyMaArray):
        if x._tyarray.mask is None:
            return None
        return ndx.Array._from_tyarray(x._tyarray.mask)
    return None


def get_data(x: ndx.Array, /) -> ndx.Array:
    """Get data part of a masked, datetime or timedelta array.

    If the ``x`` is not masked, return ``x``.
    """
    if isinstance(x._tyarray, tydx.masked_onnx.TyMaArray | tydx.datetime.TimeBaseArray):
        return ndx.Array._from_tyarray(x._tyarray._data)
    return ndx.Array._from_tyarray(x._tyarray)


def put(a: ndx.Array, indices: ndx.Array, updates: ndx.Array, /) -> None:
    """Replaces specified elements of an array with given values.

    This function follows the semantics of `numpy.put` with
    `mode="raises". The data types of the update array and the updates
    must match. The indices must be provided as a 1D int64 array.
    """
    if not isinstance(indices._tyarray, tydx.onnx.TyArrayInt64):
        # using isinstance here to get the type narrowing below
        raise TypeError(
            f"'indices' must be provided as an int64 tensor, found `{indices.dtype}`"
        )
    if a.dtype != updates.dtype:
        raise TypeError(
            f"data types of 'a' (`{a.dtype}`) and 'updates' (`{updates.dtype}`) must match"
        )
    a._tyarray.put(indices._tyarray, updates._tyarray)


def _year_is_leap_year(year: ndx.Array) -> ndx.Array:
    cycle4 = (year % 4) == 0
    cycle100 = (year % 100) == 0
    cycle400 = (year % 400) == 0

    return cycle4 & (cycle100 | cycle400)


def _number_of_days_and_leap_days_since_16000101(
    dt: ndx.Array,
) -> tuple[ndx.Array, ndx.Array]:
    """Compute number of leap days between `dt` and 1600-01-01 (closest year that aligns
    the 4, 100, and 400 year cycles of leap years)."""
    base = np.asarray("1600-01-01", "datetime64[s]").astype(np.int64)

    delta_s = dt.astype(ndx.DateTime64DType("s")).astype(ndx.int64) - ndx.asarray(base)
    # delta in hours (not days due to the latter use of astro-years)
    delta = delta_s // (60 * 60)

    # Compute number of passed leap years between base and arr
    hours_in_astro_year = int(24 * 365.25)
    cycle4 = delta // (4 * hours_in_astro_year)
    cycle100 = delta // (100 * hours_in_astro_year)
    cycle400 = delta // (400 * hours_in_astro_year)
    leaps = cycle4 - cycle100 + cycle400

    # add days to align again with the first of January
    # delta += (31 + 28) * 24
    # Convert from hours to days
    delta_days = delta // 24

    return delta_days, leaps


def datetime_to_year_month_day(
    arr: ndx.Array,
) -> tuple[ndx.Array, ndx.Array, ndx.Array]:
    """Split date time elements of the provided array into year, month, and day
    components."""
    if not isinstance(arr.dtype, ndx.DateTime64DType):
        raise TypeError(
            f"expected array with date time data type but got `{arr.dtype}` instead"
        )
    # Compute number of passed leap years between base and arr
    delta_days, leaps = _number_of_days_and_leap_days_since_16000101(arr)

    # # Offset by 1 since month counts start at 1
    # delta_days -= 1
    delta_days_no_leaps = delta_days - leaps
    years_delta = delta_days_no_leaps // 365
    year = years_delta + 1600

    is_leap = _year_is_leap_year(year)
    days_in_year_astro = delta_days_no_leaps % 365
    days_in_year = days_in_year_astro + (
        is_leap & (days_in_year_astro >= (31 + 28))
    ).astype(ndx.int64)

    month = ndx.full_like(arr, 1, dtype=ndx.int16)
    days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_edges = np.cumsum(days_in_months)
    for edge in month_edges:
        month += (days_in_year_astro > edge).astype(ndx.int16)

    days_in_past_months = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])
    days_in_past_months_leap = np.cumsum(
        [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30]
    )
    day = days_in_year - ndx.extensions.static_map(
        month + _year_is_leap_year(year).astype(ndx.int16) * 100,
        mapping={k: v for (k, v) in enumerate(days_in_past_months, start=1)}
        | {k: v for (k, v) in enumerate(days_in_past_months_leap, start=101)},
    )

    return year, month, day


def is_onnx_dtype(dtype: ndx.DType, /) -> TypeIs[tydx.onnx.DTypes]:
    """Return ``True`` if ``dtype`` is of a data type found in the ONNX standard."""
    return isinstance(dtype, tydx.onnx.DTypes)


def is_numeric_dtype(dtype: ndx.DType, /) -> TypeIs[tydx.onnx.NumericDTypes]:
    """Return ``True`` if ``dtype`` is of a numeric data type.

    This does not include masked data types.
    """
    return isinstance(dtype, tydx.onnx.NumericDTypes)


def is_float_dtype(dtype: ndx.DType, /) -> TypeIs[tydx.onnx.FloatingDTypes]:
    """Return ``True`` if ``dtype`` is of a floating point data type.

    This does not include masked data types.
    """
    return isinstance(dtype, tydx.onnx.FloatingDTypes)


def is_integer_dtype(dtype: ndx.DType, /) -> TypeIs[tydx.onnx.IntegerDTypes]:
    """Return ``True`` if ``dtype`` is of an integer data type.

    Returns ``False`` for boolean data types.
    """
    return isinstance(dtype, tydx.onnx.IntegerDTypes)


def is_signed_integer_dtype(
    dtype: ndx.DType, /
) -> TypeIs[tydx.onnx.SignedIntegerDTypes]:
    """Return ``True`` if ``dtype`` is of a signed integer data type.

    Returns ``False`` for boolean data types.
    """
    return isinstance(dtype, tydx.onnx.SignedIntegerDTypes)


def is_unsigned_integer_dtype(
    dtype: ndx.DType, /
) -> TypeIs[tydx.onnx.UnsignedIntegerDTypes]:
    """Return ``True`` if ``dtype`` is of an unsigned integer data type.

    Returns ``False`` for boolean data types.
    """
    return isinstance(dtype, tydx.onnx.UnsignedIntegerDTypes)


def is_nullable_dtype(dtype: ndx.DType, /) -> TypeIs[tydx.masked_onnx.DTypes]:
    """Return ``True`` if ``dtype`` is a nullable (i.e. "masked") data type.

    Floating point and datetime data types are not considered as "nullable" by this
    function.
    """
    return isinstance(dtype, tydx.masked_onnx.DTypes)


def is_nullable_integer_dtype(
    dtype: ndx.DType, /
) -> TypeIs[tydx.masked_onnx.IntegerDTypes]:
    """Return ``True`` if ``dtype`` is a nullable integer (i.e. "masked") data type."""
    return isinstance(dtype, tydx.masked_onnx.IntegerDTypes)


def is_nullable_float_dtype(
    dtype: ndx.DType, /
) -> TypeIs[tydx.masked_onnx.FloatDTypes]:
    """Return ``True`` if ``dtype`` is a nullable integer (i.e. "masked") data type."""
    if isinstance(dtype, tydx.masked_onnx.FloatDTypes):
        return True
    return False


def is_time_unit(s: str, /) -> TypeIs[tydx.datetime.Unit]:
    return s in get_args(tydx.datetime.Unit)


__all__ = [
    "datetime_to_year_month_day",
    "fill_null",
    "get_mask",
    "is_float_dtype",
    "is_integer_dtype",
    "is_nullable_dtype",
    "is_nullable_float_dtype",
    "is_nullable_integer_dtype",
    "is_numeric_dtype",
    "is_onnx_dtype",
    "is_signed_integer_dtype",
    "is_time_unit",
    "is_unsigned_integer_dtype",
    "isin",
    "make_nullable",
    "put",
    "shape",
    "static_map",
]
