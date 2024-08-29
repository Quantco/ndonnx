# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypeAlias, TypeVar

import numpy as np

import ndonnx as ndx
from ndonnx import _opset_extensions as opx

if TYPE_CHECKING:
    from ndonnx import Array
    from ndonnx._array import IndexType

Scalar = TypeVar("Scalar", int, float, str)

Key: TypeAlias = Scalar
Value = TypeVar("Value", int, float, str)


def shape(x: Array) -> Array:
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
    out = x.dtype._ops.shape(x)
    if out is NotImplemented:
        raise ndx.UnsupportedOperationError(f"`shape` not implemented for `{x.dtype}`")
    return out


def isin(x: Array, items: Sequence[Scalar]) -> Array:
    """Return true where the input ``Array`` contains an element in ``items``.

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
    if isinstance(x.dtype, ndx.Nullable):
        return isin(x.values, items) & ~x.null
    elif len(items) == 1:
        if isinstance(items[0], float) and np.isnan(items[0]):
            return ndx.isnan(x)
        else:
            return x == items[0]
    else:
        mapping = dict(zip(items, (1,) * len(items)))
        return static_map(x, mapping, 0).astype(ndx.bool)


def static_map(
    x: Array,
    mapping: Mapping[Key, Value],
    default: Value | None = None,
) -> Array:
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
    if not isinstance(x.dtype, ndx.CoreType):
        raise ndx.UnsupportedOperationError(
            "'static_map' accepts only non-nullable arrays"
        )
    data = opx.static_map(x._core(), mapping, default)
    return ndx.Array._from_fields(data.dtype, data=data)


def fill_null(x: Array, value: Array | Scalar) -> Array:
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

    out = x.dtype._ops.fill_null(x, value)
    if out is NotImplemented:
        raise ndx.UnsupportedOperationError(
            f"`fill_null` not implemented for `{x.dtype}`"
        )
    return out


def make_nullable(x: Array, null: Array) -> Array:
    """Given an array ``x`` of values and a null mask ``null``, construct a new Array
    with a nullable data type.

    Parameters
    ----------
    x: Array
        Array of values

    null: Array
        Array of booleans indicating whether each element of ``x`` is null.

    Returns
    -------
    out: Array
        A new Array with a nullable data type.

    Raises
    ------
    TypeError
        If the data type of ``x`` does not have a nullable counterpart.
    """
    out = x.dtype._ops.make_nullable(x, null)
    if out is NotImplemented:
        raise ndx.UnsupportedOperationError(
            f"'make_nullable' not implemented for `{x.dtype}`"
        )
    return out


def _getitem(x: Array, index: IndexType) -> ndx.Array:
    out = x.dtype._ops.getitem(x, index)
    if out is NotImplemented:
        raise ndx.UnsupportedOperationError(
            f"'getitem' not implemented for `{x.dtype}`"
        )
    return out


def _static_shape(x: Array) -> tuple[int | None, ...]:
    """Return shape of the array as a tuple. Typical implementations will make use of
    ONNX shape inference, with `None` entries denoting unknown or symbolic dimensions.

    Parameters
    ----------
    x: Array
        Array to get static shape of

    Returns
    -------
    out: tuple[int | None, ...]
    """
    out = x.dtype._ops.static_shape(x)
    if out is NotImplemented:
        raise ndx.UnsupportedOperationError(
            f"`static_shape` not implemented for `{x.dtype}`"
        )
    return out
