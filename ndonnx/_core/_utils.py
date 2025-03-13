# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import functools
import itertools
from typing import TYPE_CHECKING

import numpy as np

import ndonnx as ndx
import ndonnx._data_types as dtypes
from ndonnx._utility import promote

if TYPE_CHECKING:
    from ndonnx import Array
    from ndonnx._data_types import Dtype


def binary_op(
    x,
    y,
    op,
):
    return variadic_op([x, y], op)


def unary_op(
    x,
    op,
):
    return variadic_op([x], op)


def variadic_op(
    args,
    op,
):
    args = promote(*args)
    out_dtype = args[0].dtype
    if not isinstance(out_dtype, (dtypes.CoreType, dtypes.NullableCore)):
        raise TypeError(
            f"Expected ndx.Array with CoreType or NullableCoreType, got {args[0].dtype}"
        )
    data, nulls = split_nulls_and_values(*args)
    values = from_corearray(op(*(x._core() for x in data)))

    if (out_null := functools.reduce(_or_nulls, nulls)) is not None:
        dtype = dtypes.into_nullable(values.dtype)
        return ndx.Array._from_fields(dtype, values=values, null=out_null)
    else:
        return values


def split_nulls_and_values(*xs):
    """Helper function that splits a series of ndx.Arrays into their constituent value
    and null mask components.

    Raises if an unexpected typed array is provided.
    """
    data: list[Array] = []
    nulls: list[Array | None] = []
    for x in xs:
        if isinstance(x.dtype, dtypes.Nullable):
            nulls.append(x.null)
            data.append(x.values)
        elif isinstance(x, ndx.Array):
            nulls.append(None)
            data.append(x)
        else:
            raise TypeError(f"Expected ndx.Array got {x}")
    return data, nulls


def _or_nulls(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return ndx.logical_or(x, y)


def from_corearray(
    corearray,
):
    return ndx.Array._from_fields(corearray.dtype, data=corearray)


def validate_core(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, ndx.Array) and not isinstance(
                arg.dtype, (dtypes.CoreType, dtypes.NullableCore)
            ):
                return NotImplemented
        return func(*args, **kwargs)

    return wrapper


def assemble_output_recurse(dtype: Dtype, values: dict) -> np.ndarray:
    if isinstance(dtype, dtypes.CoreType):
        return dtype._assemble_output(values)
    else:
        fields = {
            name: assemble_output_recurse(field_dtype, values[name])
            for name, field_dtype in dtype._fields().items()
        }
        return dtype._assemble_output(fields)
