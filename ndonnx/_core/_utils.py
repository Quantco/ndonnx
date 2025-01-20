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
    via_dtype=None,
):
    return variadic_op([x, y], op, via_dtype)


def unary_op(
    x,
    op,
    via_dtype=None,
):
    return variadic_op([x], op, via_dtype)


def variadic_op(
    args,
    op,
    via_dtype=None,
    cast_return=True,
):
    args = promote(*args)
    out_dtype = args[0].dtype
    if not isinstance(out_dtype, (dtypes.CoreType, dtypes.NullableCore)):
        raise TypeError(
            f"Expected ndx.Array with CoreType or NullableCoreType, got {args[0].dtype}"
        )
    data, nulls = split_nulls_and_values(*args)
    if via_dtype is None:
        values = from_corearray(op(*(x._core() for x in data)))
    else:
        values = _via_dtype(op, via_dtype, data, cast_return=cast_return)

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


def _via_dtype(
    fn,
    dtype,
    arrays,
    *,
    cast_return=True,
):
    """Call ``fn`` after casting to ``dtype`` and cast result back to the input dtype.

    The point of this function is to work around quirks / limited type support in the
    onnxruntime.

    ndx.Arrays are promoted to a common type prior to their first use.
    """
    promoted = promote(*arrays)
    out_dtype = promoted[0].dtype

    if isinstance(out_dtype, dtypes.NullableCore) and out_dtype.values == dtype:
        dtype = out_dtype

    values, nulls = split_nulls_and_values(
        *[ndx.astype(arr, dtype) for arr in promoted]
    )

    out_value = from_corearray(fn(*[value._core() for value in values]))
    out_null = functools.reduce(_or_nulls, nulls)

    if out_null is not None:
        out_value = ndx.Array._from_fields(
            dtypes.into_nullable(out_value.dtype), values=out_value, null=out_null
        )
    else:
        out_value = ndx.Array._from_fields(out_value.dtype, data=out_value._core())

    if cast_return:
        return ndx.astype(out_value, dtype=out_dtype)
    else:
        return out_value


def via_upcast(
    fn,
    arrays,
    *,
    int_dtype=None,
    float_dtype=None,
    uint_dtype=None,
    use_unsafe_uint_cast=False,
    cast_return=True,
):
    """Like ``_via_dtype`` but chooses a dtype from the available that doesn't result in
    loss.

    Raises
    ------
    TypeError
        For non-numerical types and if the type can't be safely casted to
        any available type.
    """
    promoted_values = promote(*arrays)

    dtype = promoted_values[0].dtype

    # For logging
    available_types = [t for t in [int_dtype, float_dtype, uint_dtype] if t is not None]

    via_dtype: dtypes.CoreType
    if isinstance(dtype, (dtypes.Unsigned, dtypes.NullableUnsigned)):
        if (
            uint_dtype is not None
            and ndx.iinfo(dtype).bits <= ndx.iinfo(uint_dtype).bits
        ):
            via_dtype = uint_dtype
        elif int_dtype is not None and ndx.iinfo(dtype).bits <= (
            ndx.iinfo(int_dtype).bits
            if use_unsafe_uint_cast
            else ndx.iinfo(int_dtype).bits - 1
        ):
            via_dtype = int_dtype
        else:
            raise TypeError(
                f"Can't upcast unsigned type `{dtype}`. Available implementations are for `{(*available_types,)}`"
            )
    elif isinstance(dtype, (dtypes.Integral, dtypes.NullableIntegral)) or dtype in (
        dtypes.nbool,
        dtypes.bool,
    ):
        if int_dtype is not None and ndx.iinfo(dtype).bits <= ndx.iinfo(int_dtype).bits:
            via_dtype = int_dtype
        else:
            raise TypeError(
                f"Can't upcast signed type `{dtype}`. Available implementations are for `{(*available_types,)}`"
            )
    elif isinstance(dtype, (dtypes.Floating, dtypes.NullableFloating)):
        if (
            float_dtype is not None
            and ndx.finfo(dtype).bits <= ndx.finfo(float_dtype).bits
        ):
            via_dtype = float_dtype
        else:
            raise TypeError(
                f"Can't upcast float type `{dtype}`. Available implementations are for `{(*available_types,)}`"
            )
    else:
        raise TypeError(f"Expected numerical data type, found {dtype}")

    return variadic_op(arrays, fn, via_dtype, cast_return)


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
