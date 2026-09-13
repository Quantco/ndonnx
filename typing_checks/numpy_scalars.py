# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause
# mypy: warn_unused_ignores=True

from typing import assert_type

import numpy as np

import ndonnx as ndx

assert_type(ndx.asarray(np.bool_(True)), ndx.Array)
assert_type(ndx.asarray(np.int8(-3)), ndx.Array)
assert_type(ndx.asarray(np.int16(-3)), ndx.Array)
assert_type(ndx.asarray(np.int32(-3)), ndx.Array)
assert_type(ndx.asarray(np.int64(-3)), ndx.Array)
assert_type(ndx.asarray(np.int_(-3)), ndx.Array)
assert_type(ndx.asarray(np.intp(-3)), ndx.Array)
assert_type(ndx.asarray(np.longlong(-3)), ndx.Array)
assert_type(ndx.asarray(np.uint8(3)), ndx.Array)
assert_type(ndx.asarray(np.uint16(3)), ndx.Array)
assert_type(ndx.asarray(np.uint32(3)), ndx.Array)
assert_type(ndx.asarray(np.uint64(3)), ndx.Array)
assert_type(ndx.asarray(np.uint(3)), ndx.Array)
assert_type(ndx.asarray(np.uintp(3)), ndx.Array)
assert_type(ndx.asarray(np.ulonglong(3)), ndx.Array)
assert_type(ndx.asarray(np.float16(3.25)), ndx.Array)
assert_type(ndx.asarray(np.float32(3.25)), ndx.Array)
assert_type(ndx.asarray(np.float64(3.25)), ndx.Array)
assert_type(ndx.asarray(np.str_("x")), ndx.Array)
assert_type(ndx.asarray([np.int8(1), np.int64(2)]), ndx.Array)

integer_array = ndx.asarray([1, 2], dtype=ndx.int32)
floating_array = ndx.asarray([1.0, 2.0], dtype=ndx.float32)
boolean_array = ndx.asarray([True, False])
nullable_array = ndx.asarray([1, 2], dtype=ndx.nint32)
datetime_array = ndx.asarray(np.asarray([0], dtype="datetime64[s]"))
timedelta_array = ndx.asarray(np.asarray([0], dtype="timedelta64[s]"))

# Integer data values with explicit temporal dtypes
assert_type(ndx.asarray(np.int64(1), dtype=ndx.DateTime64DType("s")), ndx.Array)
assert_type(ndx.asarray(np.uint64(1), dtype=ndx.TimeDelta64DType("ms")), ndx.Array)
assert_type(ndx.full((2,), np.int8(1), dtype=ndx.DateTime64DType("s")), ndx.Array)
assert_type(ndx.full_like(timedelta_array, np.int64(1)), ndx.Array)
datetime_array[0] = np.int64(1)
timedelta_array[0] = np.uint64(1)

# Creation and dtype-related functions
assert_type(ndx.arange(np.int8(0), np.int64(2), np.int8(1)), ndx.Array)
assert_type(ndx.full((2,), np.float32(1)), ndx.Array)
assert_type(ndx.full_like(integer_array, np.int64(1)), ndx.Array)
assert_type(ndx.linspace(np.float32(0), np.float64(1), num=2), ndx.Array)
assert_type(ndx.repeat(integer_array, np.int64(2)), ndx.Array)
ndx.result_type(ndx.int32, np.int64(1))

# Elementwise numeric, comparison, bitwise, and logical functions
assert_type(ndx.add(integer_array, np.int64(1)), ndx.Array)
boolean_scalar = np.bool_(True)
assert_type(ndx.add(integer_array, boolean_scalar), ndx.Array)
assert_type(ndx.add(boolean_scalar, integer_array), ndx.Array)
assert_type(ndx.subtract(integer_array, boolean_scalar), ndx.Array)
assert_type(ndx.subtract(boolean_scalar, integer_array), ndx.Array)
assert_type(ndx.multiply(integer_array, boolean_scalar), ndx.Array)
assert_type(ndx.multiply(boolean_scalar, integer_array), ndx.Array)
assert_type(ndx.logaddexp(floating_array, np.float32(1)), ndx.Array)
assert_type(ndx.maximum(integer_array, np.int64(1)), ndx.Array)
assert_type(ndx.greater(integer_array, np.int8(0)), ndx.Array)
assert_type(ndx.equal(boolean_array, np.bool_(True)), ndx.Array)
assert_type(ndx.bitwise_and(integer_array, np.int8(1)), ndx.Array)
assert_type(ndx.logical_or(boolean_array, np.bool_(True)), ndx.Array)
assert_type(ndx.clip(floating_array, np.float16(0), np.float64(1)), ndx.Array)

# Selection with arrays and scalar-only branches
assert_type(ndx.where(boolean_array, integer_array, np.int64(0)), ndx.Array)
assert_type(ndx.where(boolean_array, np.int8(1), integer_array), ndx.Array)
assert_type(ndx.where(boolean_array, np.int8(1), np.int64(2)), ndx.Array)

# Array scalar operators and assignment
assert_type(integer_array + np.int64(1), ndx.Array)
assert_type(integer_array > np.int8(0), ndx.Array)
assert_type(integer_array & np.uint8(1), ndx.Array)
assert_type(integer_array == np.int32(1), ndx.Array)
assert_type(datetime_array + np.timedelta64(1, "s"), ndx.Array)
assert_type(np.timedelta64(1, "s") + datetime_array, ndx.Array)
assert_type(datetime_array - np.datetime64(0, "s"), ndx.Array)
assert_type(timedelta_array + np.timedelta64(1, "s"), ndx.Array)
integer_array[0] = np.int64(1)

# Reduction corrections
assert_type(ndx.std(floating_array, correction=np.int64(1)), ndx.Array)
assert_type(ndx.var(floating_array, correction=np.float32(1)), ndx.Array)

# Public extensions
assert_type(ndx.extensions.isin(integer_array, [np.int8(1)]), ndx.Array)
assert_type(
    ndx.extensions.static_map(
        integer_array, {np.int8(1): np.int64(2)}, default=np.int64(0)
    ),
    ndx.Array,
)
assert_type(ndx.extensions.fill_null(nullable_array, np.int16(0)), ndx.Array)
assert_type(
    ndx.extensions.static_map(floating_array, {np.float16(1): 3}, default=0),
    ndx.Array,
)

ndx.asarray(np.complex64(1))  # type: ignore[arg-type]
ndx.asarray(np.void(b"x"))  # type: ignore[arg-type]
ndx.asarray(np.datetime64("2024-01-01", "s"))  # type: ignore[arg-type]
ndx.asarray(np.timedelta64(1, "s"))  # type: ignore[arg-type]
integer_array + np.complex64(1)  # type: ignore[arg-type]
integer_array + np.void(b"x")  # type: ignore[arg-type]
ndx.repeat(integer_array, np.float64(2))  # type: ignore[arg-type]
ndx.extensions.static_map(floating_array, {1: np.float16(2)})  # type: ignore[type-var]
ndx.extensions.static_map(floating_array, {}, default=np.float16(0))  # type: ignore[arg-type]
