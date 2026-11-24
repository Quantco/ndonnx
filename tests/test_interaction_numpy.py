# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

import operator

import numpy as np
import pytest

import ndonnx as ndx

from .utils import assert_array_equal


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.floordiv,
        operator.ge,
        operator.gt,
        operator.le,
        operator.lshift,
        operator.lt,
        operator.mod,
        operator.mul,
        operator.pow,
        operator.rshift,
        operator.sub,
        operator.truediv,
    ],
)
@pytest.mark.parametrize(
    "np_arr, np_gen",
    [
        (np.asarray([2], np.uint32), np.int8(2)),
        (np.asarray([2], np.uint32), np.int16(2)),
        (np.asarray([2], np.uint32), np.int32(2)),
        (np.asarray([2], np.uint32), np.int64(2)),
        (np.asarray([2], np.uint32), np.uint8(2)),
        (np.asarray([2], np.uint32), np.uint16(2)),
        (np.asarray([2], np.uint32), np.uint32(2)),
        (np.asarray([2], np.uint32), np.uint64(2)),
    ],
)
def test_dunders_numpy_generic(op, np_arr, np_gen):
    # The first operand is multiplied by two to better test the
    # correct application of non-commutative functions.

    # Forward
    def do(npx):
        return op(npx.asarray(np_arr) * 2, np_gen)

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np))

    # Backward
    def do(npx):  # type: ignore[no-redef]
        return op(np_gen * 2, npx.asarray(np_arr))

    assert_array_equal(do(ndx).unwrap_numpy(), do(np))


def test_datetime_generics():
    np_arr = np.asarray([100], dtype="datetime64[s]")
    scalar = np.asarray([42], dtype="datetime64[s]")[0]

    def do(npx):
        return npx.asarray(np_arr) - scalar

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np), strict=True)

    # backward
    def do(npx):  # type: ignore[no-redef]
        return npx.asarray(np_arr) - scalar

    assert_array_equal(do(ndx).unwrap_numpy(), do(np))


def test_numpy_array_ndx_array_reverse_dunder_called_correctly():
    np_arr = np.asarray([1, 2], dtype=np.int32)
    np_arr_2 = np.asarray([3, 4], dtype=np.int32)

    candidate = np_arr + ndx.asarray(np_arr_2)
    expected = np_arr + np_arr_2

    assert_array_equal(candidate.unwrap_numpy(), expected)


@pytest.mark.parametrize(
    "np_dtype, np_gen",
    [
        (np.int32, np.int64(1)),
        (np.int64, np.int8(1)),
        (np.uint8, np.int8(1)),
        (np.int32, np.float32(1)),
        (np.float32, np.float16(1)),
        (np.float32, np.float32(1)),
        # `np.float64` subclasses `float`, but must not be treated as a Python scalar.
        (np.float32, np.float64(1)),
        (np.bool, np.bool(True)),
        (np.int8, np.bool(True)),
    ],
)
def test_result_type_numpy_generic(np_dtype, np_gen):
    arr = ndx.asarray(np.asarray([1], dtype=np_dtype))
    expected = ndx.from_numpy_dtype(np.result_type(np_dtype, np_gen))

    assert ndx.result_type(arr, np_gen) == expected
    assert ndx.result_type(np_gen, arr) == expected
    assert ndx.result_type(arr.dtype, np_gen) == expected
    # Consistent with binary operations (addition is not defined for two booleans)
    if expected != ndx.bool:
        assert (arr + np_gen).dtype == expected


def test_result_type_numpy_generic_datetime():
    arr = ndx.asarray(np.asarray([1], dtype="datetime64[s]"))
    np_gen = np.timedelta64(1, "ms")
    expected = ndx.from_numpy_dtype(np.result_type(arr.unwrap_numpy(), np_gen))

    assert ndx.result_type(arr, np_gen) == expected
    assert (arr + np_gen).dtype == expected


def test_result_type_numpy_generic_counts_as_dtype():
    # NumPy scalars are treated as their data type rather than as Python scalars.
    assert ndx.result_type(np.int8(1), 1) == ndx.int8
    assert ndx.result_type(np.float64(1)) == ndx.float64


@pytest.mark.parametrize(
    "obj",
    [
        [np.int8(1)],
        [np.uint16(1), np.uint16(2)],
        [[np.float32(1)], [np.float32(2)]],
        [np.bool(True), np.bool(False)],
    ],
)
def test_asarray_sequence_of_numpy_generics(obj):
    candidate = ndx.asarray(obj)
    expected = np.asarray(obj)

    np.testing.assert_array_equal(candidate.unwrap_numpy(), expected, strict=True)
