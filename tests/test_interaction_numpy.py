# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

import operator

import numpy as np
import pytest

import ndonnx as ndx

from .utils import assert_array_equal


@pytest.mark.parametrize(
    "scalar, expected_dtype",
    [
        (np.bool_(True), ndx.bool),
        (np.int8(-3), ndx.int8),
        (np.int64(-3), ndx.int64),
        (np.longlong(-3), ndx.int64),
        (np.uint8(3), ndx.uint8),
        (np.uint64(3), ndx.uint64),
        (np.ulonglong(3), ndx.uint64),
        (np.float16(3.25), ndx.float16),
        (np.float32(3.25), ndx.float32),
        (np.float64(3.25), ndx.float64),
        (np.str_("x"), ndx.utf8),
    ],
)
def test_asarray_numpy_scalar(scalar, expected_dtype):
    candidate = ndx.asarray(scalar)

    assert candidate.dtype == expected_dtype
    assert candidate.shape == ()
    np.testing.assert_array_equal(
        candidate.unwrap_numpy(), np.asarray(scalar), strict=True
    )


def test_asarray_nested_numpy_scalars():
    candidate = ndx.asarray([np.int8(1), np.int64(2)])

    assert candidate.dtype == ndx.int64
    np.testing.assert_array_equal(candidate.unwrap_numpy(), np.asarray([1, 2]))


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
