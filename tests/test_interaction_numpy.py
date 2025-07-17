# Copyright (c) QuantCo 2023-2025
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
    # Forwards
    def do(npx):
        return op(npx.asarray(np_arr), np_gen)

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np))

    # Backwards
    def do(npx):  # type: ignore[no-redef]
        return op(np_gen, npx.asarray(np_arr))

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
