# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import ndonnx as ndx


@pytest.mark.skipif(np.__version__ < "2", reason="NumPy 1.x does not provide 'concat'")
def test_concat_axis_none_zero_sized():
    def do(npx):
        a1 = npx.zeros(shape=(0, 0), dtype=npx.int8)
        a2 = npx.asarray(0, dtype=npx.int32)
        return npx.concat([a1, a2], axis=None)

    np.testing.assert_array_equal(do(np), do(ndx))


@pytest.mark.parametrize(
    "x, repeats, axis",
    [
        ([1, 2, 3], 2, None),
        ([1, 2, 3], np.asarray(2), None),
        ([1, 2, 3], np.asarray(2), 0),
        ([1, 2, 3], np.asarray([2]), None),
        ([1, 2, 3], [1, 0, 2], None),
        ([1, 2, 3], [1, 0, 2], 0),
        ([[1, 2, 3]], [1, 0, 2], None),
        ([[1, 2, 3]], [1, 0, 2], 1),
        (np.arange(27).reshape((3, 3, 3)), [1, 0, 2], 0),
        (np.arange(27).reshape((3, 3, 3)), [1, 0, 2], 1),
        (np.arange(27).reshape((3, 3, 3)), [1, 0, 2], 2),
        # zero-size
        (np.ones((3, 0, 3)), 2, None),
        (np.ones((3, 0, 3)), 2, 0),
        (np.ones((3, 0, 3)), 2, 1),
        (np.ones((3, 0, 3)), 2, 2),
        (np.ones((3, 0, 3)), [2], 2),
        (np.ones((3, 0, 3)), [1, 0, 2], 2),
        # other data types
        (["a", "b", "c"], np.asarray([2]), None),
        (np.asarray([1, 2, 3], dtype="datetime64[s]"), np.asarray([2]), None),
    ],
)
def test_repeat(x, repeats, axis):
    def do(npx):
        repeats_ = repeats
        if isinstance(repeats, list | np.ndarray):
            repeats_ = npx.asarray(repeats_)
        return npx.repeat(npx.asarray(x), repeats_, axis=axis)

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np))
