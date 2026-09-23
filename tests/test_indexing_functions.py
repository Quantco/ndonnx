# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import ndonnx as ndx


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_take_along_axis(axis):
    def do(npx):
        np_arr = np.arange(0, 9).reshape(3, 3)
        np_indices = np.argsort(np_arr, axis=axis)

        indices = npx.asarray(np_indices)
        arr = npx.asarray(np_arr)

        return npx.take_along_axis(arr, indices, axis=axis)

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np), strict=True)


def test_take_along_axis_raises_incompatible_indices():
    # Raise if indices has different rank than x
    x = np.arange(9).reshape((3, 3))  # 2D
    indices = np.asarray([1, 2], dtype=np.int64)  # 1D
    with pytest.raises(ValueError, match="same number of axes"):
        # Not providing axis raises
        ndx.take_along_axis(ndx.asarray(x), ndx.asarray(indices))


@pytest.mark.parametrize("indices", [1, [[2]]])
def test_take_raises_indices_not_1d(indices):
    x = np.arange(9)
    with pytest.raises(ValueError, match="must be a 1D array"):
        ndx.take(
            ndx.asarray(ndx.asarray(x)),
            ndx.asarray(np.asarray(indices, dtype=np.int64)),
        )


def test_take_raises_no_axis_for_ndim_x():
    # The array-api states:
    # If x is a one-dimensional array, providing an axis must be
    # optional; however, if x has more than one axis, providing an
    # axis must be required.
    x = np.arange(9).reshape((3, 3))
    with pytest.raises(ValueError, match="more than one axis"):
        # Not providing axis raises
        ndx.take(
            ndx.asarray(ndx.asarray(x)), ndx.asarray(np.asarray([0], dtype=np.int64))
        )
