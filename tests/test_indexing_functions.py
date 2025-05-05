# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import ndonnx as ndx


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_take_along_axis(axis):
    def do(npx):
        np_arr = np.arange(0, 9).reshape(
            3,
            3,
        )
        np_indices = np.argsort(np_arr, axis=axis)

        indices = npx.asarray(np_indices)
        arr = npx.asarray(np_arr)

        return npx.take_along_axis(arr, indices, axis=axis)

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np), strict=True)
