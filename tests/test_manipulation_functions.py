# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import ndonnx as ndx


def test_concat_axis_none_zero_sized():
    def do(npx):
        a1 = npx.zeros(shape=(0, 0), dtype=npx.int8)
        a2 = npx.asarray(0, dtype=npx.int32)
        return npx.concat([a1, a2], axis=None)

    np.testing.assert_array_equal(do(np), do(ndx))
