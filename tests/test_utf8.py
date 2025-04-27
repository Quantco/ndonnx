# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import ndonnx as ndx


def test_cast():
    def do(npx):
        arr = npx.asarray(["foo", "", "-1", "0", "true", "True", "false", "False"])
        return arr.astype(npx.bool)

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np), strict=True)
