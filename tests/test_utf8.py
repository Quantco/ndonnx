# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import ndonnx as ndx


def test_cast():
    def do(npx):
        arr = npx.asarray(["foo", "", "-1", "0", "true", "True", "false", "False"])
        # np1x does not provide np.bool
        return arr.astype(ndx.bool if npx == ndx else bool)

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np), strict=True)
