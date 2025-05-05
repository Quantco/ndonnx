# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import ndonnx as ndx


@pytest.mark.skipif(np.__version__ < "2", reason="np1x semantics differ")
def test_cast():
    def do(npx):
        arr = npx.asarray(["foo", "", "-1", "0", "true", "True", "false", "False"])
        return arr.astype(npx.bool)

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np), strict=True)
