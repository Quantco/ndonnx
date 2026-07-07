# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import ndonnx as ndx


@pytest.mark.parametrize("x", [-3.0, 3.0])
@pytest.mark.parametrize("y", [-2.0, 2.0])
def test_floating_point_modulo_follows_python(x, y):
    def do(npx):
        x_arr = npx.asarray(x, dtype=npx.float64)
        y_arr = npx.asarray(y, dtype=npx.float64)

        return x_arr % y_arr

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np))
    assert do(ndx).unwrap_numpy() == x % y  # Compare to Python result


@pytest.mark.parametrize("a,b", [(12.3, 3), (12, 3.3)])
def test_mod_float_int(a, b):
    a1 = np.asarray(a)
    a2 = np.asarray(b)

    def do(npx):
        return npx.asarray(a1) % npx.asarray(a2)

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np))
