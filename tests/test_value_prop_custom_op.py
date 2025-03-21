# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import ndonnx as ndx
from ndonnx._future import propagate_with_custom_operators


@propagate_with_custom_operators(None)  # type: ignore
def square(a: ndx.Array) -> ndx.Array:
    return a * a


def test_propagate_with_custom_operators():
    a = ndx.asarray(2)

    b = square(a)

    np.testing.assert_array_equal(b.unwrap_numpy(), 4)
