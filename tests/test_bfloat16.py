# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import ml_dtypes
import numpy as np
import pytest

import ndonnx as ndx


def test_asarray_from_numpy():
    np_arr = np.asarray([1.1, 2.2], ml_dtypes.bfloat16)
    arr = ndx.asarray(np_arr)
    assert arr.dtype == ndx.bfloat16
    np.testing.assert_array_equal(arr.unwrap_numpy(), np_arr)


def test_asarray_from_list():
    lst = [1.1, 2.2]
    np_arr = np.asarray(lst, ml_dtypes.bfloat16)
    arr = ndx.asarray(lst, dtype=ndx.bfloat16)
    assert arr.dtype == ndx.bfloat16
    np.testing.assert_array_equal(arr.unwrap_numpy(), np_arr)


@pytest.mark.parametrize(
    "op", ["abs", "negative", "cumulative_sum", "cumulative_prod", "argmin", "argmax"]
)
def test_unary_operations(op):
    arr = np.asarray([-1.1, 0, 1.1], dtype=ml_dtypes.bfloat16)

    def do(npx):
        return getattr(npx, op)(npx.asarray(arr))

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np))


@pytest.mark.parametrize(
    "op",
    [
        "add",
        "subtract",
        "multiply",
        "divide",
    ],
)
def test_binary_operations(op):
    arr = np.asarray([1.1], dtype=ml_dtypes.bfloat16)

    def do(npx):
        a = npx.asarray(arr)
        b = npx.asarray(arr * 2.0)
        return getattr(npx, op)(a, b)

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np))
