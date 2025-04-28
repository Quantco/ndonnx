# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import ndonnx as ndx


@pytest.mark.parametrize(
    "values",
    [
        [True, False],
        ["foo", "", "0", "1"],
    ],
)
def test_count_nonzero_bool_and_string(values):
    def do(npx):
        arr = npx.asarray(values)
        return npx.count_nonzero(arr)

    np.testing.assert_array_equal(
        do(ndx).unwrap_numpy(), do(np), strict=np.__version__ > "2"
    )


@pytest.mark.parametrize(
    "values",
    [
        [True, False],
        ["foo", "", "0", "1"],
    ],
)
def test_nonzero_bool_and_string(values):
    def do(npx):
        arr = npx.asarray(values)
        return npx.nonzero(arr)

    for ndx_res, np_res in zip(do(ndx), do(np), strict=True):
        np.testing.assert_array_equal(
            ndx_res.unwrap_numpy(), np_res, strict=np.__version__ > "2"
        )
