# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import pytest

import ndonnx as ndx

ARRAYS = [np.linspace(-10, 10, 10), np.array([])]

FLOAT_DTYPES = [np.float32, np.float64]


def _compare_to_numpy(ndx_fun, np_fun, np_array):
    candidate = ndx_fun(ndx.asarray(np_array))
    expectation = np_fun(np_array)
    np.testing.assert_allclose(candidate.unwrap_numpy(), expectation, rtol=1e-5)


@pytest.mark.parametrize(
    "name",
    [
        "sin",
        "cos",
        "tan",
        "cosh",
        "tanh",
        "sinh",
        "arcsin",
        "arccos",
        "arctan",
        "arccosh",
        "arctanh",
        "arcsinh",
    ],
)
@pytest.mark.parametrize("np_arr", ARRAYS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_trigonometric(name, np_arr, dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # function names are using "arc" for compatibility with numpy v1
        array_api_name = name.replace("arc", "a")
        _compare_to_numpy(
            getattr(ndx, array_api_name), getattr(np, name), np_arr.astype(dtype)
        )
