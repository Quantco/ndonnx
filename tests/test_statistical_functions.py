# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from packaging.version import parse

import ndonnx as ndx

AXES = [None, 0, 1, (0, 1), (1, 0), (), (-1, 0), (-1)]

CORRECTIONS = [0, 1, 0.0, 1.0]

FLOAT_DTYPES = [np.float32, np.float64]
NUMERIC_DTYPES = (
    FLOAT_DTYPES
    + [np.int8, np.int16, np.int32, np.int64]
    + [np.uint8, np.uint16, np.uint32, np.uint64]
)

ARRAYS = [
    np.array([[-3, -1], [2, 3]]),
    np.array([[-3], [-1], [2], [3]]),
    np.array([[], [], [], []]),
]


if parse(np.__version__).major < 2:
    pytest.skip(
        reason="Statistical functions are not tested on NumPy 1 due to API incompatibilities",
        allow_module_level=True,
    )


def _compare_to_numpy(ndx_fun, np_fun, np_array, kwargs):
    candidate = ndx_fun(ndx.asarray(np_array), **kwargs)
    expectation = np_fun(np_array, **kwargs)
    np.testing.assert_allclose(candidate.unwrap_numpy(), expectation)


@pytest.mark.parametrize(
    "correction",
    [
        0,
    ],
)  # 1, 0.0, 1.0])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("np_arr", ARRAYS)
def test_std(dtype, correction, keepdims, axis, np_arr):
    np_arr = np_arr.astype(dtype)
    kwargs = {"correction": correction, "keepdims": keepdims, "axis": axis}

    _compare_to_numpy(ndx.std, np.std, np_arr, kwargs)


@pytest.mark.parametrize("correction", [0, 1, 0.0, 1.0])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("np_arr", ARRAYS)
def test_var(dtype, correction, keepdims, axis, np_arr):
    np_arr = np_arr.astype(dtype)
    kwargs = {"correction": correction, "keepdims": keepdims, "axis": axis}

    _compare_to_numpy(ndx.var, np.var, np_arr, kwargs)


@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("np_arr", ARRAYS)
def test_mean(dtype, keepdims, axis, np_arr):
    np_arr = np_arr.astype(dtype)
    kwargs = {"keepdims": keepdims, "axis": axis}

    _compare_to_numpy(ndx.mean, np.mean, np_arr, kwargs)


@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize("np_arr", ARRAYS)
def test_prod(dtype, keepdims, axis, np_arr):
    # Take abs to avoid overflow issues with unsigned data types
    np_arr = np.abs(np_arr).astype(dtype)
    kwargs = {"keepdims": keepdims, "axis": axis}

    _compare_to_numpy(ndx.prod, np.prod, np_arr, kwargs)


@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize("np_arr", ARRAYS)
def test_sum(dtype, keepdims, axis, np_arr):
    # Take abs to avoid overflow issues with unsigned data types
    np_arr = np.abs(np_arr).astype(dtype)
    kwargs = {"keepdims": keepdims, "axis": axis}

    _compare_to_numpy(ndx.sum, np.sum, np_arr, kwargs)


@pytest.mark.parametrize("axis", [-1, (0, -1), (2, -1), (-3, 3)])
def test_mean_higher_dim(axis):
    np_arr = np.arange(0, 3 * 3 * 3 * 3, dtype=np.float32).reshape((3, 3, 3, 3))
    axis = -2
    expected = np.mean(np_arr, axis=axis)
    candidate = ndx.mean(ndx.asarray(np_arr), axis=axis).unwrap_numpy()

    np.testing.assert_array_equal(candidate, expected)
