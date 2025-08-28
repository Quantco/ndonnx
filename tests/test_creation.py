# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import ndonnx as ndx


@pytest.mark.parametrize(
    "start, stop, step, dtype",
    [
        (0, 10, 2, ndx.int64),
        (-10, 0, 2, ndx.int64),
        (-10, 0, 3, ndx.int64),
        (1, 10, 1, ndx.int64),
        (0.0, None, -1, ndx.int64),
        # Hypothesis test cases
        (
            -9_223_371_349_660_010_402,
            -9_223_370_112_709_427_707,
            45_812_983_809,
            ndx.float64,
        ),
        (-9_223_371_349_660_010_402, -9_223_371_349_660_010_401, 1, ndx.float64),
    ],
)
def test_arange_pyscalar(start, stop, step, dtype: ndx.DType | None):
    def do(npx):
        dtype_: np.dtype | ndx.DType | None = dtype
        if dtype is not None and npx == np:
            dtype_ = dtype.unwrap_numpy()
        return npx.arange(start, stop, step, dtype=dtype_)

    np_res, ndx_res = do(np), do(ndx).unwrap_numpy()

    np.testing.assert_array_equal(np_res, ndx_res, strict=True)


@pytest.mark.parametrize(
    "start, stop, step",
    [
        (np.asarray(0.0), np.asarray(10.0), 1),
        (np.asarray(0.0), 10.0, 1),
        (np.asarray(10.0), None, 1),
        (np.asarray(0, "datetime64[s]"), np.asarray(10, "datetime64[s]"), 1),
        (np.asarray(0, "datetime64[s]"), np.asarray(10_000, "datetime64[ms]"), 1),
        (
            np.asarray(0, "datetime64[s]"),
            np.asarray(10_000, "datetime64[ms]"),
            np.asarray(1_000, "timedelta64[ms]"),
        ),
        (np.asarray(0, "timedelta64[s]"), np.asarray(10, "timedelta64[s]"), 1),
        (np.asarray(0, "timedelta64[s]"), np.asarray(10_000, "timedelta64[ms]"), 1),
        (
            np.asarray(0, "timedelta64[s]"),
            np.asarray(10_000, "timedelta64[ms]"),
            np.asarray(1_000, "timedelta64[ms]"),
        ),
    ],
)
def test_arange_array_arg(start, stop, step):
    def do(npx):
        sss = [
            el if isinstance(el, int | float | None) else npx.asarray(el)
            for el in [start, stop, step]
        ]
        return npx.arange(*sss)

    np_res, ndx_res = do(np), do(ndx).unwrap_numpy()

    np.testing.assert_array_equal(np_res[0], ndx_res[0], strict=True)


def test_timedelta_creation_from_timedelta():
    arr = ndx.asarray(np.asarray([1]), dtype=ndx.TimeDelta64DType("ns"))
    np.testing.assert_array_equal(arr.to_numpy(), ndx.asarray(arr).to_numpy())
