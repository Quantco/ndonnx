# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import ndonnx as ndx


@pytest.mark.skip(
    reason="NumPy is data dependent in this case but the standard leaves as UB"
)
def test_asarray_infer_dtype_very_large_integer():
    def do(npx):
        return npx.asarray(9223372036854775808)

    np.testing.assert_equal(do(np), do(ndx).unwrap_numpy())


@pytest.mark.parametrize(
    "start,stop,step",
    [
        (-1, 9_223_372_036_854_775_807, 288_230_376_151_711_744),
        (9_223_372_036_854_775_806, 9_223_372_036_854_775_807, 1),
    ],
)
def test_arange(start, stop, step):
    def do(npx):
        return npx.arange(start, stop, step, dtype=npx.float32)

    np_res = do(np)
    ndx_res = do(ndx).unwrap_numpy()

    assert np_res.shape == ndx_res.shape

    np.testing.assert_equal(np_res[0], ndx_res[0])


def test_all():
    def do(npx):
        arr = npx.asarray(np.ones(shape=(0, 0), dtype=bool))
        return npx.all(arr)

    np.testing.assert_equal(do(np), do(ndx).unwrap_numpy())


def test_any():
    def do(npx):
        arr = npx.asarray(np.ones(shape=(0, 0), dtype=bool))
        return npx.any(arr)

    np.testing.assert_equal(do(np), do(ndx).unwrap_numpy())


def test_min():
    def do(npx):
        arr = npx.asarray([-2147483649, -1, -1, -1], dtype=npx.int64)
        return npx.min(arr)

    np.testing.assert_equal(do(np), do(ndx).unwrap_numpy())


def test_max():
    def do(npx):
        arr = npx.asarray([2147483649, 1, 1, 1], dtype=npx.int64)
        return npx.max(arr)

    np.testing.assert_equal(do(np), do(ndx).unwrap_numpy())


@pytest.mark.parametrize(
    "arr,key,value",
    [
        (
            np.ones(shape=(0, 1), dtype=bool),
            (..., slice(0, None)),
            np.ones(shape=(0, 1)),
        ),
        (
            np.arange(6).reshape((1, 2, 3)),
            (..., slice(0, None)),
            np.arange(10, 16).reshape((1, 2, 3)),
        ),
        (
            np.arange(6).reshape((1, 2, 3)),
            (0, ..., slice(0, None)),
            np.arange(10, 16).reshape((2, 3)),
        ),
    ],
)
def test_setitem(arr, key, value):
    def do(npx):
        x = npx.asarray(arr)
        x[key] = npx.asarray(value)
        return x

    np.testing.assert_equal(do(np), do(ndx).unwrap_numpy())
