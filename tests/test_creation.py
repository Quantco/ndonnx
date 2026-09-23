# Copyright (c) QuantCo 2023-2026
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


# A bare-integer step against datetime/timedelta bounds triggers NumPy's
# deprecation of the implicit 'generic' timedelta unit. That bare-integer
# behavior is exactly what we assert works identically in numpy and ndonnx.
@pytest.mark.filterwarnings(
    "ignore:The 'generic' unit for NumPy timedelta:DeprecationWarning"
)
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


@pytest.mark.parametrize(
    "time_dtype_np, time_dtype_ndx",
    [("timedelta64", ndx.TimeDelta64DType), ("datetime64", ndx.DateTime64DType)],
)
@pytest.mark.parametrize("initial_unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("new_unit", ["s", "ms", "us", "ns"])
def test_time_dtype_creation_from_time_dtype(
    time_dtype_np, time_dtype_ndx, initial_unit, new_unit
):
    def do(npx):
        initial_dtype = (
            time_dtype_np + f"[{initial_unit}]"
            if npx == np
            else time_dtype_ndx(initial_unit)
        )
        new_dtype = (
            time_dtype_np + f"[{new_unit}]" if npx == np else time_dtype_ndx(new_unit)
        )
        arr = npx.asarray(np.asarray([1]), dtype=initial_dtype)
        return npx.asarray(arr, dtype=new_dtype)

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np))


@pytest.mark.parametrize(
    "alias",
    [
        int,
        bool,
        float,
        "bool",
        "int",
        "float",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
        "timedelta64[s]",
        "timedelta64[ms]",
        "timedelta64[us]",
        "timedelta64[ns]",
    ],
)
def test_dtype_aliases_resolve(alias):
    def do(npx):
        return npx.asarray(1, dtype=alias)

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np))
