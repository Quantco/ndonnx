# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import ndonnx as ndx
from ndonnx._typed_array.string import _not_set


def assert_array_equal(candidate: ndx.Array, expected: np.ndarray):
    np.testing.assert_array_equal(candidate.unwrap_numpy(), expected, strict=True)


def get_na_object_kwarg(dtype: np.dtype) -> dict:
    if not hasattr(dtype, "na_object"):
        return {}
    return {"na_object": dtype.na_object}  # type: ignore


@pytest.fixture(params=[np.nan, "__sentinel__", _not_set])
def np_array(request) -> np.ndarray:
    # Create a NumPy array for each na_object variant
    val = ["foo"]
    kwarg = {} if request.param == _not_set else {"na_object": request.param}

    if request.param != _not_set:
        val += [request.param]

    return np.asarray(val, dtype=np.dtypes.StringDType(**kwarg, coerce=False))


@pytest.mark.parametrize("from_np", [False, True])
def test_creation_explicit_dtype(np_array: np.ndarray, from_np: bool):
    def do(npx):
        val_ = np_array if from_np else np_array.tolist()
        return npx.asarray(
            val_,
            dtype=npx.dtypes.StringDType(
                **get_na_object_kwarg(np_array.dtype), coerce=False
            ),
        )

    assert_array_equal(do(ndx), do(np))


def test_creation_inferred_dtype(np_array: np.ndarray):
    def do(npx):
        return npx.asarray(np_array)

    assert_array_equal(do(ndx), do(np))


@pytest.mark.parametrize("to_na_object", [np.nan, "sentinel", _not_set])
def test_cast_to_other_string_dtype(np_array: np.ndarray, to_na_object):
    kwarg = {} if to_na_object == _not_set else {"na_object": to_na_object}

    def do(npx):
        to_dtype_ = npx.dtypes.StringDType(**kwarg, coerce=False)
        return npx.asarray(np_array).astype(to_dtype_)

    if hasattr(np_array.dtype, "na_object") and to_na_object == _not_set:
        # We don't allow casting from missing-data to no-missing-data (for now)
        with pytest.raises(ValueError, match="is undefined"):
            do(ndx)
        return
    assert_array_equal(do(ndx), do(np))


@pytest.mark.parametrize("key", [0, -1, slice(None)])
def test_get_item(np_array, key):
    def do(npx):
        arr = npx.asarray(np_array)
        if isinstance(key, int) and npx == np:
            return np.asarray(arr[key], dtype=np_array.dtype)
        return arr[key]

    assert_array_equal(do(ndx), do(np))


def test_set_item(np_array):
    def do(npx):
        arr = npx.asarray(np_array)
        arr[0] = "FOOBAR"
        return arr

    assert_array_equal(do(ndx), do(np))


def test_set_item_nan():
    def do(npx):
        arr = npx.asarray(
            np.asarray(
                ["a", "b"], dtype=np.dtypes.StringDType(na_object=np.nan, coerce=False)
            )
        )
        arr[1] = np.nan
        return arr

    assert_array_equal(do(ndx), do(np))


def test_isnan(np_array):
    def do(npx):
        return npx.isnan(npx.asarray(np_array))

    assert_array_equal(do(ndx), do(np))
