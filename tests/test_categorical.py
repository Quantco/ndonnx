# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import ndonnx._refactor as ndx
from ndonnx._refactor._typed_array.categorical import CategoricalDType


def test_dtype_repr():
    dtype = CategoricalDType(categories=["a", "b"], ordered=True)

    assert repr(dtype) == "CategoricalDType(categories=['a', 'b'], ordered=True)"


def test_dtype_schema():
    dtype = CategoricalDType(categories=["a", "b"], ordered=True)

    assert dtype.__ndx_infov1__.meta == {"categories": ["a", "b"], "ordered": True}


def test_cast_to_and_from_utf8():
    dtype = CategoricalDType(categories=["a", "b"], ordered=True)
    initial = ndx.asarray(["a", "b"])
    cat_arr = initial.astype(dtype)
    np.testing.assert_equal(
        initial.unwrap_numpy(), cat_arr.astype(ndx.utf8).unwrap_numpy()
    )


def test_equal():
    dtype = CategoricalDType(categories=["a", "b"], ordered=True)
    arr1 = ndx.asarray(np.array(["a", "a", "a", "c"])).astype(dtype)
    arr2 = ndx.asarray(np.array(["a", "b", "c", "c"]))
    expected = np.array([True, False, False, False])

    # test equal to utf8
    res = arr1 == arr2
    np.testing.assert_equal(res.unwrap_numpy(), expected)

    # test equal to categorical of same type
    res = arr1 == arr2.astype(dtype)
    np.testing.assert_equal(res.unwrap_numpy(), expected)

    with pytest.raises(TypeError):
        _ = arr1 == arr2.astype(CategoricalDType(categories=["foo"], ordered=True))


def test_setitem():
    dtype = CategoricalDType(categories=["a", "b"], ordered=True)
    arr = ndx.asarray(np.array(["a", "b", "c"])).astype(dtype)

    arr[0] = "FOO"
    expected = np.array([np.nan, "b", np.nan], object)
    candidate = arr.unwrap_numpy()
    assert expected.dtype == candidate.dtype
    assert_equal_object_array(expected, candidate)


def assert_equal_object_array(desired, actual):
    assert desired.shape == actual.shape
    assert desired.dtype == actual.dtype
    for a, b in zip(desired, actual):
        if a == b:
            continue
        if np.isnan(a) and np.isnan(b):
            continue
        if a is None and b is None:
            continue
        raise AssertionError(f"`{a}` is not equivalent to `{b}`")
