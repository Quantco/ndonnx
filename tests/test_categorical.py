# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import ndonnx._refactor as ndx
from ndonnx._refactor._typed_array.categorical import CategoricalDType


def test_dtype_repr():
    dtype = CategoricalDType(categories=["a", "b"], ordered=True)

    assert repr(dtype) == "CategoricalDType(categories=['a', 'b'], ordered=True)"


def test_dtype_schema():
    dtype = CategoricalDType(categories=["a", "b"], ordered=True)

    assert dtype._infov1.meta == {"categories": ["a", "b"], "ordered": True}


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
    arr2 = ndx.asarray(np.array(["a", "b", "c", "c"])).astype(dtype)

    res = arr1 == arr2
    expected = np.array([True, False, False, False])

    np.testing.assert_equal(res.unwrap_numpy(), expected)
