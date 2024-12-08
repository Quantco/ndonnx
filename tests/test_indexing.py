# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import ndonnx._refactor as ndx
from ndonnx._refactor import _typed_array as tydx
from ndonnx._refactor._typed_array.indexing import (
    _get_indices,
    _key_to_indices,
    _move_ellipsis_back,
)


@pytest.mark.parametrize(
    "s, length",
    [
        # (slice(None, None, None), 10),
        (slice(None, None, -1), 10),
        # (slice(2, 5, None), 10),
        # (slice(2, None, None), 10),
        # (slice(None, 5, None), 10),
        # (slice(None, -5, None), 10),
        # (slice(-2, -5, -1), 10),
        # (slice(-5, -2, -1), 10),
    ],
)
def test_get_indices(s: slice, length: int):
    py_res = _get_indices(s, length)
    assert py_res == tuple(tydx.astyarray(el) for el in s.indices(length))

    assert py_res == _get_indices(s, tydx.astyarray(length, ndx.int64))


@pytest.mark.parametrize(
    "key",
    [
        (1, 1, 1),
        (1,),
        (-1,),
        (-1, 1),
        (slice(None),),
        (slice(None, None, -1),),
        (slice(0, 1, 1),),
        (slice(0, 2, 1),),
        (slice(0, 2, -1),),
        (slice(-2, 0, -1),),
        (slice(0, 2, 1), slice(0, 2, 1)),
        (slice(0, 2, 1), -1, slice(0, 2, 1)),
        (slice(0, 100, 10),),
        # empty slice
        (slice(0, 0, None),),
    ],
)
def test_key_to_indices(key):
    np_arr = np.ones((2, 2, 2), np.int64)
    arr = ndx.asarray(np_arr)

    shape = arr._tyarray.dynamic_shape
    arr_key = _key_to_indices(key, shape)
    two = tydx.astyarray(2)
    arr._tyarray._setitem_int_array(arr_key, two)

    np_arr[tuple(key)] = 2
    np.testing.assert_array_equal(arr.unwrap_numpy(), np_arr)


def test_move_ellipsis():
    ndim = 3
    assert ((1, 2, 0), (2, 0, 1), (-1, 1)) == _move_ellipsis_back(ndim, (..., -1, 1))


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.bool,
        np.str_,
    ],
)
def test_set_index_fancy(dtype):
    # make sure we have double digits since NumPy picks `<U1` as the string type
    np_arr = np.array(list(range(10, 19)), dtype=dtype).reshape((3, 3))
    arr = ndx.asarray(np_arr)

    key = (1, ..., slice(0, 2, 1))

    np_arr[key] = 42
    arr[key] = 42

    np.testing.assert_array_equal(np_arr, arr.unwrap_numpy())


def test_empty_data():
    np_arr = np.ones((0, 0), dtype=bool)
    key = (slice(None, None, None), slice(None, None, None))
    arr = ndx.asarray(np_arr)

    arr[key] = True
    np_arr[key] = True

    np.testing.assert_array_equal(arr.unwrap_numpy(), np_arr)


def test_empty_update():
    np_arr = np.ones((2, 2), dtype=bool)
    key = (slice(0, 0, None), slice(None, None, None))
    arr = ndx.asarray(np_arr)

    np_arr[key] = ~np_arr[key]
    update = ~arr[key]
    arr[key] = update

    np.testing.assert_array_equal(arr.unwrap_numpy(), np_arr)