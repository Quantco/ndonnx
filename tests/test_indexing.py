# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import array_api_strict as np_strict
import numpy as np
import pytest

import ndonnx as ndx
from ndonnx._typed_array.funcs import astyarray
from ndonnx._typed_array.onnx import _get_indices, _key_to_indices, _move_ellipsis_back


@pytest.mark.parametrize(
    "s, length",
    [
        (slice(None, None, None), 10),
        (slice(None, None, -1), 10),
        (slice(2, 5, None), 10),
        (slice(2, None, None), 10),
        (slice(None, 5, None), 10),
        (slice(None, -5, None), 10),
        (slice(-2, -5, -1), 10),
        (slice(-5, -2, -1), 10),
    ],
)
def test_get_indices(s: slice, length: int):
    py_res = _get_indices(s, length)
    assert py_res == tuple(astyarray(el) for el in s.indices(length))

    assert py_res == _get_indices(s, astyarray(length, ndx.int64))


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
    two = astyarray(2)
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
        np.bool_,
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


def test_getitem_too_long_key_tuple_raises():
    def do(npx):
        arr = npx.asarray([1, 2, 3])
        arr[(1, 2)]

    with pytest.raises(IndexError):
        do(np)
    with pytest.raises(IndexError):
        do(ndx)


def test_setitem_too_long_key_tuple_raises():
    def do(npx):
        arr = npx.asarray([1, 2, 3])
        arr[(1, 2)] = 42

    with pytest.raises(IndexError):
        do(np)
    with pytest.raises(IndexError):
        do(ndx)


def test_empty_data():
    def do(npx):
        arr = npx.ones((0, 0), dtype=npx.bool)
        key = (slice(None, None, None), slice(None, None, None))
        arr[key] = True
        return arr

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np_strict), strict=True)


def test_empty_update():
    def do(npx):
        arr = npx.ones((2, 2), dtype=npx.bool)
        key = (slice(0, 0, None), slice(None, None, None))
        arr[key] = ~arr[key]
        return arr

    np.testing.assert_array_equal(do(np_strict), do(ndx).unwrap_numpy(), strict=True)


def test_setitem_boolean_mask_all_false():
    def do(npx):
        arr = npx.asarray([1, 2], dtype=npx.int64)
        key = npx.asarray([False, False])
        arr[key] = arr[key]
        return arr

    np.testing.assert_array_equal(do(np_strict), do(ndx).unwrap_numpy(), strict=True)


def test_setitem_boolean_mask_value_is_array():
    # Check that we do the correct thing when the value is an array of
    # rank>0 such that it aligns with the compressed array it is
    # assigning to.
    def do(npx):
        arr = npx.asarray([1, 2, 3], dtype=npx.int64)
        key = npx.asarray([True, True, False])
        arr[key] = arr[key]
        return arr

    np.testing.assert_array_equal(do(np_strict), do(ndx).unwrap_numpy(), strict=True)


def test_scalar_array_key():
    def do(npx):
        arr = npx.asarray([1, 2, 3], dtype=npx.int64)
        key = npx.asarray(1, dtype=npx.int64)
        return arr[key]

    np.testing.assert_array_equal(do(np_strict), do(ndx).unwrap_numpy(), strict=True)


def test_index_scalar_with_empty_tuple():
    def do(npx):
        arr = npx.asarray(1, dtype=npx.int64)
        return arr[()]

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np_strict), strict=True)


def test_index_scalar_with_scalar_bool():
    def do(npx):
        arr = npx.asarray(1, dtype=npx.int64)
        key = npx.asarray(False)
        return arr[key]

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np_strict), strict=True)


def test_integer_array_indexing():
    def do(npx):
        arr = npx.reshape(npx.arange(0, 16, dtype=npx.int64), (4, 4))
        key = (npx.asarray([0, 1]), npx.asarray([2, 3]))

        return arr[key]

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np_strict), strict=True)


@pytest.mark.parametrize(
    "key",
    [
        ndx.asarray([1]),
        (ndx.asarray([1]),),
    ],
)
def test_setitem_fancy_indexing_integer_array_raises(key):
    arr = ndx.asarray([1.0, 1.0])

    with pytest.raises(IndexError, match="__setitem__"):
        arr[key] = 10.0
