# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from packaging.version import parse

import ndonnx as ndx

from .utils import assert_equal_dtype_shape


@pytest.mark.parametrize("k", [-1, 0, 1])
@pytest.mark.parametrize(
    "func",
    [
        np.tril,
        np.triu,
    ],
)
def test_trilu(func, k):
    a = np.ones((3, 3))
    expected = func(a, k=k)
    actual = getattr(ndx, func.__name__)(ndx.asarray(a), k=k)
    np.testing.assert_array_equal(expected, actual.unwrap_numpy())


def test_reshape_with_array():
    expected_shape = (2, 1)
    new_shape = ndx.asarray(np.array(expected_shape, np.int64))
    candidate_shape = ndx.reshape(
        ndx.asarray(np.array([[1, 2]], np.int64)), new_shape
    ).shape
    assert expected_shape == candidate_shape


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize(
    "np_arrays",
    [
        [np.asarray([[1], [2]]), np.asarray([[3.0], [4.0]])],
        [np.ma.array([[1], [2]]), np.ma.array([[3.0], [4.0]])],
        [np.ma.array([[1], [2]]), np.ma.array([[3.0], [4.0]], mask=[[True], [False]])],
    ],
)
@pytest.mark.skipif(
    parse(np.__version__).major < 2,
    reason="NumPy 1.x does not provide `concat` function",
)
def test_concat(np_arrays, axis):
    arrays = [ndx.asarray(arr) for arr in np_arrays]
    expected = np.concat(np_arrays, axis=axis)
    candidate = ndx.concat(arrays, axis=axis).unwrap_numpy()

    np.testing.assert_equal(expected, candidate)


@pytest.mark.parametrize("op", ["maximum", "minimum"])
@pytest.mark.parametrize(
    "np_dtype, ndx_dtype",
    [
        (np.dtype("int32"), ndx.int32),
        (np.dtype("datetime64[s]"), ndx.DateTime64DType("s")),
        (np.dtype("timedelta64[s]"), ndx.TimeDelta64DType("s")),
    ],
)
@pytest.mark.parametrize(
    "np_array1, np_array2",
    [
        (np.array([1, 2]), np.array([3])),
    ],
)
def test_min_max(op, ndx_dtype, np_dtype, np_array1, np_array2):
    arr1 = ndx.asarray(np_array1).astype(ndx_dtype)
    arr2 = ndx.asarray(np_array2).astype(ndx_dtype)

    candidate = getattr(ndx, op)(arr1, arr2).unwrap_numpy()
    expectation = getattr(np, op)(
        np_array1.astype(np_dtype), np_array2.astype(np_dtype)
    )

    np.testing.assert_array_equal(candidate, expectation)


@pytest.mark.parametrize(
    "x_ty, y_ty, res_ty",
    [
        (ndx.int16, ndx.int32, ndx.int32),
        (ndx.nint16, ndx.int32, ndx.nint32),
        (ndx.int32, ndx.nint16, ndx.nint32),
    ],
)
def test_where(x_ty, y_ty, res_ty):
    shape = ("N", "M")
    cond = ndx.argument(shape=shape, dtype=ndx.bool)
    x = ndx.argument(shape=shape, dtype=x_ty)
    y = ndx.argument(shape=shape, dtype=y_ty)

    res = ndx.where(cond, x, y)

    assert_equal_dtype_shape(res, res_ty, shape)


@pytest.mark.parametrize("shape", [(), (1,), (2, 2)])
@pytest.mark.parametrize("dtype", [None, ndx.int32, ndx.float64, ndx.utf8])
def test_ones(dtype, shape):
    candidate = ndx.ones(shape, dtype=dtype)
    assert candidate.dtype == dtype or ndx.float64

    if dtype is None:
        dtype = ndx._default_float
    np.testing.assert_equal(
        candidate.unwrap_numpy(), np.ones(shape, dtype=dtype.unwrap_numpy())
    )


@pytest.mark.parametrize(
    "a, b, axes",
    [
        (
            np.arange(60).reshape(3, 4, 5),
            np.arange(24).reshape(4, 3, 2),
            ([1, 0], [0, 1]),
        ),
        (np.arange(60).reshape(3, 4, 5), np.arange(60).reshape(4, 5, 3), 2),
        (np.arange(60).reshape(3, 4, 5), np.arange(60).reshape(4, 5, 3), 0),
        (np.arange(60).reshape(4, 5, 3), np.arange(60).reshape(4, 5, 3), 3),
        (np.arange(5).reshape(5), np.arange(5).reshape(5), 1),
        (np.arange(36).reshape(6, 6), np.arange(36).reshape(6, 6), 1),
        (np.arange(24).reshape(3, 2, 4), np.arange(24).reshape(4, 2, 3), 1),
        (np.arange(35).reshape(5, 7), np.arange(35).reshape(7, 5), 1),
        (np.arange(35).reshape(7, 5), np.arange(35).reshape(7, 5), 2),
        (np.arange(48).reshape(4, 3, 4), np.arange(48).reshape(4, 4, 3), 0),
        (
            np.arange(32).reshape(4, 4, 2),
            np.arange(32).reshape(2, 4, 4),
            ([2, 0], [0, 1]),
        ),
        (np.arange(30).reshape(3, 10), np.arange(20).reshape(10, 2), ([1], [0])),
    ],
)
def test_tensordot(a, b, axes):
    np_result = np.tensordot(a, b, axes=axes)
    ndx_result = ndx.tensordot(ndx.asarray(a), ndx.asarray(b), axes=axes)
    np.testing.assert_array_equal(np_result, ndx_result.unwrap_numpy(), strict=True)


@pytest.mark.parametrize(
    "a, b",
    [
        (np.arange(60).reshape(3, 4, 5), np.arange(60).reshape(4, 5, 3)),
    ],
)
def test_tensordot_no_axes(a, b):
    np_result = np.tensordot(a, b)
    ndx_result = ndx.tensordot(ndx.asarray(a), ndx.asarray(b)).unwrap_numpy()
    np.testing.assert_array_equal(np_result, ndx_result, strict=True)


def test_raise_if_neither_argument_is_array():
    with pytest.raises(TypeError, match="add"):
        ndx.add(1, 1)
