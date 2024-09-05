# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from ndonnx._logic_in_data import Array, dtypes
from ndonnx._logic_in_data._typed_array.date_time import DateTime, TimeDelta
from ndonnx._logic_in_data.array import asarray, where


@pytest.mark.parametrize(
    "scalar, dtype, res_dtype",
    [
        (1, dtypes.int32, dtypes.int32),
        (1.0, dtypes.int32, dtypes.float64),
        (1.0, dtypes.float32, dtypes.float32),
        (1.0, dtypes.float64, dtypes.float64),
        (1.0, dtypes.nfloat64, dtypes.nfloat64),
        (1.0, dtypes.nint32, dtypes.nfloat64),
    ],
)
def test_add_pyscalar(scalar, dtype, res_dtype):
    shape = ("N",)
    res = scalar + Array(shape, dtype)

    assert res.dtype == res_dtype
    assert res._data.shape == shape
    assert res.shape == (None,)

    # Same check for scalar on rhs
    res = Array(shape, dtype) + scalar

    assert res.dtype == res_dtype
    assert res._data.shape == shape
    assert res.shape == (None,)


@pytest.mark.parametrize(
    "dtype1, dtype2, res_dtype",
    [
        (dtypes.int32, dtypes.int32, dtypes.int32),
        (dtypes.int64, dtypes.int32, dtypes.int64),
        (dtypes.float64, dtypes.int32, dtypes.float64),
        (dtypes.nint32, dtypes.int32, dtypes.nint32),
    ],
)
def test_core_add(dtype1, dtype2, res_dtype):
    shape = ("N",)
    res = Array(shape, dtype1) + Array(shape, dtype2)

    assert res.dtype == res_dtype
    assert res._data.shape == shape
    assert res.shape == (None,)


@pytest.mark.parametrize(
    "dtype1, dtype2, res_dtype",
    [
        (dtypes.int32, dtypes.int32, dtypes.int32),
        (dtypes.int64, dtypes.int32, dtypes.int64),
        (dtypes.bool_, dtypes.bool_, dtypes.bool_),
    ],
)
def test_core_or(dtype1, dtype2, res_dtype):
    shape = ("N",)
    res = Array(shape, dtype1) | Array(shape, dtype2)

    assert res.dtype == res_dtype
    assert res._data.shape == shape
    assert res.shape == (None,)


def test_value_prop():
    arr = Array(value=1)
    np.testing.assert_allclose((arr + arr).unwrap_numpy(), np.array(2))

    with pytest.raises(ValueError, match="no propagated value available"):
        Array(("N",), dtypes.int32).unwrap_numpy()


def test__getitem__():
    arr = Array(("N", "M"), dtypes.nint32)
    assert arr[0]._data.shape == ("M",)
    assert arr[0].shape == (None,)


@pytest.mark.parametrize(
    "x_ty, y_ty, res_ty",
    [
        (dtypes.int16, dtypes.int32, dtypes.int32),
        (dtypes.nint16, dtypes.int32, dtypes.nint32),
        (dtypes.int32, dtypes.nint16, dtypes.nint32),
    ],
)
def test_where(x_ty, y_ty, res_ty):
    shape = ("N", "M")
    cond = Array(shape, dtypes.bool_)
    x = Array(shape, x_ty)
    y = Array(shape, y_ty)

    res = where(cond, x, y)

    assert res.dtype == res_ty
    assert res._data.shape == shape
    assert res.shape == (None, None)


def test_datetime():
    arr = Array(("N",), DateTime("s"))
    one_s_td = (arr + 1) - arr
    assert one_s_td.dtype == TimeDelta("s")

    ten_s_td = one_s_td * 10

    res = arr + ten_s_td
    assert res.dtype == DateTime("s")


def test_datetime_value_prop():
    arr = asarray(np.asarray([1, 2])).astype(DateTime("s"))
    np.testing.assert_equal(arr.to_numpy(), np.asarray([1, 2], dtype="datetime64[s]"))
