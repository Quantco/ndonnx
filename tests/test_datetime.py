# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import operator

import numpy as np
import pytest

import ndonnx._refactor as ndx

from .utils import assert_equal_dtype_shape


def test_value_prop_datetime():
    arr = ndx.asarray(np.asarray([1, 2])).astype(ndx.DateTime("s"))
    np.testing.assert_equal(
        arr.unwrap_numpy(), np.asarray([1, 2], dtype="datetime64[s]")
    )


def test_arithmetic():
    arr = ndx.Array(shape=("N",), dtype=ndx.DateTime("s"))
    one_s_td = (arr + 1) - arr
    assert one_s_td.dtype == ndx.TimeDelta("s")

    ten_s_td = one_s_td * 10

    res = arr + ten_s_td
    assert res.dtype == ndx.DateTime("s")


@pytest.mark.parametrize("unit", ["s", "ns"])
@pytest.mark.parametrize("ty", ["datetime64", "timedelta64"])
def test_datetime_from_np_array(ty, unit):
    np_arr = np.array([0, 1], f"{ty}[{unit}]")
    arr = ndx.asarray(np_arr)
    np.testing.assert_array_equal(arr.unwrap_numpy(), np_arr, strict=True)


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.sub,
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, res_dtype",
    [
        # TODO: Implement other units
        (1, ndx.DateTime("s"), ndx.DateTime("s")),
    ],
)
def test_add_pyscalar_datetime(scalar, dtype, res_dtype, op):
    shape = ("N",)
    arr = ndx.Array(shape=shape, dtype=dtype)

    assert_equal_dtype_shape(op(scalar, arr), res_dtype, shape)
    assert_equal_dtype_shape(op(arr, scalar), res_dtype, shape)


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.sub,
        operator.mul,
    ],
)
def test_arithmetic_pyscalar_timedelta(op):
    shape = ("N",)
    scalar = 1
    arr = ndx.Array(shape=shape, dtype=ndx.TimeDelta("s"))

    expected_dtype = ndx.TimeDelta("s")
    assert_equal_dtype_shape(op(scalar, arr), expected_dtype, shape)
    assert_equal_dtype_shape(op(arr, scalar), expected_dtype, shape)


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.sub,
    ],
)
def test_arithmetic_timedelta_timedelta(op):
    shape = ("N",)
    arr = ndx.Array(shape=shape, dtype=ndx.TimeDelta("s"))

    expected_dtype = ndx.TimeDelta("s")
    assert_equal_dtype_shape(op(arr, arr), expected_dtype, shape)
    assert_equal_dtype_shape(op(arr, arr), expected_dtype, shape)


def test_arithmetic_timedelta_datetime():
    shape = ("N",)
    arr_dt = ndx.Array(shape=shape, dtype=ndx.DateTime("s"))
    arr_td = ndx.Array(shape=shape, dtype=ndx.TimeDelta("s"))

    expected_dtype = ndx.DateTime("s")
    assert_equal_dtype_shape(arr_dt + arr_td, expected_dtype, shape)
    assert_equal_dtype_shape(arr_td + arr_dt, expected_dtype, shape)

    assert_equal_dtype_shape(arr_dt - arr_td, expected_dtype, shape)

    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = arr_td - arr_dt


@pytest.mark.parametrize(
    "op", [operator.gt, operator.lt, operator.ne, operator.eq, operator.le, operator.ge]
)
@pytest.mark.parametrize("x, y", [(["1900-01-11", "NaT"], ["NaT", "1900-01-12"])])
@pytest.mark.parametrize("unit", ["s"])
def test_comparisons(op, x, y, unit):
    np_x = np.array(x, f"datetime64[{unit}]")
    np_y = np.array(y, f"datetime64[{unit}]")

    desired = op(np_x, np_y)
    actual = op(ndx.asarray(np_x), ndx.asarray(np_y))

    np.testing.assert_equal(actual.unwrap_numpy(), desired, strict=True)


def test_isnan():
    np_arr = np.asarray(["NaT", 1], dtype="datetime64[s]")
    arr = ndx.asarray(np_arr)

    np.testing.assert_array_equal(
        ndx.isnan(arr).unwrap_numpy(), np.isnan(np_arr), strict=True
    )


@pytest.mark.parametrize("dtype", ["datetime64[s]", "timedelta64[s]"])
def test_where(dtype):
    cond = np.asarray([False, True, False])
    np_arr1 = np.asarray(["NaT", 1, 2], dtype=dtype)
    np_arr2 = np.asarray(["NaT", "NaT", "NaT"], dtype=dtype)

    expected = np.where(cond, np_arr1, np_arr2)
    actual = ndx.where(ndx.asarray(cond), ndx.asarray(np_arr1), ndx.asarray(np_arr2))

    np.testing.assert_array_equal(actual.unwrap_numpy(), expected, strict=True)


@pytest.mark.parametrize("data", [[], ["Nat", 1, 3]])
@pytest.mark.parametrize("min", [None, 0, 2, 5, "NaT"])
@pytest.mark.parametrize("max", [None, 0, 2, 5, "NaT"])
def test_clip(data, min, max):
    if min is None and max is None:
        pytest.skip("'clip' is not defined in NumPy if both 'min' and 'max' are 'None'")
    dtype = "datetime64[s]"
    np_arr = np.asarray(data, dtype=dtype)
    min = None if min is None else np.asarray(min, dtype=dtype)
    max = None if max is None else np.asarray(max, dtype=dtype)

    desired = np.clip(np_arr, min, max)
    actual = ndx.clip(
        ndx.asarray(np_arr),
        min=None if min is None else ndx.asarray(min),
        max=None if max is None else ndx.asarray(max),
    ).unwrap_numpy()

    np.testing.assert_array_equal(actual, desired, strict=True)
