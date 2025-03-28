# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import operator
from datetime import datetime, timedelta
from typing import get_args

import numpy as np
import pytest

import ndonnx as ndx
from ndonnx import from_numpy_dtype
from ndonnx._typed_array.datetime import Unit

from .utils import assert_equal_dtype_shape

_NAT_SENTINEL = np.iinfo(np.int64).min


@pytest.fixture(params=get_args(Unit))
def unit(request):
    return request.param


@pytest.mark.parametrize("cls", [ndx.DateTime64DType, ndx.TimeDelta64DType])
def test_dtype_constructor_valid_units(unit, cls):
    cls(unit=unit)


@pytest.mark.parametrize("cls", [ndx.DateTime64DType, ndx.TimeDelta64DType])
def test_dtype_constructor_invalid_unit(cls):
    with pytest.raises(TypeError):
        cls(unit="foo")


def test_value_prop_datetime(unit: Unit):
    arr = ndx.asarray(np.asarray([1, 2])).astype(ndx.DateTime64DType(unit))
    np.testing.assert_equal(
        arr.unwrap_numpy(), np.asarray([1, 2], dtype=f"datetime64[{unit}]")
    )


def test_arithmetic(unit: Unit):
    arr_np = np.array([1, 2, "NaT"], f"datetime64[{unit}]")
    arr = ndx.asarray(arr_np)
    one_s_td = (arr + 1) - arr
    one_s_td_np = (arr_np + 1) - arr_np

    assert one_s_td.dtype == ndx.TimeDelta64DType(unit)
    np.testing.assert_array_equal(one_s_td.unwrap_numpy(), one_s_td_np, strict=True)


@pytest.mark.parametrize("ty", ["datetime64", "timedelta64"])
def test_datetime_from_np_array(ty, unit):
    np_arr = np.array([0, 1], f"{ty}[{unit}]")
    arr = ndx.asarray(np_arr)
    np.testing.assert_array_equal(arr.unwrap_numpy(), np_arr, strict=True)


@pytest.mark.parametrize(
    "scalar, dtype, res_dtype",
    [
        (1, ndx.DateTime64DType("s"), ndx.DateTime64DType("s")),
        (1, ndx.DateTime64DType("ns"), ndx.DateTime64DType("ns")),
    ],
)
def test_add_pyscalar_datetime(scalar, dtype, res_dtype):
    shape = ("N",)
    arr = ndx.argument(shape=shape, dtype=dtype)

    assert_equal_dtype_shape(scalar + arr, res_dtype, shape)
    assert_equal_dtype_shape(arr + scalar, res_dtype, shape)


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.sub,
        operator.mul,
    ],
)
def test_arithmetic_pyscalar_timedelta(op, unit: Unit):
    scalar = 1

    def do(npx, forward):
        dtype = (
            ndx.TimeDelta64DType(unit)
            if npx == ndx
            else np.dtype(f"timedelta64[{unit}]")
        )
        arr = npx.asarray([-10000, 0, int(1e9)]).astype(dtype=dtype)
        return op(scalar, arr) if forward else op(arr, scalar)

    np.testing.assert_array_equal(
        do(np, True), do(ndx, True).unwrap_numpy(), strict=True
    )

    np.testing.assert_array_equal(
        do(np, False), do(ndx, False).unwrap_numpy(), strict=True
    )


def test_divide_timedelta(unit: Unit):
    # Fail when dividing int by timedelta
    with pytest.raises(TypeError):
        _ = 1 / ndx.asarray(np.asarray([1]), dtype=ndx.TimeDelta64DType(unit))

    def do(npx):
        dtype = (
            ndx.TimeDelta64DType(unit)
            if npx == ndx
            else np.dtype(f"timedelta64[{unit}]")
        )
        return npx.asarray([100, np.iinfo(np.int64).min]).astype(dtype) / 10

    np.testing.assert_array_equal(do(np), do(ndx).unwrap_numpy())


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.sub,
    ],
)
def test_arithmetic_timedelta_timedelta(op, unit):
    shape = ("N",)
    arr = ndx.argument(shape=shape, dtype=ndx.TimeDelta64DType(unit))

    expected_dtype = ndx.TimeDelta64DType(unit)
    assert_equal_dtype_shape(op(arr, arr), expected_dtype, shape)
    assert_equal_dtype_shape(op(arr, arr), expected_dtype, shape)


def test_arithmetic_timedelta_datetime_lazy(unit):
    shape = ("N",)
    arr_dt = ndx.argument(shape=shape, dtype=ndx.DateTime64DType(unit))
    arr_td = ndx.argument(shape=shape, dtype=ndx.TimeDelta64DType(unit))

    expected_dtype = ndx.DateTime64DType(unit)
    assert_equal_dtype_shape(arr_dt + arr_td, expected_dtype, shape)
    assert_equal_dtype_shape(arr_td + arr_dt, expected_dtype, shape)

    assert_equal_dtype_shape(arr_dt - arr_td, expected_dtype, shape)

    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = arr_td - arr_dt


def test_arithmetic_datetime_timedelta(unit):
    np_arr = np.array(
        [datetime(year=1982, month=5, day=24, hour=12, second=1)], f"datetime64[{unit}]"
    )
    np_delta = np.asarray(timedelta(days=5), dtype=f"timedelta64[{unit}]")
    np_res = np_arr + np_delta

    arr = ndx.asarray(np_arr)
    delta = ndx.asarray(np_delta)
    res = arr + delta

    np.testing.assert_array_equal(np_res, res.unwrap_numpy(), strict=True)


@pytest.mark.parametrize(
    "op", [operator.gt, operator.lt, operator.ne, operator.eq, operator.le, operator.ge]
)
@pytest.mark.parametrize(
    "x, y",
    [
        ([_NAT_SENTINEL], [1000]),
        ([1000], [_NAT_SENTINEL]),
        ([1000], [2000]),
        ([_NAT_SENTINEL], [_NAT_SENTINEL]),
    ],
)
def test_comparisons_timedelta(op, x, y, unit):
    np_x = np.array(x, f"timedelta64[{unit}]")
    np_y = np.array(y, f"timedelta64[{unit}]")

    desired = op(np_x, np_y)
    actual = op(ndx.asarray(np_x), ndx.asarray(np_y))

    np.testing.assert_array_equal(actual.unwrap_numpy(), desired, strict=True)


@pytest.mark.parametrize(
    "op", [operator.gt, operator.lt, operator.ne, operator.eq, operator.le, operator.ge]
)
@pytest.mark.parametrize(
    "x, y",
    [
        (["NaT"], ["1900-01-12"]),
        (["1900-01-12"], ["NaT"]),
        (["1900-01-11"], ["1900-01-12"]),
        (["NaT"], ["NaT"]),
    ],
)
def test_comparisons_datetime(op, x, y, unit):
    np_x = np.array(x, f"datetime64[{unit}]")
    np_y = np.array(y, f"datetime64[{unit}]")

    desired = op(np_x, np_y)
    actual = op(ndx.asarray(np_x), ndx.asarray(np_y))

    np.testing.assert_array_equal(actual.unwrap_numpy(), desired, strict=True)


@pytest.mark.parametrize(
    "x, y",
    [
        (["NaT"], ["1900-01-12"]),
        (["1900-01-12"], ["NaT"]),
        (["1900-01-11"], ["1900-01-12"]),
        (["NaT"], ["NaT"]),
    ],
)
@pytest.mark.parametrize("forward", [True, False])
def test_subtraction_datetime_arrays(x, y, unit, forward):
    np_x = np.array(x, f"datetime64[{unit}]")
    np_y = np.array(y, f"datetime64[{unit}]")

    desired = np_x - np_y if forward else np_y - np_x
    actual = (
        ndx.asarray(np_x) - ndx.asarray(np_y)
        if forward
        else ndx.asarray(np_y) - ndx.asarray(np_x)
    )

    np.testing.assert_array_equal(actual.unwrap_numpy(), desired, strict=True)


@pytest.mark.parametrize("x", ["NaT", "1900-01-12"])
def test_subtraction_datetime_scalar(x, unit):
    np_x = np.array(x, f"datetime64[{unit}]")
    scalar = 42

    desired = np_x - scalar
    actual = ndx.asarray(np_x) - scalar

    np.testing.assert_array_equal(actual.unwrap_numpy(), desired, strict=True)


def test_scalar_minus_datetime_raises():
    x = ndx.asarray(np.array(1000, "datetime64[s]"))

    with pytest.raises(TypeError):
        _ = 42 - x


def test_timedelta_minus_datetime_raises():
    dt = ndx.asarray(np.array(1000, "datetime64[s]"))
    td = ndx.asarray(np.array(1000, "timedelta64[s]"))

    with pytest.raises(TypeError):
        _ = td - dt


def test_isnan(unit):
    np_arr = np.asarray(["NaT", 1], dtype=f"datetime64[{unit}]")
    arr = ndx.asarray(np_arr)

    np.testing.assert_array_equal(
        ndx.isnan(arr).unwrap_numpy(), np.isnan(np_arr), strict=True
    )


@pytest.mark.parametrize(
    "dtype", ["datetime64[s]", "timedelta64[s]", "datetime64[ns]", "timedelta64[ns]"]
)
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
def test_clip(data, min, max, unit):
    if min is None and max is None:
        pytest.skip("'clip' is not defined in NumPy if both 'min' and 'max' are 'None'")
    dtype = f"datetime64[{unit}]"
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


@pytest.mark.parametrize(
    "date",
    [
        "1970-01-01",
        "2070-01-01",
        "2370-01-01",
        "1670-01-01",
        "1270-01-01",
        "1270-01-02",
        "1999-02-28",
        "1999-03-01",
        "2000-02-29",
        "2000-03-01",
    ],
)
def test_day_month_year(date, unit):
    np_arr = np.asarray(date, dtype=f"datetime64[{unit}]")
    arr = ndx.asarray(np_arr)

    np_y = np_arr.astype("datetime64[Y]").astype(int) + 1970
    np_m = np_arr.astype("datetime64[M]").astype(int) % 12 + 1
    np_d = (np_arr.astype("datetime64[D]") - np_arr.astype("datetime64[M]")).astype(
        int
    ) + 1

    y, m, d = ndx.extensions.datetime_to_year_month_day(arr)

    np.testing.assert_array_equal(y.unwrap_numpy(), np_y)
    np.testing.assert_array_equal(m.unwrap_numpy(), np_m)
    np.testing.assert_array_equal(d.unwrap_numpy(), np_d)


@pytest.mark.parametrize("from_unit", get_args(Unit))
@pytest.mark.parametrize("to_unit", get_args(Unit))
@pytest.mark.parametrize("time_dtype", ["datetime64", "timedelta64"])
def test_unit_conversion_preserves_nat(from_unit, to_unit, time_dtype):
    arr = ndx.asarray(np.asarray(["NaT"], f"{time_dtype}[{from_unit}]"))
    ndx_to_dtype = from_numpy_dtype(np.dtype(f"{time_dtype}[{to_unit}]"))
    assert ndx.isnan(arr.astype(ndx_to_dtype)).unwrap_numpy()


@pytest.mark.parametrize("from_unit", get_args(Unit))
@pytest.mark.parametrize("to_unit", get_args(Unit))
@pytest.mark.parametrize("time_dtype", ["datetime64", "timedelta64"])
def test_unit_conversion(from_unit, to_unit, time_dtype):
    # make sure we are above 1e9 so that a conversion from ns to s is lossless
    np_arr0 = np.asarray([int(1e12), -int(1e12)], f"{time_dtype}[{from_unit}]")
    np_to_dtype = np.dtype(f"{time_dtype}[{to_unit}]")
    np_arr1 = np_arr0.astype(np_to_dtype)
    arr0 = ndx.asarray(np_arr0)
    arr1 = arr0.astype(from_numpy_dtype(np_to_dtype))

    np.testing.assert_array_equal(arr1.unwrap_numpy(), np_arr1, strict=True)


@pytest.mark.parametrize("time_dtype", ["datetime64", "timedelta64"])
def test_round_trip_int64(unit, time_dtype):
    arr = ndx.asarray(np.asarray([1_000_000, -1_000_000], f"{time_dtype}[{unit}]"))

    ndx_dtype = from_numpy_dtype(np.dtype(f"{time_dtype}[{unit}]"))
    np.testing.assert_array_equal(
        (arr.astype(ndx.int64).astype(ndx_dtype) == arr).unwrap_numpy(), [True, True]
    )


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.sub,
        operator.truediv,
    ],
)
@pytest.mark.parametrize("unit1", get_args(Unit))
@pytest.mark.parametrize("unit2", get_args(Unit))
def test_timedelta_arithmetic(op, unit1, unit2):
    lhs = np.asarray(
        [timedelta(days=1), timedelta(days=2), timedelta(days=3)],
        dtype=f"timedelta64[{unit1}]",
    )
    rhs = np.asarray(
        [timedelta(days=1), np.timedelta64("NaT"), timedelta(days=2)],
        dtype=f"timedelta64[{unit2}]",
    )
    pd_result = op(lhs, rhs)

    onnx_result = op(ndx.asarray(lhs), ndx.asarray(rhs))
    np.testing.assert_array_equal(pd_result, onnx_result.unwrap_numpy(), strict=True)


def test_timedelta_dtypes_have_numpy_repr(unit):
    dtype = ndx.TimeDelta64DType(unit)

    assert dtype.unwrap_numpy() == np.dtype(f"timedelta64[{unit}]")


def test_datetime_dtypes_have_numpy_repr(unit):
    dtype = ndx.DateTime64DType(unit)

    assert dtype.unwrap_numpy() == np.dtype(f"datetime64[{unit}]")
