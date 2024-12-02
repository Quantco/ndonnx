# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
import json
import operator

import numpy as np
import pytest

import ndonnx._refactor as ndx
from ndonnx._refactor import _dtypes as dtypes

from .utils import assert_equal_dtype_shape


def build_and_run(fn, *np_args):
    # Only works for core data types
    import onnxruntime as ort

    ins_np = {f"in{i}": arr for i, arr in enumerate(np_args)}
    ins = {
        k: ndx.Array(shape=a.shape, dtype=dtypes.from_numpy(a.dtype))
        for k, a in ins_np.items()
    }

    out = {"out": fn(*ins.values())}
    mp = ndx.build(ins, out)
    session = ort.InferenceSession(mp.SerializeToString())
    (out,) = session.run(None, {f"{k}": a for k, a in ins_np.items()})
    return out


def constant_prop(fn, *np_args):
    return fn(*[ndx.asarray(a) for a in np_args]).unwrap_numpy()


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.sub,
        operator.mul,
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, res_dtype",
    [
        (1, ndx.int32, ndx.int32),
        (1.0, ndx.int32, ndx.float64),
        (1.0, ndx.float32, ndx.float32),
        (1.0, ndx.float64, ndx.float64),
        (1.0, ndx.nfloat64, ndx.nfloat64),
        (1.0, ndx.nint32, ndx.nfloat64),
    ],
)
def test_ops_pyscalar_coretypes(scalar, dtype, res_dtype, op):
    shape = ("N",)
    arr = ndx.Array(shape=shape, dtype=dtype)

    assert_equal_dtype_shape(op(scalar, arr), res_dtype, shape)
    assert_equal_dtype_shape(op(arr, scalar), res_dtype, shape)


@pytest.mark.parametrize(
    "dtype1, dtype2, res_dtype",
    [
        (ndx.int32, ndx.int32, ndx.int32),
        (ndx.int64, ndx.int32, ndx.int64),
        (ndx.float64, ndx.int32, ndx.float64),
        (ndx.nint32, ndx.int32, ndx.nint32),
    ],
)
def test_type_promotion_standard_types(dtype1, dtype2, res_dtype):
    shape = ("N",)
    res = ndx.Array(shape=shape, dtype=dtype1) + ndx.Array(shape=shape, dtype=dtype2)

    assert_equal_dtype_shape(res, res_dtype, shape)


@pytest.mark.parametrize(
    "dtype1, dtype2, res_dtype",
    [
        (ndx.int32, ndx.int32, ndx.int32),
        (ndx.int64, ndx.int32, ndx.int64),
        (ndx.bool, ndx.bool, ndx.bool),
    ],
)
def test_type_promotion_or(dtype1, dtype2, res_dtype):
    shape = ("N",)
    res = ndx.Array(shape=shape, dtype=dtype1) | ndx.Array(shape=shape, dtype=dtype2)

    assert_equal_dtype_shape(res, res_dtype, shape)


def test_value_prop():
    arr = ndx.Array(value=1)
    np.testing.assert_allclose((arr + arr).unwrap_numpy(), np.array(2))

    with pytest.raises(ValueError, match="no propagated value available"):
        ndx.Array(shape=("N",), dtype=ndx.int32).unwrap_numpy()


def test_value_prop_datetime():
    arr = ndx.asarray(np.asarray([1, 2])).astype(ndx.DateTime("s"))
    np.testing.assert_equal(
        arr.unwrap_numpy(), np.asarray([1, 2], dtype="datetime64[s]")
    )


def test_datetime():
    arr = ndx.Array(shape=("N",), dtype=ndx.DateTime("s"))
    one_s_td = (arr + 1) - arr
    assert one_s_td.dtype == ndx.TimeDelta("s")

    ten_s_td = one_s_td * 10

    res = arr + ten_s_td
    assert res.dtype == ndx.DateTime("s")


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
def test_add_pyscalar_timedelta(op):
    shape = ("N",)
    scalar = 1
    arr = ndx.Array(shape=shape, dtype=ndx.TimeDelta("s"))

    expected_dtype = ndx.TimeDelta("s")
    assert_equal_dtype_shape(op(scalar, arr), expected_dtype, shape)
    assert_equal_dtype_shape(op(arr, scalar), expected_dtype, shape)


@pytest.mark.parametrize(
    "fun",
    [
        operator.add,
        operator.eq,
        operator.ge,
        operator.gt,
        operator.le,
        operator.lt,
        operator.mod,
        operator.mul,
        operator.ne,
        operator.pow,
        operator.sub,
        operator.truediv,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ],
)
@pytest.mark.parametrize("values", [[], 1, [1], [1, 2], [[1], [2]]])
def test_numerical_ops_with_ort_compat(dtype, values, fun):
    np_arr = np.asarray(values, dtype=dtype)

    expected = fun(np_arr, np_arr)

    candidate = build_and_run(fun, np_arr, np_arr)
    np.testing.assert_equal(candidate, expected)

    candidate = constant_prop(fun, np_arr, np_arr)
    np.testing.assert_equal(expected, candidate)


def test_indexing_shape():
    arr = ndx.Array(shape=("N", "M"), dtype=ndx.nint32)
    assert arr[0, :]._tyarray.shape == ("M",)
    assert arr[0, :].shape == (None,)


@pytest.mark.parametrize(
    "idx", [(0, ...), (-1, ...), (0, ...), (..., 0), (None, ..., -1)]
)
@pytest.mark.parametrize("np_array", [np.asarray([[1, 2]]), np.asarray([1, 2])])
def test_indexing_value_prop_scalar_index(np_array, idx):
    arr = ndx.asarray(np_array)
    assert arr[idx].shape == np_array[idx].shape
    assert arr[idx].dtype == arr.dtype
    np.testing.assert_equal(arr[idx].unwrap_numpy(), np_array[idx])


@pytest.mark.parametrize("np_array", [np.asarray([[1, 2]]), np.asarray([1, 2])])
def test_indexing_value_prop_scalar_slice(np_array):
    arr = ndx.asarray(np_array)
    idx = (slice(None, 1), ...)
    assert arr[idx].shape == np_array[idx].shape
    assert arr[idx].dtype == arr.dtype
    np.testing.assert_equal(arr[idx].unwrap_numpy(), np_array[idx])


def test_indexing_value_prop_tuple_index():
    np_arr = np.asarray([[1, 2]])
    arr = ndx.asarray(np_arr)
    for idx in np.ndindex(arr.shape):  # type: ignore
        el = arr[idx]
        assert el.shape == ()
        assert el.dtype == arr.dtype
        np.testing.assert_equal(el.unwrap_numpy(), np_arr[idx])


@pytest.mark.parametrize("idx", [(0, 1), (-1, ...), (..., 1), (-1, ..., 1)])
@pytest.mark.parametrize(
    "np_array",
    [np.asarray([[42, 42]]), np.ma.asarray([[42, 42]])],
)
def test_indexing_setitem_scalar(np_array, idx):
    np_array = np_array.copy()
    arr = ndx.asarray(np_array.copy())
    arr[idx] = -1
    np_array[idx] = -1
    np.testing.assert_equal(arr.unwrap_numpy(), np_array)


@pytest.mark.parametrize(
    "np_array, idx",
    [
        (np.array([False, False]), (slice(None, None, -1),)),
        (np.array([False, False]), (slice(None, None, -2),)),
        (
            np.full((0, 2), dtype=bool, fill_value=True),
            (slice(None), slice(None, None, 2)),
        ),
    ],
)
def test_indexing_slicing(np_array, idx):
    np_array = np_array.copy()
    arr = ndx.asarray(np_array)

    np.testing.assert_equal(arr[idx].unwrap_numpy(), np_array[idx])
    arr[idx] = -1
    np_array[idx] = -1
    np.testing.assert_equal(arr.unwrap_numpy(), np_array)


def test_indexing_assign_to_zero_dim():
    np_array = np.array([])
    arr = ndx.asarray(np_array)

    idx = ...
    np.testing.assert_equal(arr[idx].unwrap_numpy(), np_array[idx])
    arr[idx] = ndx.asarray(np_array)
    np_array[idx] = np_array.copy()
    np.testing.assert_equal(arr.unwrap_numpy(), np_array)


@pytest.mark.parametrize("value", ["foo", np.array("foo"), np.array(["foo"])])
@pytest.mark.parametrize("string_dtype", [ndx.utf8, ndx.nutf8])
def test_string_arrays(value, string_dtype):
    arr = ndx.asarray(value).astype(string_dtype)

    assert "foobar" == (arr + "bar").unwrap_numpy()
    assert "barfoo" == ("bar" + arr).unwrap_numpy()

    arr2 = ndx.reshape(arr, (1,))
    if string_dtype == ndx.utf8:
        assert arr2[0] == arr
    else:
        assert (arr2._tyarray[0] == arr._tyarray).unwrap_numpy()


def test_repr_eager():
    assert "array(data: [1], shape=(1,), dtype=Int64)" == str(
        ndx.asarray(np.array([1]))
    )
    assert "array(data: [1], mask: None, shape=(1,), dtype=NInt64)" == str(
        ndx.asarray(np.ma.array([1]))
    )
    assert "array(data: [1], mask: [True], shape=(1,), dtype=NInt64)" == str(
        ndx.asarray(np.ma.array([1], mask=[True]))
    )


def test_repr_lazy():
    # TODO: Show dynamic shape parameters rather than `None`
    assert "array(data: *lazy*, shape=(None,), dtype=Int64)" == str(
        ndx.Array(shape=("N",), dtype=ndx.int64)
    )
    assert "array(data: *lazy*, mask: *lazy*, shape=(None,), dtype=NInt64)" == str(
        ndx.Array(shape=("N",), dtype=ndx.nint64)
    )


def test_schema_v1():
    a = ndx.array(shape=("N",), dtype=ndx.int64)
    b = ndx.array(shape=("N",), dtype=ndx.nint64)
    mp = ndx.build({"a": a, "b": b}, {"c": a + b})

    meta = json.loads({el.key: el.value for el in mp.metadata_props}["ndonnx_schema"])

    # Schema as used prior to the rewrite
    expected = {
        "input_schema": {
            "a": {"author": "ndonnx", "meta": None, "type_name": "Int64"},
            "b": {"author": "ndonnx", "meta": None, "type_name": "NInt64"},
        },
        "output_schema": {
            "c": {"author": "ndonnx", "meta": None, "type_name": "NInt64"}
        },
        "version": 1,
    }
    assert meta == expected


@pytest.mark.parametrize("np_arr2", [np.array([2]), np.array([-2])])
def test_remainder(np_arr2):
    np_arr1 = np.array([-3, -1, 2, 3])

    candidate = ndx.asarray(np_arr1) % ndx.asarray(np_arr2)
    np.testing.assert_equal(np_arr1 % np_arr2, candidate.unwrap_numpy())
