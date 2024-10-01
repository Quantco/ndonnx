# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
import operator

import numpy as np
import pytest

import ndonnx._logic_in_data as ndx
from ndonnx._logic_in_data import _dtypes as dtypes
from ndonnx._logic_in_data._schema import get_schemas


def check_dtype_shape(arr, dtype, shape):
    assert arr.dtype == dtype
    assert arr._data.shape == shape
    assert arr.shape == tuple(None if isinstance(el, str) else el for el in shape)


def build_and_run(fn, *np_args):
    import onnxruntime as ort

    ins_np = {f"in{i}": arr for i, arr in enumerate(np_args)}
    ins = {
        k: ndx.Array(shape=a.shape, dtype=dtypes.from_numpy(a.dtype))
        for k, a in ins_np.items()
    }

    out = {"out": fn(*ins.values())}
    mp = ndx.build(ins, out)
    session = ort.InferenceSession(mp.SerializeToString())
    (out,) = session.run(None, {f"{k}__var": a for k, a in ins_np.items()})
    return out


def constant_prop(fn, *np_args):
    return fn(*[ndx.asarray(a) for a in np_args]).unwrap_numpy()


@pytest.fixture(autouse=True)
def error_when_prop_fails():
    from spox import _future

    _future.set_type_warning_level(_future.TypeWarningLevel.OUTPUTS)
    yield


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
def test_add_pyscalar_coretypes(scalar, dtype, res_dtype, op):
    shape = ("N",)
    arr = ndx.Array(shape=shape, dtype=dtype)

    check_dtype_shape(op(scalar, arr), res_dtype, shape)
    check_dtype_shape(op(arr, scalar), res_dtype, shape)


@pytest.mark.parametrize(
    "dtype1, dtype2, res_dtype",
    [
        (ndx.int32, ndx.int32, ndx.int32),
        (ndx.int64, ndx.int32, ndx.int64),
        (ndx.float64, ndx.int32, ndx.float64),
        (ndx.nint32, ndx.int32, ndx.nint32),
    ],
)
def test_core_add(dtype1, dtype2, res_dtype):
    shape = ("N",)
    res = ndx.Array(shape=shape, dtype=dtype1) + ndx.Array(shape=shape, dtype=dtype2)

    check_dtype_shape(res, res_dtype, shape)


@pytest.mark.parametrize(
    "dtype1, dtype2, res_dtype",
    [
        (ndx.int32, ndx.int32, ndx.int32),
        (ndx.int64, ndx.int32, ndx.int64),
        (ndx.bool, ndx.bool, ndx.bool),
    ],
)
def test_core_or(dtype1, dtype2, res_dtype):
    shape = ("N",)
    res = ndx.Array(shape=shape, dtype=dtype1) | ndx.Array(shape=shape, dtype=dtype2)

    check_dtype_shape(res, res_dtype, shape)


def test_value_prop():
    arr = ndx.Array(value=1)
    np.testing.assert_allclose((arr + arr).unwrap_numpy(), np.array(2))

    with pytest.raises(ValueError, match="no propagated value available"):
        ndx.Array(shape=("N",), dtype=ndx.int32).unwrap_numpy()


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
    cond = ndx.Array(shape=shape, dtype=ndx.bool)
    x = ndx.Array(shape=shape, dtype=x_ty)
    y = ndx.Array(shape=shape, dtype=y_ty)

    res = ndx.where(cond, x, y)

    check_dtype_shape(res, res_ty, shape)


def test_datetime():
    arr = ndx.Array(shape=("N",), dtype=ndx.DateTime("s"))
    one_s_td = (arr + 1) - arr
    assert one_s_td.dtype == ndx.TimeDelta("s")

    ten_s_td = one_s_td * 10

    res = arr + ten_s_td
    assert res.dtype == ndx.DateTime("s")


def test_datetime_value_prop():
    arr = ndx.asarray(np.asarray([1, 2])).astype(ndx.DateTime("s"))
    np.testing.assert_equal(
        arr.unwrap_numpy(), np.asarray([1, 2], dtype="datetime64[s]")
    )


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

    check_dtype_shape(op(scalar, arr), res_dtype, shape)
    check_dtype_shape(op(arr, scalar), res_dtype, shape)


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
    check_dtype_shape(op(scalar, arr), expected_dtype, shape)
    check_dtype_shape(op(arr, scalar), expected_dtype, shape)


def test_build():
    a = ndx.Array(shape=("N",), dtype=ndx.nint64)
    b = a[0]

    mp = ndx.build({"a": a}, {"b": b})

    schemas = get_schemas(mp)

    # test json round trip of schema data
    assert schemas.arguments["a"] == a._data.disassemble()[1]
    assert schemas.results["b"] == b._data.disassemble()[1]


@pytest.mark.parametrize(
    "dtype, expect_ort_success",
    [
        (np.float16, True),
        (np.float32, True),
        (np.float64, True),
        (np.int8, False),
        (np.int16, False),
        (np.int32, True),
        (np.int64, True),
        (np.uint8, False),
        (np.uint16, False),
        (np.uint32, False),
        (np.uint64, False),
    ],
)
@pytest.mark.parametrize("values", [[], 1, [1], [1, 2], [[1], [2]]])
def test_function(dtype, values, expect_ort_success):
    np_arr = np.asarray([1, 2], dtype=dtype)
    fun = operator.add

    expected = fun(np_arr, np_arr)

    if expect_ort_success:
        candidate = build_and_run(fun, np_arr, np_arr)
        np.testing.assert_equal(expected, candidate)
    else:
        import onnxruntime as ort

        with pytest.raises(ort.capi.onnxruntime_pybind11_state.NotImplemented):
            build_and_run(fun, np_arr, np_arr)

    candidate = constant_prop(fun, np_arr, np_arr)
    np.testing.assert_equal(expected, candidate)


@pytest.mark.parametrize("dtype", [None, ndx.int32, ndx.float64])
def test_ones(dtype):
    from ndonnx._logic_in_data import ones
    from ndonnx._logic_in_data._dtypes import as_numpy

    shape = (2,)
    candidate = ones(shape, dtype=dtype)
    assert candidate.dtype == dtype or ndx.float64

    if dtype is None:
        dtype = ndx._default_float
    np.testing.assert_equal(
        candidate.unwrap_numpy(), np.ones(shape, dtype=as_numpy(dtype))
    )


def test_indexing_shape():
    arr = ndx.Array(shape=("N", "M"), dtype=ndx.nint32)
    assert arr[0, :]._data.shape == ("M",)
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
def test_more_slicing(np_array, idx):
    np_array = np_array.copy()
    arr = ndx.asarray(np_array)

    np.testing.assert_equal(arr[idx].unwrap_numpy(), np_array[idx])
    arr[idx] = -1
    np_array[idx] = -1
    np.testing.assert_equal(arr.unwrap_numpy(), np_array)


def test_assign_to_zero_dim():
    np_array = np.array([])
    arr = ndx.asarray(np_array)

    idx = ...
    np.testing.assert_equal(arr[idx].unwrap_numpy(), np_array[idx])
    arr[idx] = ndx.asarray(np_array)
    np_array[idx] = np_array.copy()
    np.testing.assert_equal(arr.unwrap_numpy(), np_array)


@pytest.mark.parametrize(
    "np_dtype, ndx_dtype",
    [
        (np.dtype("int32"), ndx.int32),
        (np.dtype("datetime64[s]"), ndx.DateTime("s")),
    ],
)
@pytest.mark.parametrize(
    "np_array1, np_array2",
    [
        (np.array([1, 2]), np.array([3])),
    ],
)
def test_min_max(ndx_dtype, np_dtype, np_array1, np_array2):
    arr1 = ndx.asarray(np_array1).astype(ndx_dtype)
    arr2 = ndx.asarray(np_array2).astype(ndx_dtype)

    candidate = ndx.maximum(arr1, arr2).unwrap_numpy()
    expectation = np.maximum(np_array1.astype(np_dtype), np_array2.astype(np_dtype))

    np.testing.assert_array_equal(candidate, expectation)


@pytest.mark.parametrize("value", ["foo", np.array("foo"), np.array(["foo"])])
@pytest.mark.parametrize("string_dtype", [ndx.string, ndx.nstring])
def test_string_arrays(value, string_dtype):
    arr = ndx.asarray(value).astype(string_dtype)

    assert "foobar" == (arr + "bar").unwrap_numpy()
    assert "barfoo" == ("bar" + arr).unwrap_numpy()

    arr2 = ndx.reshape(arr, (1,))
    if string_dtype == ndx.string:
        assert arr2[0] == arr
    else:
        # TODO: Properly implement eq for masked arrays
        assert arr2._data[0].data == arr._data.data  # type: ignore


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


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize(
    "np_arrays",
    [
        [np.asarray([[1], [2]]), np.asarray([[3.0], [4.0]])],
        [np.ma.array([[1], [2]]), np.ma.array([[3.0], [4.0]])],
        [np.ma.array([[1], [2]]), np.ma.array([[3.0], [4.0]], mask=[[True], [False]])],
    ],
)
def test_concat(np_arrays, axis):
    arrays = [ndx.asarray(arr) for arr in np_arrays]
    expected = np.concat(np_arrays, axis=axis)
    candidate = ndx.concat(arrays, axis=axis).unwrap_numpy()

    np.testing.assert_equal(expected, candidate)


def test_reshape_with_array():
    expected_shape = (2, 1)
    new_shape = ndx.asarray(np.array(expected_shape))
    candidate_shape = ndx.reshape(ndx.asarray(np.array([[1, 2]])), new_shape).shape
    assert expected_shape == candidate_shape