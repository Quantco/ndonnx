# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
import json
import operator

import numpy as np
import pytest
from packaging.version import parse

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
    np.testing.assert_array_equal(candidate, expected)

    candidate = constant_prop(fun, np_arr, np_arr)
    np.testing.assert_array_equal(expected, candidate)


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
    np.testing.assert_array_equal(arr[idx].unwrap_numpy(), np_array[idx])


@pytest.mark.parametrize("np_array", [np.asarray([[1, 2]]), np.asarray([1, 2])])
def test_indexing_value_prop_scalar_slice(np_array):
    arr = ndx.asarray(np_array)
    idx = (slice(None, 1), ...)
    assert arr[idx].shape == np_array[idx].shape
    assert arr[idx].dtype == arr.dtype
    np.testing.assert_array_equal(arr[idx].unwrap_numpy(), np_array[idx])


def test_indexing_value_prop_tuple_index():
    np_arr = np.asarray([[1, 2]])
    arr = ndx.asarray(np_arr)
    for idx in np.ndindex(arr.shape):  # type: ignore
        el = arr[idx]
        assert el.shape == ()
        assert el.dtype == arr.dtype
        np.testing.assert_array_equal(el.unwrap_numpy(), np_arr[idx])


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
    np.testing.assert_array_equal(arr.unwrap_numpy(), np_array)


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

    np.testing.assert_array_equal(arr[idx].unwrap_numpy(), np_array[idx])
    arr[idx] = -1
    np_array[idx] = -1
    np.testing.assert_array_equal(arr.unwrap_numpy(), np_array)


def test_indexing_assign_to_zero_dim():
    np_array = np.array([])
    arr = ndx.asarray(np_array)

    idx = ...
    np.testing.assert_array_equal(arr[idx].unwrap_numpy(), np_array[idx])
    arr[idx] = ndx.asarray(np_array)
    np_array[idx] = np_array.copy()
    np.testing.assert_array_equal(arr.unwrap_numpy(), np_array)


@pytest.mark.parametrize("idx_list", ([[True, False], [False, True]], [True, False]))
def test_indexing_boolean_array(idx_list):
    np_arr = np.ones((2, 2), np.float64)
    arr = ndx.asarray(np_arr)

    np_idx = np.array(idx_list)
    idx = ndx.asarray(np_idx)

    update = 42.0
    np_arr[np_idx] = update
    arr[idx] = update
    np.testing.assert_array_equal(arr.unwrap_numpy(), np_arr)


@pytest.mark.skip(reason="Unclear index type")
def test_indexing_boolean_array_equivalent_nonzero():
    idx_list = [True, False]
    np_arr = np.ones((2, 2), np.float64)
    arr1 = ndx.asarray(np_arr)
    arr2 = ndx.asarray(np_arr)

    np_idx = np.array(idx_list)
    idx = ndx.asarray(np_idx)

    arr1[idx] = 42.0
    arr2[ndx.nonzero(idx)] = 42.0

    np_arr[np.nonzero(np_idx)] = 42

    np.testing.assert_array_equal(arr1.unwrap_numpy(), np_arr)


@pytest.mark.skip(reason="Indexing with integer arrays is not defined by the standard.")
def test_indexing_set_with_int_array():
    np_a = np.array([1, 2, 3])
    np_index = np.array([0, 2])

    a = ndx.asarray(np_a)
    index = ndx.asarray(np_index)

    c = a.copy()
    c[index] = 0

    expected_c = np_a
    expected_c[np_index] = 0

    np.testing.assert_array_equal(expected_c, c.unwrap_numpy(), strict=True)


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
        ndx.asarray(np.array([1], np.int64))
    )
    assert "array(data: [1], mask: None, shape=(1,), dtype=NInt64)" == str(
        ndx.asarray(np.ma.array([1], np.int64))
    )
    assert "array(data: [1], mask: [True], shape=(1,), dtype=NInt64)" == str(
        ndx.asarray(np.ma.array([1], mask=[True], dtype=np.int64))
    )


def test_repr_lazy():
    assert "array(data: *lazy*, shape=('N',), dtype=Int64)" == str(
        ndx.Array(shape=("N",), dtype=ndx.int64)
    )
    assert "array(data: *lazy*, mask: *lazy*, shape=('N',), dtype=NInt64)" == str(
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
    np.testing.assert_array_equal(
        candidate.unwrap_numpy(), np_arr1 % np_arr2, strict=True
    )


def test_dynamic_shape_propagates_staticlly_known_shape():
    shape = (
        2,
        2,
    )
    assert tuple(ndx.ones(shape).dynamic_shape.unwrap_numpy()) == shape
    assert tuple(ndx.ones(shape)._tyarray.dynamic_shape.unwrap_numpy()) == shape


@pytest.mark.parametrize(
    "np_left, right",
    [
        (np.array([-1, -128, 10], np.int8), 2),
        (np.array([-1, -128, 10], np.int8), np.array(2, np.uint16)),
        (np.array([-1, -128, 10], np.int8), np.array(0, np.uint16)),
        (2, np.array([0, 1, 2], np.int8)),
        (np.array(2, np.uint16), np.array([0, 1, 2], np.int8)),
        (np.array(0, np.uint16), np.array([0, 1, 2], np.int8)),
        (np.array(3, np.uint16), np.array(1, np.uint8)),
    ],
)
def test_bitshift_right(np_left, right):
    def to_ndx_if_array(obj: np.ndarray | int) -> ndx.Array | int:
        if isinstance(obj, np.ndarray):
            return ndx.asarray(obj)
        return obj

    ndx_left = to_ndx_if_array(np_left)
    ndx_right = to_ndx_if_array(right)

    np_res = np_left >> right
    ndx_res = ndx_left >> ndx_right

    np.testing.assert_array_equal(
        ndx_res.unwrap_numpy(), np_res, strict=parse(np.__version__).major >= 2
    )  # type: ignore


@pytest.mark.parametrize(
    "left, right",
    [
        (np.array([-1, -128, 10], np.int8), 2),
        (np.array([-1, -128, 10], np.int8), np.array(2, np.uint16)),
        (np.array([-1, -128, 10], np.int8), np.array(0, np.uint16)),
        (2, np.array([0, 1, 2], np.int8)),
        (np.array(2, np.uint16), np.array([0, 1, 2], np.int8)),
        (np.array(0, np.uint16), np.array([0, 1, 2], np.int8)),
        (np.array(3, np.uint16), np.array(1, np.uint8)),
    ],
)
@pytest.mark.skipif(
    parse(np.__version__).major < 2, reason="NumPy 1.x has different casting rules."
)
def test_bitshift_left(left, right):
    def to_ndx_if_array(obj: np.ndarray | int) -> ndx.Array | int:
        if isinstance(obj, np.ndarray):
            return ndx.asarray(obj)
        return obj

    ndx_left = to_ndx_if_array(left)
    ndx_right = to_ndx_if_array(right)

    np_res = left << right
    ndx_res = ndx_left << ndx_right

    np.testing.assert_array_equal(ndx_res.unwrap_numpy(), np_res, strict=True)  # type: ignore


def test_masked_clip():
    x = ndx.asarray(np.ma.MaskedArray([1, 2, 3, 4], [True, False, True, False]))
    res = ndx.clip(x, 0, 3)
    np.testing.assert_array_equal(
        res.unwrap_numpy(), np.ma.MaskedArray([1, 2, 3, 3], [True, False, True, False])
    )


@pytest.mark.xfail(reason="Not implemented, yet")
def test_masked_where():
    cond = ndx.asarray(
        np.ma.MaskedArray([True, True, False, False], [True, False, True, False])
    )
    x = ndx.asarray([1, 2, 3, 4])
    y = ndx.asarray([10, 20, 30, 40])
    res = ndx.where(cond, x, y)
    np.testing.assert_array_equal(
        res.unwrap_numpy(), np.ma.MaskedArray([0, 0, 0, 0], [True, False, True, False])
    )


@pytest.mark.parametrize("scalar", [True, False, 0, 1, 0.0, -0.0, np.float32(1)])
def test_asarray_matches_numpy(scalar):
    is_np2 = np.__version__.startswith("2")
    np.testing.assert_array_equal(
        np.asarray(scalar), ndx.asarray(scalar).unwrap_numpy(), strict=is_np2
    )
