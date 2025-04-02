# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
import json
import operator

import numpy as np
import pytest
from packaging.version import parse

import ndonnx as ndx

from .utils import assert_equal_dtype_shape, build_and_run, run


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
    arr = ndx.argument(shape=shape, dtype=dtype)

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
    res = ndx.argument(shape=shape, dtype=dtype1) + ndx.argument(
        shape=shape, dtype=dtype2
    )

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
    res = ndx.argument(shape=shape, dtype=dtype1) | ndx.argument(
        shape=shape, dtype=dtype2
    )

    assert_equal_dtype_shape(res, res_dtype, shape)


def test_value_prop():
    arr = ndx.asarray(1)
    np.testing.assert_allclose((arr + arr).unwrap_numpy(), np.array(2))

    with pytest.raises(ValueError, match="no propagated value available"):
        ndx.argument(shape=("N",), dtype=ndx.int32).unwrap_numpy()


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
    arr = ndx.argument(shape=("N", "M"), dtype=ndx.nint32)
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
    assert "array(data: [1], dtype=int64)" == str(ndx.asarray(np.array([1], np.int64)))
    assert "array(data: [1], mask: None, dtype=nint64)" == str(
        ndx.asarray(np.ma.array([1], np.int64))
    )
    assert "array(data: [1], mask: [True], dtype=nint64)" == str(
        ndx.asarray(np.ma.array([1], mask=[True], dtype=np.int64))
    )


def test_repr_lazy():
    assert "array(data: *lazy*, shape=('N',), dtype=int64)" == str(
        ndx.argument(shape=("N",), dtype=ndx.int64)
    )
    assert "array(data: *lazy*, mask: *lazy*, shape=('N',), dtype=nint64)" == str(
        ndx.argument(shape=("N",), dtype=ndx.nint64)
    )


def test_schema_v1():
    a = ndx.argument(shape=("N",), dtype=ndx.int64)
    b = ndx.argument(shape=("N",), dtype=ndx.nint64)
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
        # Overflow is undefined behavior (C++ and Rust seem to cycle
        # at least on some architectures)
        # (np.array(1, np.int32), np.array(32, np.int16)),
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


_NP_INTEGER_DTYPES = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]


@pytest.mark.parametrize("np_dtype", _NP_INTEGER_DTYPES)
def test_isin_with_type_promotion(np_dtype):
    np_arr = np.asarray([1, 2, 3], dtype=np_dtype)
    test_elements = [1, 2.2, 3.8]

    np_res = np.isin(np_arr, test_elements)

    res = ndx.extensions.isin(ndx.asarray(np_arr), test_elements)

    np.testing.assert_array_equal(res.unwrap_numpy(), np_res, strict=True)


@pytest.mark.parametrize(
    "np_dtype",
    [np.int64, np.uint16, np.dtype("datetime64[s]"), np.dtype("timedelta64[s]")],
)
def test_put(np_dtype):
    np_idx = np.array([0, 2], np.int64)
    np_arr = np.array([1, 2, 3], dtype=np_dtype)
    arr = ndx.asarray(np_arr)
    idx = ndx.asarray(np_idx)

    np.put(np_arr, np_idx, np.asarray(5, dtype=np_dtype))
    ndx.extensions.put(arr, idx, ndx.asarray(5, dtype=ndx.from_numpy_dtype(np_dtype)))


@pytest.mark.parametrize(
    "arrays",
    [
        (np.asarray([1, 2, 3]), np.asarray([5])),
        (np.asarray([[1], [2], [3]]), np.asarray([5])),
        (np.asarray([[1], [2], [3]]), np.asarray([5]), -5),
    ],
)
def test_broadcast_shapes(arrays):
    np_result = np.broadcast_arrays(*arrays)
    ndx_arrays = [ndx.asarray(arr) for arr in arrays]
    ndx_result = ndx.broadcast_arrays(*ndx_arrays)
    for np_arr, ndx_arr in zip(np_result, ndx_result):
        np.testing.assert_equal(np_arr, ndx_arr.unwrap_numpy())


def test_array_repr_lazy():
    arr = ndx.argument(shape=("N",), dtype=ndx.DateTime64DType("s"))
    res = repr(arr)
    assert res == "array(data: *lazy*, shape=('N',), dtype=datetime64[s])"


@pytest.mark.parametrize("shape", [(3, 4), (1,), ()])
def test_dynamic_size(shape):
    onnx_shape = tuple(f"foo_{dim_len}" for dim_len in shape)
    a = ndx.argument(shape=onnx_shape, dtype=ndx.int64)
    model = ndx.build({"a": a}, {"size": a.dynamic_size})

    np_arr = np.ones(shape, dtype=np.int64)
    (res,) = run(model, {"a": np_arr}).values()

    np.testing.assert_array_equal(res, np_arr.size)


@pytest.mark.parametrize(
    "shape1, shape2",
    [
        ((1,), (1,)),
        ((1,), (1, 1)),
        ((1, 1), (1,)),
        ((1,), (1,)),
        ((1,), (1, 1, 1)),
        ((1, 1, 1), (1,)),
    ],
)
@pytest.mark.parametrize("dtype", [np.int32])
def test_matmul(shape1, shape2, dtype):
    def do(npx):
        a1 = npx.asarray(np.arange(0, np.prod(shape1), 1, dtype=dtype).reshape(shape1))
        a2 = npx.asarray(np.arange(0, np.prod(shape2), 1, dtype=dtype).reshape(shape2))

        return a1 @ a2

    np.testing.assert_array_equal(do(np), do(ndx).unwrap_numpy(), strict=True)


@pytest.mark.parametrize("method_name", ["sum"])
@pytest.mark.parametrize("axis", [None, 0, 1, (0,), (1,), (0, 1)])
@pytest.mark.parametrize("keepdims", [True, False])
def test_non_standard_array_reduction_methods(method_name, axis, keepdims):
    def do(npx):
        arr = npx.reshape(npx.arange(1, 10, dtype=npx.int64), (3, 3))
        return getattr(arr, method_name)(axis=axis, keepdims=keepdims)

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np), strict=True)


@pytest.mark.parametrize("constant", [ndx.pi, ndx.e, ndx.nan])
def test_constants_are_python_floats(constant):
    # TODO: Upstream this test
    assert isinstance(constant, float)


def test_newaxis_is_alias_none():
    # TODO: Upstream this test
    # TODO: Clarify the specs. They currently say that constants need
    # to be floats but also that `newaxis` is an alias for `None`.

    assert ndx.newaxis is None
