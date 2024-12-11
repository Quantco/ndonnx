# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re

import numpy as np
import pytest
import spox.opset.ai.onnx.v19 as op

import ndonnx._refactor as ndx
from ndonnx._refactor import _dtypes as dtypes
from ndonnx._utility import promote

from .utils import assert_array_equal, get_numpy_array_api_namespace, run


def numpy_to_graph_input(arr, eager=False):
    from ndonnx._refactor._typed_array.masked_onnx import as_nullable

    dtypes.from_numpy

    dtype: dtypes.DType
    if isinstance(arr, np.ma.MaskedArray):
        dtype = as_nullable(dtypes.from_numpy(arr.dtype))
    else:
        dtype = dtypes.from_numpy(arr.dtype)
    return (
        ndx.array(
            shape=arr.shape,
            dtype=dtype,
        )
        if not eager
        else ndx.asarray(
            arr,
            dtype=dtype,
        )
    )


@pytest.fixture
def _a():
    return np.array([1, 2, 3])


@pytest.fixture
def _b():
    return np.array([2, 3, 3])


@pytest.fixture
def _c():
    return np.array([[1, 2, 3], [4, 5, 6]])


@pytest.mark.parametrize(
    "dtype, input",
    [
        (ndx.bool, np.array([True, False, True])),
        (ndx.int8, np.array([1, 5, -12], np.int8)),
        (ndx.nbool, np.ma.masked_array([True, True, False], mask=[1, 0, 0])),
        (ndx.nint8, np.ma.masked_array([1, 5, -12], mask=[0, 1, 1], dtype=np.int8)),
    ],
)
@pytest.mark.parametrize("shape", [(3,), ("N",)])
def test_make_graph_input(dtype, input, shape):
    a = ndx.array(shape=shape, dtype=dtype)
    model = ndx.build({"a": a}, {"b": a})
    actual = run(model, {"a": input})
    assert_array_equal(actual["b"], input)


def test_null_promotion():
    a = ndx.array(shape=("N",), dtype=ndx.nfloat64)
    b = ndx.array(shape=("N",), dtype=ndx.float64)
    model = ndx.build({"a": a, "b": b}, {"c": a + b})
    inputs = {
        "a": np.ma.masked_array([1.0, 2.0, 3.0, 4.0], mask=[0, 0, 1, 0]),
        "b": np.array([4.0, 5.0, 6.0, 7.0]),
    }
    actual = run(model, inputs)
    assert_array_equal(actual["c"], np.add(inputs["a"], inputs["b"]))


@pytest.mark.parametrize(
    "array, dtype, expected_dtype",
    [
        ([1, 2, 3], ndx.int64, ndx.int64),
        (np.array([1, 2, 3], np.int64), None, ndx.int64),
        (1, ndx.int64, ndx.int64),
        (1, ndx.float64, ndx.float64),
        (["a", "b"], ndx.utf8, ndx.utf8),
        (np.array(["a", "b"]), None, ndx.utf8),
        (np.array(["a", "b"], object), None, ndx.utf8),
        ("a", ndx.utf8, ndx.utf8),
        (np.array("a"), None, ndx.utf8),
        (np.array("a", object), None, ndx.utf8),
        ([["a"]], None, ndx.utf8),
        (np.array([["a"]]), None, ndx.utf8),
        (np.array([["a"]], object), None, ndx.utf8),
    ],
)
def test_asarray(array, dtype, expected_dtype):
    a = ndx.asarray(array, dtype=dtype)
    assert a.dtype == expected_dtype
    assert_array_equal(np.array(array, expected_dtype.to_numpy_dtype()), a.to_numpy())


@pytest.mark.parametrize(
    "np_arr",
    [
        np.ma.masked_array([1, 2, 3], mask=[0, 0, 1]),
        np.ma.masked_array(1, mask=0),
        np.ma.masked_array(["a", "b"], mask=[1, 0]),
        np.ma.masked_array("a", mask=0),
    ],
)
def test_asarray_masked(np_arr):
    ndx_arr = ndx.asarray(np_arr)
    assert isinstance(ndx_arr, ndx.Array)
    assert isinstance(ndx_arr.to_numpy(), np.ma.MaskedArray)
    assert_array_equal(np_arr, ndx_arr.to_numpy())


@pytest.mark.parametrize(
    "np_arr",
    [
        np.ma.masked_array(["a", "b"], mask=[1, 0], dtype=object),
        np.ma.masked_array("a", mask=0, dtype=object),
    ],
)
def test_asarray_object_dtype(np_arr):
    ndx_arr = ndx.asarray(np_arr)
    assert isinstance(ndx_arr, ndx.Array)
    assert isinstance(ndx_arr.to_numpy(), np.ma.MaskedArray)
    np.testing.assert_array_equal(np_arr, ndx_arr.to_numpy())


def test_basic_eager_add(_a, _b):
    a = ndx.asarray(_a)
    b = ndx.asarray(_b)
    x = ndx.asarray([1]) + 2
    c = a + b + x
    assert_array_equal(c.to_numpy(), _a + _b + np.array([1]) + 2)


def test_string_concatenation():
    a = ndx.asarray(["a", "b", "c"])
    b = ndx.asarray(["d", "e", "f"])
    assert_array_equal((a + b).to_numpy(), np.array(["ad", "be", "cf"]))
    assert_array_equal((a + "d").to_numpy(), np.array(["ad", "bd", "cd"]))


def test_combining_lazy_eager():
    a = ndx.array(shape=(3,), dtype=ndx.int64)
    b = ndx.asarray(np.array([1, 2, 3], dtype=np.int64))
    c = a + b
    assert not c.to_numpy() is not None


def test_basic_indexing():
    a = ndx.asarray([1, 2, 3])
    b = a[0]
    a[0] = 2
    assert (a[0] != b).to_numpy().item()
    assert (a[0] == 2).to_numpy().item()


def test_lazy_array(_a, _b):
    a = numpy_to_graph_input(_a)
    b = ndx.asarray(_b)
    c = a * b

    model = ndx.build({"a": a}, {"c": c})
    actual = run(model, {"a": _a})["c"]
    expected = _a * _b
    assert_array_equal(actual, expected)


def test_indexing(_a):
    a = numpy_to_graph_input(_a)
    b = a[0]

    model = ndx.build({"a": a}, {"b": b})
    actual = run(model, {"a": _a})["b"]
    expected = _a[0]
    assert_array_equal(expected, actual)


@pytest.mark.parametrize(
    "condition, x, y",
    [
        (
            np.array([True, False, False], dtype=np.bool_),
            np.array([1, 2, 3]),
            np.array([3.12, 3.24, -124.0]),
        ),
        (
            np.array([True, False, True], dtype=np.bool_),
            np.ma.masked_array([1, 2, 3], mask=[0, 0, 1]),
            np.array([3.12, 3.24, -124.0]),
        ),
        (
            np.array([True, False, True], dtype=np.bool_),
            np.ma.masked_array([1, 2, 3], mask=[0, 0, 1]),
            np.ma.masked_array([3.12, 3.24, -124.0], mask=[0, 0, 0]),
        ),
    ],
)
def test_where(condition, x, y):
    expected = np.where(condition, x, y)
    condition_onnx = numpy_to_graph_input(condition)
    x_onnx = numpy_to_graph_input(x)
    y_onnx = numpy_to_graph_input(y)
    out = ndx.where(condition_onnx, x_onnx, y_onnx)
    model = ndx.build(
        {"condition": condition_onnx, "x": x_onnx, "y": y_onnx}, {"out": out}
    )
    actual = run(model, {"condition": condition, "x": x, "y": y})["out"]
    # NumPy simply drops the masked array in `np.where`. We do not want to do the same.
    np.testing.assert_array_equal(actual, expected)


def test_indexing_on_scalar():
    res = ndx.asarray(1)
    res = res[()]
    res[()] = 2
    assert res == 2


def test_indexing_on_scalar_mask():
    res = ndx.asarray([])
    res = res[False]
    assert_array_equal(res.unwrap_numpy().shape, np.asarray([0, 0]))


def test_indexing_with_mask(_a):
    _mask = np.array([True, False, True])

    a = numpy_to_graph_input(_a)
    mask = numpy_to_graph_input(_mask)
    c = ndx.reshape(a[mask], [-1])

    expected_c = _a[_mask]

    model = ndx.build({"a": a, "mask_": mask}, {"c": c})
    actual = run(model, {"a": _a, "mask_": _mask})["c"]
    assert_array_equal(expected_c, actual)


def test_indexing_with_mask_raises(_a):
    a = numpy_to_graph_input(_a)
    mask = ndx.array(shape=(3, 1, 1), dtype=ndx.bool)

    with pytest.raises(IndexError):
        a[mask]


@pytest.mark.skip
def test_indexing_with_array(_a):
    _index = np.array([0, 2])

    a = numpy_to_graph_input(_a)
    index = numpy_to_graph_input(_index)
    c = ndx.reshape(a[index], [-1])

    expected_c = _a[_index]

    model = ndx.build({"a": a, "index_": index}, {"c": c})
    actual = run(model, {"a": _a, "index_": _index})["c"]
    assert_array_equal(expected_c, actual)


def test_indexing_with_tuple_of_array(_a):
    a = numpy_to_graph_input(_a)
    index = numpy_to_graph_input(np.array([0, 2]))

    with pytest.raises(
        IndexError,
        match=re.escape("Index array must be a scalar but has rank"),
    ):
        a[(index,)]


def test_slicing(_a):
    a = numpy_to_graph_input(_a)
    b = a[1:2]

    model = ndx.build({"a": a}, {"b": b})
    actual = run(model, {"a": _a})["b"]
    assert_array_equal(actual, _a[1:2])


def test_indexing_list(_a):
    a = numpy_to_graph_input(_a)

    with pytest.raises(IndexError, match="unexpected key"):
        a[[0, 2]]


def test_indexing_slice_ellipsis(_c):
    a = numpy_to_graph_input(_c)
    b = a[..., 1]

    model = ndx.build({"a": a}, {"b": b})
    actual = run(model, {"a": _c})["b"]
    expected = _c[..., 1]
    assert_array_equal(actual, expected)


def test_indexing_none(_c):
    _index = (None, 1, None, 1)

    a = numpy_to_graph_input(_c)
    b = a[_index]

    model = ndx.build({"a": a}, {"b": b})
    actual = run(model, {"a": _c})["b"]
    expected = _c[_index]
    assert_array_equal(actual, expected)


def test_illegal_indexing(_a):
    a = numpy_to_graph_input(_a)

    with pytest.raises(IndexError):
        a["invalid", ...]


def test_indexing_with_invalid_rank(_a):
    a = numpy_to_graph_input(_a)

    with pytest.raises(IndexError):
        a[()]

    b = numpy_to_graph_input(np.array([[1, 2], [3, 4]]))
    with pytest.raises(IndexError):
        b[0, 0, 0]
    with pytest.raises(IndexError):
        b[0,]

    b[0, ...]


@pytest.mark.skip("Indexing with integer arrays is ambiguous.")
def test_indexing_set_with_integer_array(_a):
    _index = np.array([0, 2])

    a = numpy_to_graph_input(_a)
    index = numpy_to_graph_input(_index)

    c = a.copy()
    c[index] = 0

    expected_c = _a
    expected_c[_index] = 0

    model = ndx.build({"a": a, "index_": index}, {"c": c})
    actual = run(model, {"a": _a, "index_": _index})["c"]
    assert_array_equal(actual, expected_c)


def test_indexing_set_scalar():
    _a = np.array(2)

    a = numpy_to_graph_input(_a)

    c = a.copy()
    c = ndx.asarray(0)

    expected_c = 0

    model = ndx.build({"a": a}, {"c": c}, drop_unused=False)
    assert_array_equal(expected_c, run(model, {"a": _a})["c"])


# Check if repr does not raise
def test_repr():
    a = ndx.array(shape=(3,), dtype=ndx.int64)
    repr(a)

    b = ndx.asarray([1, 2, 3])
    repr(b)

    c = ndx.array(shape=(3,), dtype=ndx.nint8)
    repr(c)

    d = ndx.asarray(np.array([1, 2, 3], np.int64))
    repr(d)


@pytest.mark.parametrize(
    "rop", ["__radd__", "__rsub__", "__rmul__", "__rtruediv__", "__rpow__", "__rmod__"]
)
def test_rops(rop):
    a = ndx.asarray([1.0, 2.0, 3.0])
    b = ndx.asarray([1.0, 2.0, 3.0])

    res = getattr(a, rop)(b)
    assert_array_equal(res.to_numpy(), getattr(a.to_numpy(), rop)(b.to_numpy()))


def test_rsub():
    a = ndx.asarray([1, 2, 3])

    res = 1 - a
    a_value = a.to_numpy()
    assert a_value is not None
    assert_array_equal(res.to_numpy(), 1 - a_value)


def test_matrix_transpose():
    a = ndx.array(shape=(3, 2, 3), dtype=ndx.int64)
    b = ndx.matrix_transpose(a)

    model = ndx.build({"a": a}, {"b": b})
    npx = get_numpy_array_api_namespace()
    assert_array_equal(
        run(model, {"a": np.arange(3 * 2 * 3, dtype=np.int64).reshape(3, 2, 3)})["b"],
        npx.matrix_transpose(npx.reshape(npx.arange(3 * 2 * 3), (3, 2, 3))),
    )


def test_matrix_transpose_attribute():
    a = ndx.array(shape=(3, 2, 3), dtype=ndx.int64)
    b = a.mT

    model = ndx.build({"a": a}, {"b": b})
    npx = get_numpy_array_api_namespace()
    expected = npx.reshape(npx.arange(3 * 2 * 3), (3, 2, 3)).mT

    assert_array_equal(
        run(model, {"a": np.arange(3 * 2 * 3, dtype=np.int64).reshape(3, 2, 3)})["b"],
        expected,
    )


def test_transpose_attribute():
    a = ndx.array(shape=(3, 2), dtype=ndx.int64)
    b = a.T

    model = ndx.build({"a": a}, {"b": b})
    assert_array_equal(
        np.reshape(np.arange(3 * 2), (3, 2)).T,
        run(model, {"a": np.arange(3 * 2, dtype=np.int64).reshape(3, 2)})["b"],
    )


def test_array_spox_interoperability():
    a = ndx.array(shape=(3, 2), dtype=ndx.nint64)
    add_var = op.add(a.values.disassemble(), op.const(5, dtype=np.int64))  # type: ignore
    b = ndx.from_spox_var(var=add_var)
    model = ndx.build({"a": a}, {"b": b}, drop_unused=False)
    expected = np.reshape(np.arange(3 * 2), (3, 2)) + 5
    input = np.ma.masked_array(
        np.arange(3 * 2, dtype=np.int64).reshape(3, 2), mask=np.ones((3, 2), dtype=bool)
    )
    actual = run(model, {"a": input})["b"]
    assert_array_equal(actual, expected)


def test_creation_arange():
    a = ndx.arange(0, stop=10)
    assert_array_equal(a.to_numpy(), np.arange(stop=10))

    b = ndx.arange(1, 10)
    assert_array_equal(b.to_numpy(), np.arange(1, 10))

    c = ndx.arange(1, 10, 2)
    assert_array_equal(c.to_numpy(), np.arange(1, 10, 2))

    d = ndx.arange(0.0, None, step=-1)
    assert_array_equal(np.arange(0.0, None, step=-1), d.to_numpy())


def test_creation_full():
    a = ndx.full((2, 3), 5)
    assert_array_equal(a.to_numpy(), np.full((2, 3), 5))

    b = ndx.full((2, 3), 5, dtype=ndx.float32)
    assert_array_equal(b.to_numpy(), np.full((2, 3), 5, dtype=np.float32))
    c = ndx.full((2, 3), "a", dtype=ndx.nutf8)
    assert_array_equal(
        c.to_numpy(), np.ma.masked_array(np.full((2, 3), "a"), mask=False)
    )

    d = ndx.full(2, 5, dtype=ndx.int8)
    assert_array_equal(d.to_numpy(), np.full(2, 5, dtype=np.int8))

    # Check lazy creation
    e = ndx.array(shape=tuple(), dtype=ndx.int64)
    f = ndx.full(e, 10)
    model_proto = ndx.build({"e": e}, {"f": f})
    actual = run(model_proto, {"e": np.array(5, dtype=np.int64)})["f"]
    assert_array_equal(actual, np.array([10] * 5, dtype=np.int64))

    # Note we must know the output shape to export an ONNX artifact.
    g = ndx.array(shape=(2,), dtype=ndx.int64)
    h = ndx.full(g, 10)
    model_proto = ndx.build({"g": g}, {"h": h})
    assert_array_equal(
        run(model_proto, {"g": np.array([2, 3], dtype=np.int64)})["h"],
        np.array([[10, 10, 10], [10, 10, 10]]),
    )


def test_creation_ones_like():
    a = ndx.array(shape=("N",), dtype=ndx.int64)
    b = ndx.ones_like(a)
    model = ndx.build({"a": a}, {"b": b})
    assert_array_equal(
        run(model, {"a": np.array([1, 2, 3], dtype=np.int64)})["b"],
        np.ones(3, dtype=np.int64),
    )


@pytest.mark.parametrize(
    "args, expected",
    [
        ((ndx.int32, ndx.int64), ndx.int64),
        (
            (
                ndx.asarray([1, 2, 3], dtype=ndx.int64),
                ndx.asarray([1, 2, 3], dtype=ndx.int32),
            ),
            ndx.int64,
        ),
        ((ndx.int32, ndx.asarray([1, 2, 3], dtype=ndx.int64)), ndx.int64),
        ((ndx.float32, ndx.float64), ndx.float64),
        ((ndx.float64, ndx.int32), ndx.float64),
    ],
)
def test_result_type(args, expected):
    assert ndx.result_type(*args) == expected


def test_ceil():
    a = ndx.asarray(np.array([1.2, 1.3, -13.13]))
    assert_array_equal(ndx.ceil(a).to_numpy(), [2.0, 2.0, -13.0])


def test_propagates_minimal_dtype():
    a = ndx.asarray([1, 2, 4], dtype=ndx.int8)
    b = a + 1
    assert b.dtype == ndx.int8
    assert_array_equal(b.to_numpy(), np.array([2, 3, 5], dtype=np.int8))


@pytest.mark.parametrize(
    "x",
    [
        ndx.asarray([True, False, False]),
        ndx.asarray([False]),
        ndx.asarray([True]),
        ndx.asarray([True, True]),
        ndx.asarray([], dtype=ndx.bool),
    ],
)
def test_all(x):
    assert_array_equal(ndx.all(x).to_numpy(), np.all(x.to_numpy()))


@pytest.mark.parametrize(
    "side",
    [
        "left",
        "right",
    ],
)
def test_searchsorted(side):
    a_val = [0, 1, 2, 5, 5, 6, 10, 15]
    b_val = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 10, 15, 20, 20]
    c_val = np.searchsorted(a_val, b_val, side)

    a = ndx.asarray(a_val, dtype=ndx.int64)
    b = ndx.asarray(b_val, dtype=ndx.int64)
    c = ndx.searchsorted(a, b, side=side)
    assert_array_equal(c_val, c.to_numpy())


@pytest.mark.skip(reason="TODO: onnxruntime")
@pytest.mark.parametrize(
    "side",
    [
        "left",
        "right",
    ],
)
def test_searchsorted_nans(side):
    a_val = np.array([0, 1, 2, 5, 5, 6, 10, 15, np.nan])
    b_val = np.array([0, 1, 2, np.nan, np.nan])
    c_val = np.searchsorted(a_val, b_val, side)

    a = ndx.array(shape=(len(a_val),), dtype=ndx.float64)
    b = ndx.array(shape=(len(b_val),), dtype=ndx.float64)
    c = ndx.searchsorted(a, b, side=side)

    model = ndx.build({"a": a, "b": b}, {"c": c})

    assert_array_equal(c_val, run(model, dict(a=a_val, b=b_val))["c"])


def test_searchsorted_raises():
    with pytest.raises(TypeError):
        a = ndx.array(shape=(), dtype=ndx.int64)
        b = ndx.array(shape=(), dtype=ndx.float64)

        ndx.searchsorted(a, b)

    with pytest.raises(ValueError):
        a = ndx.array(shape=(3,), dtype=ndx.int64)
        b = ndx.array(shape=(3,), dtype=ndx.int64)

        ndx.searchsorted(a, b, side="middle")  # type: ignore[arg-type]


def test_truediv():
    x = ndx.asarray([1, 2, 3], dtype=ndx.int64)
    y = ndx.asarray([2, 3, 3], dtype=ndx.int64)
    z = x / y
    assert isinstance(z.dtype, ndx.Floating)
    assert_array_equal(z.to_numpy(), np.array([0.5, 2 / 3, 1.0]))


@pytest.mark.parametrize(
    "dtype",
    [
        ndx.float32,
        ndx.nfloat32,
        ndx.int8,
        ndx.int16,
        ndx.int32,
        ndx.int64,
        ndx.nint8,
        ndx.nint16,
        ndx.nint32,
        ndx.nint64,
    ],
)
def test_prod(dtype):
    x = ndx.asarray([2, 2]).astype(dtype)
    y = ndx.prod(x)
    if isinstance(dtype, ndx.Nullable):
        input = np.asarray([2, 2], dtype=ndx.as_non_nullable(dtype).to_numpy_dtype())
        input = np.ma.masked_array(input, mask=False)
    else:
        input = np.asarray([2, 2], dtype=dtype.to_numpy_dtype())
    actual = np.prod(input)

    assert_array_equal(y.to_numpy(), actual)


@pytest.mark.parametrize(
    "dtype",
    [
        ndx.uint8,
        ndx.uint16,
        ndx.uint32,
        ndx.nuint8,
        ndx.nuint16,
        ndx.nuint32,
    ],
)
def test_prod_unsigned(dtype):
    x = ndx.asarray([2, 2]).astype(dtype)
    y = ndx.prod(x)
    if isinstance(dtype, ndx.Nullable):
        input = np.asarray([2, 2], dtype=ndx.as_non_nullable(dtype).to_numpy_dtype())
        input = np.ma.masked_array(input, mask=False)
    else:
        input = np.asarray([2, 2], dtype=dtype.to_numpy_dtype())
    actual = np.prod(input)

    assert_array_equal(y.to_numpy(), actual)


@pytest.mark.parametrize(
    "dtype",
    [
        ndx.uint64,
        ndx.nuint64,
    ],
)
def test_prod_no_implementation(dtype):
    x = ndx.asarray([2, 2]).astype(dtype)
    with pytest.warns():
        ndx.prod(x)


@pytest.mark.skip(reason="`_from_fields` was removed")
def test_array_creation_with_invalid_fields():
    with pytest.raises(TypeError):
        ndx.Array._from_fields(
            ndx.nutf8,
            values=ndx.array(shape=(3,), dtype=ndx.int32),
            invalid_field=ndx.array(shape=(3,), dtype=ndx.utf8),
        )

    with pytest.raises(TypeError):
        ndx.Array._from_fields(
            ndx.nutf8,
            values=ndx.array(shape=(3,), dtype=ndx.int32),
            null=ndx.array(shape=(3,), dtype=ndx.bool),
        )

    with pytest.raises(TypeError):
        ndx.Array._from_fields(
            ndx.utf8,
            values=ndx.array(shape=(3,), dtype=ndx.utf8),
        )

    with pytest.raises(TypeError):
        ndx.Array._from_fields(
            ndx.utf8,
            values=ndx.array(shape=(3,), dtype=ndx.int32)._core(),
        )


@pytest.mark.parametrize(
    "operation", [ndx.sin, ndx.cos, ndx.tan, ndx.sinh, ndx.mean, ndx.sum, ndx.abs]
)
@pytest.mark.parametrize("dtype", [ndx.utf8, ndx.nutf8, ndx.bool, ndx.nbool])
def test_numerical_unary_operations_fail_on_non_numeric_input(operation, dtype):
    a = ndx.array(shape=(3,), dtype=dtype)
    with pytest.raises(TypeError, match="is not implemented for data type"):
        operation(a)


def test_string_shape_operations():
    a = ndx.asarray(["a", "b", "c"])
    b = ndx.broadcast_to(a, (2, 3))
    assert_array_equal(b.to_numpy(), np.array([["a", "b", "c"], ["a", "b", "c"]]))

    c = ndx.concat([a, a], axis=0)
    assert_array_equal(c.to_numpy(), np.array(["a", "b", "c", "a", "b", "c"]))


@pytest.mark.parametrize(
    "dtype",
    [
        ndx.utf8,
        ndx.bool,
    ],
)
def test_zeros(dtype):
    x = ndx.zeros((2, 3), dtype=dtype)
    assert_array_equal(x.to_numpy(), np.zeros((2, 3), dtype=dtype.to_numpy_dtype()))


@pytest.mark.xfail(reason="https://github.com/onnx/onnx/issues/6276")
def test_empty_concat_eager():
    a = ndx.asarray([], ndx.int64)
    b = ndx.asarray([1, 2, 3], ndx.int64)
    out = ndx.concat([a, b], axis=0)
    assert_array_equal(out.to_numpy(), b.to_numpy())


def test_empty_concat_lazy_known_shape():
    a = ndx.array(shape=(0,), dtype=ndx.int64)
    b = ndx.array(shape=(3,), dtype=ndx.int64)
    out = ndx.concat([a, b])
    model = ndx.build({"a": a, "b": b}, {"out": out})

    anp = np.array([], np.int64)
    bnp = np.array([1, 2, 3], np.int64)

    out = run(model, {"a": anp, "b": bnp})["out"]

    assert_array_equal(out, bnp)


def test_empty_concat_lazy_unknown_shape():
    a = ndx.array(shape=(None,), dtype=ndx.int64)
    b = ndx.array(shape=(None,), dtype=ndx.int64)
    out = ndx.concat([a, b])
    model = ndx.build({"a": a, "b": b}, {"out": out})

    anp = np.array([], np.int64)
    bnp = np.array([1, 2, 3], np.int64)

    out = run(model, {"a": anp, "b": bnp})["out"]

    assert_array_equal(out, bnp)


# if the precision loss looks concerning, note https://data-apis.org/array-api/latest/API_specification/type_promotion.html#mixing-arrays-with-python-scalars
@pytest.mark.parametrize(
    "array, scalar, expected",
    [
        (ndx.array(shape=("N",), dtype=ndx.uint8), 1, ndx.uint8),
        (ndx.array(shape=("N",), dtype=ndx.uint8), -1, ndx.uint8),
        (ndx.array(shape=("N",), dtype=ndx.int8), 1, ndx.int8),
        (ndx.array(shape=("N",), dtype=ndx.nint8), 1, ndx.nint8),
        (ndx.array(shape=("N",), dtype=ndx.nuint8), -1, ndx.nuint8),
        (ndx.array(shape=("N",), dtype=ndx.float64), 0.123456789, ndx.float64),
        (
            ndx.array(shape=("N",), dtype=ndx.float64),
            np.float64(0.123456789),
            ndx.float64,
        ),
        (ndx.array(shape=("N",), dtype=ndx.float32), 0.123456789, ndx.float32),
        # (
        #     [
        #         ndx.array(shape=("N",), dtype=ndx.float32),
        #         ndx.asarray([1.5], dtype=ndx.float64),
        #     ],
        #     0.123456789,
        # ),
        (ndx.asarray(["a", "b"], dtype=ndx.utf8), "hello", ndx.utf8),
    ],
)
def test_scalar_promote(array, scalar, expected):
    res1 = array + scalar
    res2 = scalar + array

    assert res1.dtype == res2.dtype == expected


@pytest.mark.parametrize(
    "arrays, scalar",
    [
        ([ndx.asarray(["a", "b"], dtype=ndx.utf8)], 1),
        ([ndx.asarray([1, 2], dtype=ndx.int32)], "hello"),
    ],
)
def test_promotion_failures(arrays, scalar):
    with pytest.raises(TypeError, match="Cannot promote"):
        promote(*arrays, scalar)


@pytest.mark.skipif(
    np.__version__ <= "1",
    reason="Cross kind scalar promotion not specified in NumPy < 2",
)
@pytest.mark.parametrize(
    "x, y",
    [
        (np.asarray([1, 2, 3], dtype=np.int64), 1.12),
        (1.23, np.asarray([1, 2, 3], dtype=np.int64)),
        (True, np.asarray([1, 2, 3], dtype=np.int8)),
        # (np.asarray([True, False]), 1.12),
        (np.asarray([True, False]), 4),
        (np.asarray([1.23, 2.34]), True),
        (np.asarray([1.23, 2.34]), 2),
    ],
)
def test_cross_kind_promotions(x, y):
    np_result = x + y
    if isinstance(x, np.ndarray):
        x = ndx.asarray(x)
    if isinstance(y, np.ndarray):
        y = ndx.asarray(y)
    onnx_result = x + y
    assert_array_equal(onnx_result.to_numpy(), np_result)


@pytest.mark.parametrize(
    "x, y, expected",
    [
        ([0], [], []),
        ([[0], [0]], [0, 0, 0], [[0, 0, 0], [0, 0, 0]]),
        ([1, 2, 3],) * 3,
        ([],) * 3,
    ],
)
@pytest.mark.parametrize(
    "x_dtype, y_dtype, expected_dtype",
    [
        (ndx.int64,) * 3,
        (ndx.nint64,) * 3,
        (ndx.float64,) * 3,
        (ndx.nfloat64,) * 3,
        (ndx.int64, ndx.nint64, ndx.nint64),
        (ndx.float64, ndx.nfloat64, ndx.nfloat64),
        (ndx.nint64, ndx.float64, ndx.nfloat64),
    ],
)
def test_where_equal_arrays(x, y, expected, x_dtype, y_dtype, expected_dtype):
    cond = ndx.array(shape=(), dtype=ndx.bool)
    x = ndx.asarray(x).astype(x_dtype)
    y = ndx.asarray(y).astype(y_dtype)

    result = ndx.where(cond, x, y)

    assert result.dtype == expected_dtype
    # NumPy simply drops the masked array in `np.where`. We do not want to do the same.

    # This optimization looks odd. I see that we may want to do this
    # if `x is y` but it seems odd to actually do an optimization like
    # this based on the values.
    # np.testing.assert_array_equal(result.to_numpy(), expected)


@pytest.mark.parametrize(
    "x, expected_shape",
    [
        (ndx.array(shape=(1, 2), dtype=ndx.utf8), (1, 2)),
        (ndx.array(shape=(1, 2), dtype=ndx.nuint8), (1, 2)),
        (ndx.array(shape=(1, None, 5), dtype=ndx.nuint8), (1, None, 5)),
        (ndx.array(shape=("N", "M"), dtype=ndx.float64), (None, None)),
        (
            ndx.reshape(
                ndx.array(shape=(4,), dtype=ndx.utf8),
                ndx.array(shape=(1,), dtype=ndx.int64),
            ),
            (None,),
        ),
        (
            ndx.reshape(
                ndx.array(shape=(4,), dtype=ndx.utf8),
                ndx.asarray([2, 2], dtype=ndx.int64),
            ),
            (2, 2),
        ),
    ],
)
def test_lazy_array_shape(x, expected_shape):
    assert x.shape == expected_shape


@pytest.mark.parametrize(
    "x, shape",
    [
        (
            ndx.array(shape=("N",), dtype=ndx.utf8),
            ndx.array(shape=("M",), dtype=ndx.int64),
        ),
        (
            ndx.array(shape=("N", 1), dtype=ndx.utf8),
            ndx.array(shape=("N",), dtype=ndx.int64),
        ),
        (
            ndx.array(shape=(1,), dtype=ndx.utf8),
            ndx.array(shape=("N",), dtype=ndx.int64),
        ),
        (
            ndx.array(shape=(), dtype=ndx.utf8),
            ndx.array(shape=("N",), dtype=ndx.int64),
        ),
    ],
)
def test_dynamic_reshape_has_no_static_shape(x, shape):
    with pytest.raises(
        ValueError,
        match="'shape' must be a 1D tensor of static shape if provided as an 'Array'",
    ):
        ndx.reshape(x, shape).shape


@pytest.mark.skipif(
    not np.__version__.startswith("2"), reason="NumPy >= 2 used for test assertions"
)
@pytest.mark.parametrize("include_initial", [True, False])
@pytest.mark.parametrize(
    "array_dtype",
    [ndx.int32, ndx.int64, ndx.float32, ndx.float64, ndx.uint8, ndx.uint16, ndx.uint32],
)
@pytest.mark.parametrize(
    "array, axis",
    [
        ([1, 2, 3], None),
        ([100, 100], None),
        ([1, 2, 3], 0),
        ([[1, 2], [3, 4]], 0),
        ([[1, 2], [3, 4]], 1),
        ([[1, 2, 50], [3, 4, 5]], 1),
        ([[[[1]]], [[[3]]]], 0),
        ([[[[1]]], [[[3]]]], 1),
    ],
)
@pytest.mark.parametrize(
    "cumsum_dtype",
    [None, ndx.int32, ndx.float32, ndx.float64, ndx.uint8, ndx.int8],
)
def test_cumulative_sum(array, axis, include_initial, array_dtype, cumsum_dtype):
    a = ndx.asarray(array, dtype=array_dtype)
    assert_array_equal(
        ndx.cumulative_sum(
            a, include_initial=include_initial, axis=axis, dtype=cumsum_dtype
        ).to_numpy(),
        np.cumulative_sum(
            np.asarray(array, a.dtype.to_numpy_dtype()),
            include_initial=include_initial,
            axis=axis,
            dtype=cumsum_dtype.to_numpy_dtype() if cumsum_dtype is not None else None,
        ),
    )


def test_no_unsafe_cumulative_sum_cast():
    with pytest.warns(match="A lossy cast to 'float64' is used instead"):
        a = ndx.asarray([1, 2, 3], dtype=ndx.uint64)
        ndx.cumulative_sum(a)

    with pytest.warns(match="A lossy cast to 'float64' is used instead"):
        a = ndx.asarray([1, 2, 3], dtype=ndx.int32)
        ndx.cumulative_sum(a, dtype=ndx.uint64)


@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize(
    "func, x",
    [
        (np.argmax, np.array([1, 2, 3, 4, 5], dtype=np.int32)),
        (np.argmax, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)),
        (np.argmax, np.array([1, 2, 3, 4, 5], dtype=np.int8)),
        (np.argmax, np.array([1, 2, 3, 4, 5], dtype=np.float32)),
        (np.argmax, np.array([1, 2, 3, 4, 5], dtype=np.float64)),
        (np.argmin, np.array([1, 2, 3, 4, 5], dtype=np.float32)),
        (np.argmin, np.array([[-11, 2, 3], [4, 5, -6]], dtype=np.int32)),
        (np.argmin, np.array([1, 2, 3, 4, 5], dtype=np.float64)),
        (np.argmin, np.array([1, 2, 3, 4, 5], dtype=np.int16)),
    ],
)
def test_argmaxmin(func, x, keepdims):
    np_result = func(x, keepdims=keepdims)
    ndx_result = getattr(ndx, func.__name__)(
        ndx.asarray(x), keepdims=keepdims
    ).to_numpy()
    assert_array_equal(np_result, ndx_result)
