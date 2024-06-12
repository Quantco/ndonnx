# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re

import numpy as np
import numpy.array_api as npx
import pytest
import spox.opset.ai.onnx.v19 as op

import ndonnx as ndx
import ndonnx.additional as nda
from ndonnx import _data_types as dtypes

from .utils import run


def numpy_to_graph_input(arr, eager=False):
    dtype: dtypes.CoreType | dtypes.StructType
    if isinstance(arr, np.ma.MaskedArray):
        dtype = dtypes.promote_nullable(dtypes.from_numpy_dtype(arr.dtype))
    else:
        dtype = dtypes.from_numpy_dtype(arr.dtype)
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
    np.testing.assert_equal(input, actual["b"])


def test_null_promotion():
    a = ndx.array(shape=("N",), dtype=ndx.nfloat64)
    b = ndx.array(shape=("N",), dtype=ndx.float64)
    model = ndx.build({"a": a, "b": b}, {"c": a + b})
    inputs = {
        "a": np.ma.masked_array([1.0, 2.0, 3.0, 4.0], mask=[0, 0, 1, 0]),
        "b": np.array([4.0, 5.0, 6.0, 7.0]),
    }
    actual = run(model, inputs)
    np.testing.assert_equal(np.add(inputs["a"], inputs["b"]), actual["c"])


def test_asarray():
    a = ndx.asarray([1, 2, 3], dtype=ndx.int64)
    assert a.dtype == ndx.int64
    np.testing.assert_array_equal(
        np.array([1, 2, 3], np.int64), a.to_numpy(), strict=True
    )


def test_asarray_masked():
    np_arr = np.ma.masked_array([1, 2, 3], mask=[0, 0, 1])
    ndx_arr = ndx.asarray(np_arr)
    assert isinstance(ndx_arr, ndx.Array)
    assert isinstance(ndx_arr.to_numpy(), np.ma.MaskedArray)
    np.testing.assert_array_equal(np_arr, ndx_arr.to_numpy())


def test_basic_eager_add(_a, _b):
    a = ndx.asarray(_a)
    b = ndx.asarray(_b)
    x = ndx.asarray([1]) + 2
    c = a + b + x
    np.testing.assert_array_equal(c.to_numpy(), _a + _b + np.array([1]) + 2)


def test_string_concatenation():
    a = ndx.asarray(["a", "b", "c"])
    b = ndx.asarray(["d", "e", "f"])
    np.testing.assert_array_equal((a + b).to_numpy(), np.array(["ad", "be", "cf"]))
    np.testing.assert_array_equal((a + "d").to_numpy(), np.array(["ad", "bd", "cd"]))


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
    np.testing.assert_equal(actual, expected)


def test_indexing(_a):
    a = numpy_to_graph_input(_a)
    b = a[0]

    model = ndx.build({"a": a}, {"b": b})
    actual = run(model, {"a": _a})["b"]
    expected = _a[0]
    np.testing.assert_array_equal(expected, actual)


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
    np.testing.assert_array_equal(expected, actual)


def test_indexing_on_scalar():
    res = ndx.asarray(1)
    res = res[()]
    res[()] = 2
    assert res == 2


def test_indexing_on_scalar_mask():
    res = ndx.asarray([])
    res = res[False]
    np.testing.assert_equal(nda.shape(res).to_numpy(), (0, 0))


def test_indexing_with_mask(_a):
    _mask = np.array([True, False, True])

    a = numpy_to_graph_input(_a)
    mask = numpy_to_graph_input(_mask)
    c = ndx.reshape(a[mask], [-1])

    expected_c = _a[_mask]

    model = ndx.build({"a": a, "mask_": mask}, {"c": c})
    actual = run(model, {"a": _a, "mask_": _mask})["c"]
    np.testing.assert_array_equal(expected_c, actual)


def test_indexing_with_mask_raises(_a):
    a = numpy_to_graph_input(_a)
    mask = ndx.array(shape=(3, 1, 1), dtype=ndx.bool)

    with pytest.raises(IndexError):
        a[mask]


def test_indexing_with_array(_a):
    _index = np.array([0, 2])

    a = numpy_to_graph_input(_a)
    index = numpy_to_graph_input(_index)
    c = ndx.reshape(a[index], [-1])

    expected_c = _a[_index]

    model = ndx.build({"a": a, "index_": index}, {"c": c})
    actual = run(model, {"a": _a, "index_": _index})["c"]
    np.testing.assert_array_equal(expected_c, actual)


def test_indexing_with_tuple_of_array(_a):
    a = numpy_to_graph_input(_a)
    index = numpy_to_graph_input(np.array([0, 2]))

    with pytest.raises(
        TypeError,
        match=re.escape(f"Index {index} for type {type(index)} not supported"),
    ):
        a[(index,)]


def test_slicing(_a):
    a = numpy_to_graph_input(_a)
    b = a[1:2]

    model = ndx.build({"a": a}, {"b": b})
    actual = run(model, {"a": _a})["b"]
    np.testing.assert_equal(actual, _a[1:2])


def test_indexing_list(_a):
    a = numpy_to_graph_input(_a)

    with pytest.raises(TypeError, match="not supported"):
        a[[0, 2]]


def test_indexing_slice_ellipsis(_c):
    a = numpy_to_graph_input(_c)
    b = a[..., 1]

    model = ndx.build({"a": a}, {"b": b})
    actual = run(model, {"a": _c})["b"]
    expected = _c[..., 1]
    np.testing.assert_array_equal(actual, expected)


def test_indexing_none(_c):
    _index = (None, 1, None, 1)

    a = numpy_to_graph_input(_c)
    b = a[_index]

    model = ndx.build({"a": a}, {"b": b})
    actual = run(model, {"a": _c})["b"]
    expected = _c[_index]
    np.testing.assert_array_equal(actual, expected)


def test_illegal_indexing(_a):
    a = numpy_to_graph_input(_a)

    with pytest.raises(TypeError):
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


def test_indexing_set_with_array(_a):
    _index = np.array([0, 2])

    a = numpy_to_graph_input(_a)
    index = numpy_to_graph_input(_index)

    c = a.copy()
    c[index] = 0

    expected_c = _a
    expected_c[_index] = 0

    model = ndx.build({"a": a, "index_": index}, {"c": c})
    actual = run(model, {"a": _a, "index_": _index})["c"]
    np.testing.assert_equal(actual, expected_c)


def test_indexing_set_scalar():
    _a = np.array(2)

    a = numpy_to_graph_input(_a)

    c = a.copy()
    c = ndx.asarray(0)

    expected_c = 0

    model = ndx.build({"a": a}, {"c": c})
    np.testing.assert_array_equal(expected_c, run(model, {"a": _a})["c"], strict=True)


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
    np.testing.assert_equal(res.to_numpy(), getattr(a.to_numpy(), rop)(b.to_numpy()))


def test_rsub():
    a = ndx.asarray([1, 2, 3])

    res = 1 - a
    a_value = a.to_numpy()
    assert a_value is not None
    np.testing.assert_equal(res.to_numpy(), 1 - a_value)


def test_matrix_transpose():
    a = ndx.array(shape=(3, 2, 3), dtype=ndx.int64)
    b = ndx.matrix_transpose(a)

    model = ndx.build({"a": a}, {"b": b})
    np.testing.assert_equal(
        npx.matrix_transpose(npx.reshape(npx.arange(3 * 2 * 3), (3, 2, 3))),
        run(model, {"a": np.arange(3 * 2 * 3, dtype=np.int64).reshape(3, 2, 3)})["b"],
    )


def test_matrix_transpose_attribute():
    a = ndx.array(shape=(3, 2, 3), dtype=ndx.int64)
    b = a.mT

    model = ndx.build({"a": a}, {"b": b})
    np.testing.assert_equal(
        npx.reshape(npx.arange(3 * 2 * 3), (3, 2, 3)).mT,
        run(model, {"a": np.arange(3 * 2 * 3, dtype=np.int64).reshape(3, 2, 3)})["b"],
    )


def test_transpose_attribute():
    a = ndx.array(shape=(3, 2), dtype=ndx.int64)
    b = a.T

    model = ndx.build({"a": a}, {"b": b})
    np.testing.assert_equal(
        npx.reshape(npx.arange(3 * 2), (3, 2)).T,
        run(model, {"a": np.arange(3 * 2, dtype=np.int64).reshape(3, 2)})["b"],
    )


def test_array_spox_interoperability():
    a = ndx.array(shape=(3, 2), dtype=ndx.nint64)
    add_var = op.add(a.values.data.var, op.const(5, dtype=np.int64))  # type: ignore
    b = ndx.from_spox_var(var=add_var)
    model = ndx.build({"a": a}, {"b": b})
    expected = npx.reshape(npx.arange(3 * 2), (3, 2)) + 5
    input = np.ma.masked_array(
        np.arange(3 * 2, dtype=np.int64).reshape(3, 2), mask=np.ones((3, 2), dtype=bool)
    )
    actual = run(model, {"a": input})["b"]
    np.testing.assert_equal(expected, actual)


def test_creation_arange():
    a = ndx.arange(10)
    np.testing.assert_equal(np.arange(10), a.to_numpy())

    b = ndx.arange(1, 10)
    np.testing.assert_equal(np.arange(1, 10), b.to_numpy())

    c = ndx.arange(1, 10, 2)
    np.testing.assert_equal(np.arange(1, 10, 2), c.to_numpy())

    d = ndx.arange(0.0, None, step=-1)
    np.testing.assert_array_equal(
        np.arange(0.0, None, step=-1), d.to_numpy(), strict=True
    )


def test_creation_full():
    a = ndx.full((2, 3), 5)
    np.testing.assert_equal(np.full((2, 3), 5), a.to_numpy())

    b = ndx.full((2, 3), 5, dtype=ndx.float32)
    np.testing.assert_equal(np.full((2, 3), 5, dtype=np.float32), b.to_numpy())
    c = ndx.full((2, 3), "a", dtype=ndx.nutf8)
    np.testing.assert_equal(np.full((2, 3), "a"), c.to_numpy())


@pytest.mark.parametrize(
    "args, expected",
    [
        ((ndx.int32, ndx.int64), ndx.int64),
        (
            (ndx.asarray([1, 2, 3], ndx.int64), ndx.asarray([1, 2, 3], ndx.int32)),
            ndx.int64,
        ),
        ((ndx.int32, ndx.asarray([1, 2, 3], ndx.int64)), ndx.int64),
        ((ndx.float32, ndx.float64), ndx.float64),
        ((ndx.float64, ndx.int32), ndx.float64),
    ],
)
def test_result_type(args, expected):
    assert ndx.result_type(*args) == expected


def test_ceil():
    a = ndx.asarray(np.array([1.2, 1.3, -13.13]))
    np.testing.assert_equal(ndx.ceil(a).to_numpy(), [2.0, 2.0, -13.0])


def test_propagates_minimal_dtype():
    a = ndx.asarray([1, 2, 4], dtype=ndx.int8)
    b = a + 1
    assert b.dtype == ndx.int8
    np.testing.assert_equal(b.to_numpy(), [2, 3, 5])


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
    np.testing.assert_equal(ndx.all(x).to_numpy(), np.all(x.to_numpy()))


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
    np.testing.assert_equal(c_val, c.to_numpy())


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

    np.testing.assert_equal(c_val, run(model, dict(a=a_val, b=b_val))["c"])


def test_searchsorted_raises():
    with pytest.raises(TypeError):
        a = ndx.array(shape=(), dtype=ndx.int64)
        b = ndx.array(shape=(), dtype=ndx.float64)

        ndx.searchsorted(a, b)

    with pytest.raises(ValueError):
        a = ndx.array(shape=(3,), dtype=ndx.int64)
        b = ndx.array(shape=(3,), dtype=ndx.int64)

        ndx.searchsorted(a, b, side="middle")  # type: ignore[arg-type]
