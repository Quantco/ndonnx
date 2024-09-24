# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import math
import warnings

import numpy as np
import pytest

import ndonnx as ndx
import ndonnx.additional as nda

from .utils import assert_array_equal, get_numpy_array_api_namespace, run


def testfill_null():
    a = ndx.array(shape=(3,), dtype=ndx.nint64)
    b = nda.fill_null(a, ndx.asarray(0))

    model = ndx.build({"a": a}, {"b": b})

    expected_b = np.array([1, 2, 0], dtype=np.int64)
    assert_array_equal(
        run(
            model, {"a": np.ma.masked_array([1, 2, -1], mask=[0, 0, 1], dtype=np.int64)}
        )["b"],
        expected_b,
    )


@pytest.mark.parametrize(
    "fn_name",
    ["add", "subtract", "multiply", "divide", "floor_divide", "remainder"],
)
def test_arithmetic_none_propagation(fn_name):
    fn = getattr(ndx, fn_name)
    np_fn = getattr(np.ma, fn_name)

    a = ndx.array(shape=(3,), dtype=ndx.nfloat64)
    b = ndx.array(shape=(3,), dtype=ndx.nfloat64)
    c = fn(a, b)

    a_val = np.ma.masked_array([-1, 2.0, 3.0], mask=[1, 0, 0])
    b_val = np.ma.masked_array([2.0, 1.0, -1], mask=[0, 0, 1])

    model = ndx.build({"a": a, "b": b}, {"c": c})

    ret_c = run(model, {"a": a_val, "b": b_val})["c"]
    expected_c = np_fn(a_val, b_val)

    assert_array_equal(ret_c, expected_c)


@pytest.mark.parametrize(
    "fn_name, default_value",
    [
        ("sum", 0),
        # ("prod", 1),
        ("min", math.inf),
        ("max", -math.inf),
    ],
)
def test_reduce_ops_none_filling(fn_name, default_value):
    fn = getattr(ndx, fn_name)
    np_fn = getattr(np, fn_name)

    a = ndx.array(shape=(3,), dtype=ndx.nfloat32)
    b = fn(a)

    model = ndx.build({"a": a}, {"b": b})

    inp_a = np.ma.masked_array([0.0, -2.0, 3.0], dtype=np.float32, mask=[1, 0, 0])

    ret_b = run(model, {"a": inp_a})["b"]
    expected_b = np_fn(np.ma.filled(inp_a, default_value))

    assert_array_equal(ret_b, expected_b)


@pytest.mark.parametrize(
    "fn_name, args, kwargs",
    [
        ("abs", (), {}),
        ("exp", (), {}),
        ("log", (), {}),
        ("sqrt", (), {}),
        ("sin", (), {}),
        ("cos", (), {}),
        ("tan", (), {}),
        ("asin", (), {}),
        ("acos", (), {}),
        ("atan", (), {}),
        # ("sinh", (), {}),
        ("negative", (), {}),
        ("positive", (), {}),
        ("floor", (), {}),
        ("sign", (), {}),
        ("expand_dims", (), {"axis": 0}),
        ("reshape", ((3, 1),), {}),
        ("flip", (), {}),
        ("permute_dims", ((1, 0),), {}),
        ("roll", (1,), {}),
        ("squeeze", (0,), {}),
    ],
)
def test_unary_none_propagation(fn_name, args, kwargs):
    fn = getattr(ndx, fn_name)

    a = ndx.array(shape=(1, 3), dtype=ndx.nfloat32)
    b = fn(a, *args)

    model = ndx.build({"a": a}, {"b": b})

    inp_a = np.ma.masked_array([[0, -2.0, 3.0]], mask=[[1, 0, 0]], dtype=np.float32)
    ret_b = run(model, {"a": inp_a})["b"]
    missing_a = inp_a.mask
    npx = get_numpy_array_api_namespace()
    np_fn = getattr(npx, fn_name)
    inp_a = npx.asarray(np.ma.filled(inp_a, np.nan))

    # Numpy might complain about invalid values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected_b = np_fn(inp_a, *args, **kwargs)
    np.testing.assert_almost_equal(
        np.ma.masked_array(expected_b, mask=missing_a),
        ret_b,
        decimal=5,
    )


def test_forbidden_masked():
    a = ndx.array(shape=(3,), dtype=ndx.nint64)

    with pytest.raises(TypeError):
        ndx.arange(a, 0, 1)


def test_masked_getitem():
    a = ndx.array(shape=(3,), dtype=ndx.nint64)

    model = ndx.build({"a": a}, {"b": a[0]})

    np.testing.assert_equal(
        [1],
        run(
            model, {"a": np.ma.masked_array([1, 2, 3], mask=[0, 0, 1], dtype=np.int64)}
        )["b"],
    )


def test_masked_setitem():
    a_start = ndx.array(shape=(3,), dtype=ndx.nint64)

    a = a_start.copy()
    a[0] = 1

    model = ndx.build({"a": a_start}, {"b": a})

    np.testing.assert_equal(
        [1, 2, 3],
        run(
            model, dict(a=np.ma.masked_array([1, 2, 3], mask=[0, 0, 0], dtype=np.int64))
        )["b"],
    )


def test_asarray_masked():
    a = ndx.asarray(np.ma.masked_array([1, 2, 3], mask=[0, 0, 1])) + 1
    assert_array_equal(
        a.to_numpy(),
        np.ma.masked_array([2, 3, 4], mask=[0, 0, 1]),
    )


def test_opset_extensions():
    import ndonnx._opset_extensions as opx
    from ndonnx._corearray import _CoreArray

    a = ndx.asarray(
        np.array([1, 2, 3]),
        dtype=ndx.int64,
    )
    shape = opx.shape(a.data)  # type: ignore
    # opset_extensions is an internal package that only deals with the internal state (lazy Spox Var and any eager values)
    assert isinstance(shape, _CoreArray)
    assert_array_equal(shape.to_numpy(), np.array([3], dtype=np.int64))


def test_eager_mode():
    a = ndx.asarray(np.ma.masked_array([1, 2, 3], mask=[0, 0, 1], dtype=np.int64))
    b = ndx.asarray(np.ma.masked_array([1, 2, 3], mask=[0, 0, 1], dtype=np.int64))
    c = ndx.asarray(
        np.ma.masked_array([-12, 21, 12213], mask=[1, 0, 0], dtype=np.int64)
    )
    assert_array_equal(
        (a + b).to_numpy(),
        np.ma.masked_array([2, 4, 6], mask=[0, 0, 1], dtype=np.int64),
    )
    assert_array_equal(
        (a - b).to_numpy(),
        np.ma.masked_array([0, 0, 0], mask=[0, 0, 1], dtype=np.int64),
    )
    assert_array_equal(
        (a * c).to_numpy(),
        np.ma.masked_array([-12, 42, 36639], mask=[1, 0, 1], dtype=np.int64),
    )


@pytest.mark.parametrize(
    "func",
    [
        np.tril,
        np.triu,
    ],
)
def test_trilu_masked_input(func):
    a = np.ma.masked_array(
        [[1, 2, 3], [4, 5, 6]], mask=[[0, 0, 1], [0, 0, 0]], dtype=np.int64
    )
    expected = func(a)
    actual = getattr(ndx, func.__name__)(ndx.asarray(a))
    assert_array_equal(actual.to_numpy(), expected)


@pytest.mark.parametrize(
    "arrays",
    [
        [
            np.ma.masked_array([1, 2, 3], mask=[0, 0, 1], dtype=np.int64),
            np.ma.masked_array([1, 2, 3], mask=[0, 0, 1], dtype=np.int64),
        ],
        [],
        [np.ma.masked_array([1, 2, 3], mask=[0, 0, 1], dtype=np.int64)],
        [
            np.ma.masked_array([1, 2, 3], mask=[0, 0, 1], dtype=np.int64),
            np.ma.masked_array(["a", "bc", "d"], mask=[0, 0, 1], dtype=np.str_),
        ],
        [
            np.ma.masked_array([1, 2, 3], mask=[0, 0, 1], dtype=np.int64),
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64),
            np.array("a"),
        ],
        [
            np.array([1, 2, 3], dtype=np.int64),
            np.ma.masked_array(["a", "bc", "d"], mask=[0, 0, 1], dtype=np.str_),
        ],
    ],
)
def test_broadcasting(arrays):
    expected = np.broadcast_arrays(*arrays)
    actual = ndx.broadcast_arrays(*[ndx.asarray(a) for a in arrays])
    for e, a in zip(expected, actual):
        # NumPy simply drops the masked array.
        # We do not want to do the same quite intentionally.
        np.testing.assert_equal(a.to_numpy(), e)


@pytest.mark.parametrize(
    "np_array",
    [
        np.ma.masked_array([1, 2, 3], mask=[0, 0, 1], dtype=np.int64),
        np.ma.masked_array([1, 2, 3], mask=[0, 0, 1], dtype=np.float64),
        np.ma.masked_array([1, 2]),
        np.ma.masked_array(["a", "b"], mask=True),
        np.ma.masked_array([1, 2, 3], mask=[[[0]]]),
        np.ma.masked_array([[1, 2, 3]], mask=[True, False, True]),
        np.ma.masked_array([1.0, 2.0, 3.0], mask=[0, 0, 1]),
    ],
)
def test_initialization(np_array):
    actual = ndx.asarray(np_array)
    values = actual.values.to_numpy()
    null = actual.null.to_numpy()
    assert_array_equal(actual.to_numpy(), np_array)
    assert values.shape == null.shape
