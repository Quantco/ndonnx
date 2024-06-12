# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np
import numpy.array_api as npx
import pytest

import ndonnx as ndx
import ndonnx.additional as nda

from .utils import run


def testfill_null():
    a = ndx.array(shape=(3,), dtype=ndx.nint64)
    b = nda.fill_null(a, ndx.asarray(0))

    model = ndx.build({"a": a}, {"b": b})

    expected_b = np.ma.masked_array([1, 2, 0], mask=[0, 0, 0], dtype=np.int64)
    np.testing.assert_equal(
        expected_b,
        run(
            model, {"a": np.ma.masked_array([1, 2, -1], mask=[0, 0, 1], dtype=np.int64)}
        )["b"],
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

    assert np.ma.allclose(expected_c, ret_c, masked_equal=True)


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

    np.testing.assert_almost_equal(expected_b, ret_b)


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
    np_fn = getattr(npx, fn_name)

    a = ndx.array(shape=(1, 3), dtype=ndx.nfloat32)
    b = fn(a, *args)

    model = ndx.build({"a": a}, {"b": b})

    inp_a = np.ma.masked_array([[0, -2.0, 3.0]], mask=[[1, 0, 0]], dtype=np.float32)
    ret_b = run(model, {"a": inp_a})["b"]
    missing_a = inp_a.mask
    expected_b = np_fn(npx.asarray(np.ma.filled(inp_a, np.nan)), *args, **kwargs)
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

    assert np.ma.allclose(
        np.ma.masked_array([1, 2, 3], mask=[0, 0, 1]) + 1,
        a.to_numpy(),
        masked_equal=True,
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
    np.testing.assert_equal(shape.to_numpy(), np.array([3], dtype=np.int64))


def test_eager_mode():
    a = ndx.asarray(np.ma.masked_array([1, 2, 3], mask=[0, 0, 1], dtype=np.int64))
    b = ndx.asarray(np.ma.masked_array([1, 2, 3], mask=[0, 0, 1], dtype=np.int64))
    c = ndx.asarray(
        np.ma.masked_array([-12, 21, 12213], mask=[1, 0, 0], dtype=np.int64)
    )
    np.testing.assert_equal(
        (a + b).to_numpy(),
        np.ma.masked_array([2, 4, 6], mask=[0, 0, 1], dtype=np.int64),
    )
    np.testing.assert_equal(
        (a - b).to_numpy(),
        np.ma.masked_array([0, 0, 0], mask=[0, 0, 1], dtype=np.int64),
    )
    np.testing.assert_equal(
        (a * c).to_numpy(),
        np.ma.masked_array([-12, 42, 36639], mask=[1, 0, 1], dtype=np.int64),
    )
