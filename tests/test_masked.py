# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import math
import warnings

import numpy as np
import pytest

import ndonnx as ndx
import ndonnx.extensions as nda

from .utils import assert_array_equal, get_numpy_array_api_namespace, run


def testfill_null():
    a = ndx.asarray(np.ma.masked_array([1, 2, -1], mask=[0, 0, 1], dtype=np.int64))
    b = nda.fill_null(a, ndx.asarray(0))
    expected_b = np.array([1, 2, 0], dtype=np.int64)
    assert_array_equal(b.unwrap_numpy(), expected_b)


@pytest.mark.parametrize(
    "fn_name",
    ["add", "subtract", "multiply", "divide", "floor_divide", "remainder"],
)
def test_arithmetic_none_propagation(fn_name):
    fn = getattr(ndx, fn_name)
    np_fn = getattr(np.ma, fn_name)

    a_np = np.ma.MaskedArray([-1, 2.0, 3.0], mask=[1, 0, 0])
    b_np = np.ma.MaskedArray([2.0, 1.0, -1], mask=[0, 0, 1])
    a = ndx.asarray(a_np, dtype=ndx.nfloat64)
    b = ndx.asarray(b_np, dtype=ndx.nfloat64)
    c = fn(a, b)

    expected_c = np_fn(a_np, b_np)

    assert_array_equal(c.unwrap_numpy(), expected_c)


@pytest.mark.xfail(reason="Masked reduction ops are ill-defined")
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

    a = ndx.argument(shape=(3,), dtype=ndx.nfloat32)
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
    a_np = np.ma.masked_array([[0, -2.0, 3.0]], mask=[[1, 0, 0]], dtype=np.float32)
    fn = getattr(ndx, fn_name)

    a = ndx.asarray(a_np, dtype=ndx.nfloat32)
    b = fn(a, *args)

    # model = ndx.build({"a": a}, {"b": b})
    # ret_b = run(model, {"a": inp_a})["b"]
    missing_a = a_np.mask
    if np.__version__ < "2":
        if not (np_fn := getattr(np.ma, fn_name, None)):
            pytest.skip(reason=f"function `{fn_name}` not supported for np1x.")
    else:
        npx = get_numpy_array_api_namespace()
        np_fn = getattr(npx, fn_name)

    # Numpy might complain about invalid values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected_b = np_fn(a_np, *args, **kwargs)
    np.testing.assert_almost_equal(
        np.ma.masked_array(expected_b, mask=missing_a),
        b.unwrap_numpy(),
        decimal=5,
    )


def test_forbidden_masked():
    a = ndx.argument(shape=(3,), dtype=ndx.nint64)

    with pytest.raises(TypeError):
        ndx.arange(a, 0, 1)


def test_masked_getitem():
    a_np = np.ma.masked_array([1, 2, 3], mask=[0, 0, 1], dtype=np.int64)
    a = ndx.asarray(a_np, dtype=ndx.nint64)

    np.testing.assert_equal([1], a[0].unwrap_numpy())


def test_masked_setitem():
    a_np = np.ma.masked_array([1, 2, 3], mask=[0, 0, 0], dtype=np.int64)
    a_start = ndx.asarray(a_np, dtype=ndx.nint64)

    a = a_start.copy()
    a[0] = 10

    np.testing.assert_equal([10, 2, 3], a.unwrap_numpy())


def test_asarray_masked():
    a = ndx.asarray(np.ma.masked_array([1, 2, 3], mask=[0, 0, 1])) + 1
    assert_array_equal(
        a.unwrap_numpy(),
        np.ma.masked_array([2, 3, 4], mask=[0, 0, 1]),
    )


def test_eager_mode():
    a = ndx.asarray(np.ma.masked_array([1, 2, 3], mask=[0, 0, 1], dtype=np.int64))
    b = ndx.asarray(np.ma.masked_array([1, 2, 3], mask=[0, 0, 1], dtype=np.int64))
    c = ndx.asarray(
        np.ma.masked_array([-12, 21, 12213], mask=[1, 0, 0], dtype=np.int64)
    )
    assert_array_equal(
        (a + b).unwrap_numpy(),
        np.ma.masked_array([2, 4, 6], mask=[0, 0, 1], dtype=np.int64),
    )
    assert_array_equal(
        (a - b).unwrap_numpy(),
        np.ma.masked_array([0, 0, 0], mask=[0, 0, 1], dtype=np.int64),
    )
    assert_array_equal(
        (a * c).unwrap_numpy(),
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
    pytest.skip("NumPy's 'tril'/'triu' does not propagate the mask")
    actual = getattr(ndx, func.__name__)(ndx.asarray(a))
    assert_array_equal(actual.unwrap_numpy(), expected)


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
        assert a.shape == e.shape


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
    null = nda.get_mask(actual)
    assert_array_equal(actual.unwrap_numpy(), np_array)
    if null is None:
        assert np_array.mask is np.ma.nomask
    else:
        np.testing.assert_equal(np_array.mask, null.unwrap_numpy())


@pytest.mark.parametrize(
    "npdtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ],
)
def test_masked_integer_to_datetime(npdtype):
    arr = ndx.asarray(np.ma.MaskedArray([1, 2], mask=[False, True], dtype=npdtype))

    expected = np.asarray([1, "NaT"], dtype="datetime64[s]")
    candidate = arr.astype(ndx.DateTime64DType("s"))

    np.testing.assert_array_equal(expected, candidate.unwrap_numpy())


def test_masked_int64_with_sentinel_to_datetime():
    arr = ndx.asarray(
        np.ma.MaskedArray(
            [1, 2, np.iinfo(np.int64).min], mask=[False, True, False], dtype=np.int64
        )
    )

    expected = np.asarray([1, "NaT", "NaT"], dtype="datetime64[s]")
    candidate = arr.astype(ndx.DateTime64DType("s"))

    np.testing.assert_array_equal(expected, candidate.unwrap_numpy())


def test_masked_float_to_datetime():
    arr = ndx.asarray(
        np.ma.MaskedArray([1, 2, np.nan], mask=[False, True, False], dtype=np.float64)
    )

    expected = np.asarray([1, "NaT", "NaT"], dtype="datetime64[s]")
    candidate = arr.astype(ndx.DateTime64DType("s"))

    np.testing.assert_array_equal(expected, candidate.unwrap_numpy())


def test_static_map_nutf8():
    np_in = np.ma.MaskedArray(["foo", "bar", "baz"], mask=[0, 1, 0])  # type: ignore
    arr = ndx.asarray(np_in)
    candidate = ndx.extensions.static_map(arr, {"foo": "FOO", "bar": "BAR"})

    assert candidate.dtype == ndx.nutf8
    np.testing.assert_array_equal(
        np_in.mask,
        candidate.unwrap_numpy().mask,  # type: ignore
    )

    np.testing.assert_array_equal(
        ["FOO", "MISSING"],
        candidate.unwrap_numpy().data[~candidate.unwrap_numpy().mask],  # type: ignore
    )


def test_static_map_int64():
    np_in = np.ma.MaskedArray([1, 2, 3], mask=[0, 1, 0], dtype=np.int64)  # type: ignore
    arr = ndx.asarray(np_in)
    candidate = ndx.extensions.static_map(arr, {1: 10, 2: 20})

    assert candidate.dtype == ndx.nint64
    np.testing.assert_array_equal(
        np_in.mask,
        candidate.unwrap_numpy().mask,  # type: ignore
    )

    np.testing.assert_array_equal(
        [10, 0],
        candidate.unwrap_numpy().data[~candidate.unwrap_numpy().mask],  # type: ignore
    )
