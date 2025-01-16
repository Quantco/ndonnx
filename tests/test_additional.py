# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import pytest

import ndonnx as ndx
import ndonnx.additional as nda

from .utils import assert_array_equal, run


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


def test_static_map_lazy():
    a = ndx.array(shape=(3,), dtype=ndx.int64)
    b = nda.static_map(a, {1: 2, 2: 3})
    model = ndx.build({"a": a}, {"b": b})
    assert_array_equal([0, 2, 3], run(model, {"a": np.array([0, 1, 2])})["b"])

    # nans are mapped by static_map
    a = ndx.array(shape=("N",), dtype=ndx.float64)
    b = nda.static_map(a, {1.0: 2, np.nan: 3, 3.0: 42}, default=-1)

    model = ndx.build({"a": a}, {"b": b})
    assert_array_equal(
        [-1, -1, 42, 3],
        run(model, {"a": np.array([0.0, 2.0, 3.0, np.nan])})["b"],
    )


@pytest.mark.parametrize(
    "x, mapping, default, expected",
    [
        (
            ndx.asarray(["hello", "world", "!"]),
            {"hello": "hi", "world": "earth"},
            None,
            ["hi", "earth", "MISSING"],
        ),
        (
            ndx.asarray(["hello", "world", "!"]),
            {"hello": "hi", "world": "earth"},
            "DIFFERENT",
            ["hi", "earth", "DIFFERENT"],
        ),
        (ndx.asarray([0, 1, 2], dtype=ndx.int64), {0: -1, 1: -2}, None, [-1, -2, 0]),
        (ndx.asarray([0, 1, 2], dtype=ndx.int64), {0: -1, 1: -2}, 42, [-1, -2, 42]),
        (
            ndx.asarray([[0], [1], [2]], dtype=ndx.int64),
            {0: -1, 1: -2},
            42,
            [[-1], [-2], [42]],
        ),
        (
            ndx.asarray([[0], [1], [2]], dtype=ndx.int32),
            {0: -1, 1: -2},
            42,
            [[-1], [-2], [42]],
        ),
        (
            ndx.asarray([[0], [1], [2]], dtype=ndx.int8),
            {0: -1, 1: -2},
            42,
            [[-1], [-2], [42]],
        ),
        (
            ndx.asarray([[0], [1], [2]], dtype=ndx.uint8),
            {0: -1, 1: -2},
            42,
            [[-1], [-2], [42]],
        ),
        (
            ndx.asarray([[0], [1], [np.nan]], dtype=ndx.float32),
            {0: -1, 1: -2, np.nan: 3.142},
            42,
            [[-1], [-2], [3.142]],
        ),
    ],
)
def test_static_map(x, mapping, default, expected):
    actual = nda.static_map(x, mapping, default=default)
    assert_array_equal(actual.to_numpy(), expected)


def test_static_map_unimplemented_for_nullable():
    a = ndx.asarray([1, 2, 3], dtype=ndx.int64)
    m = ndx.asarray([True, False, True])
    a = nda.make_nullable(a, m)

    with pytest.raises(ndx.UnsupportedOperationError):
        nda.static_map(a, {1: 2, 2: 3})


def test_isin():
    a = ndx.array(shape=("N",), dtype=ndx.utf8)
    b = nda.isin(a, ["foo", "bar", "baz"])

    model = ndx.build({"a": a}, {"b": b})
    assert_array_equal(
        [False, True, True, False],
        run(model, dict(a=np.array(["hello", "foo", "baz", "!"])))["b"],
    )

    a = ndx.array(shape=("N",), dtype=ndx.utf8)
    b = nda.isin(a, ["🔴", "🟡", "🟢"])

    model = ndx.build({"a": a}, {"b": b})
    assert_array_equal(
        [False, True, False],
        run(model, dict(a=np.array(["🚀", "🔴", "hi🟡"])))["b"],
    )

    a = ndx.asarray(["hello", "world"])
    assert_array_equal([True, False], nda.isin(a, ["hello"]).to_numpy())


@pytest.mark.parametrize(
    "dtype",
    [
        ndx.int64,
        ndx.utf8,
        ndx.bool,
    ],
)
@pytest.mark.parametrize(
    "mask",
    [
        [True, False, True],
        True,
        False,
        [True],
    ],
)
def test_make_nullable(dtype, mask):
    a = ndx.asarray([1, 2, 3], dtype=dtype)
    m = ndx.asarray(mask)

    result = nda.make_nullable(a, m)
    expected = np.ma.masked_array([1, 2, 3], mask, dtype.to_numpy_dtype())
    assert_array_equal(result.to_numpy(), expected)
