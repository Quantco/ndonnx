# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import onnxruntime as ort
import pytest

import ndonnx as ndx
import ndonnx.extensions as nda

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
    assert_array_equal(c_val, c.unwrap_numpy())


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

    a = ndx.argument(shape=(len(a_val),), dtype=ndx.float64)
    b = ndx.argument(shape=(len(b_val),), dtype=ndx.float64)
    c = ndx.searchsorted(a, b, side=side)

    model = ndx.build({"a": a, "b": b}, {"c": c})

    assert_array_equal(c_val, run(model, dict(a=a_val, b=b_val))["c"])


def test_searchsorted_raises():
    with pytest.raises(TypeError):
        a = ndx.argument(shape=(), dtype=ndx.int64)
        b = ndx.argument(shape=(), dtype=ndx.float64)

        ndx.searchsorted(a, b)

    with pytest.raises(ValueError):
        a = ndx.argument(shape=(3,), dtype=ndx.int64)
        b = ndx.argument(shape=(3,), dtype=ndx.int64)

        ndx.searchsorted(a, b, side="middle")  # type: ignore[arg-type]


def test_static_map_lazy():
    a = ndx.argument(shape=(3,), dtype=ndx.int64)
    b = nda.static_map(a, {1: 2, 2: 3})
    model = ndx.build({"a": a}, {"b": b})
    assert_array_equal(
        np.asarray([0, 2, 3]), run(model, {"a": np.array([0, 1, 2])})["b"]
    )

    # nans are mapped by static_map
    a = ndx.argument(shape=("N",), dtype=ndx.float64)
    b = nda.static_map(a, {1.0: 2, np.nan: 3, 3.0: 42}, default=-1)

    model = ndx.build({"a": a}, {"b": b})
    assert_array_equal(
        np.asarray([-1, -1, 42, 3]),
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
        (
            ndx.asarray([[True], [True], [False]], dtype=ndx.bool),
            {True: 1, False: 0},
            True,
            [[1], [1], [0]],
        ),
    ],
)
def test_static_map(x, mapping, default, expected):
    actual = nda.static_map(x, mapping, default=default)
    assert_array_equal(actual.unwrap_numpy(), expected)


def test_static_map_numpy_float16_key():
    x = ndx.asarray([1.0, 2.0], dtype=ndx.float32)

    candidate = nda.static_map(x, {np.float16(1): 3}, default=0)

    np.testing.assert_array_equal(
        candidate.unwrap_numpy(), np.asarray([3, 0], dtype=np.int64), strict=True
    )


def test_static_map_numpy_float16_input_execution():
    x = ndx.argument(shape=(3,), dtype=ndx.float16)
    candidate = nda.static_map(x, {np.float16(1): 3, np.float16(2): 4}, default=0)

    actual = run(
        ndx.build({"x": x}, {"y": candidate}),
        {"x": np.asarray([1, 2, 3], dtype=np.float16)},
    )["y"]

    assert_array_equal(actual, np.asarray([3, 4, 0], dtype=np.int64))


@pytest.mark.parametrize(
    "scalar_type, mapped, default",
    [
        (np.int16, 32767, -32768),
        (np.int32, 2, 0),
        (np.int64, 2, 0),
        (np.float32, 2.5, 0.5),
        (np.float64, 2.5, 0.5),
        (np.str_, "mapped", "missing"),
        (np.bool_, True, False),
    ],
)
def test_static_map_numpy_output_execution(scalar_type, mapped, default):
    x = ndx.argument(shape=(2,), dtype=ndx.int32)
    mapped, default = scalar_type(mapped), scalar_type(default)
    candidate = nda.static_map(x, {1: mapped}, default=default)

    # Test kernel support independently of ORT's string-attribute optimizer bug.
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        ndx.build({"x": x}, {"y": candidate}).SerializeToString(),
        sess_options=options,
    )
    (actual,) = session.run(None, {"x": np.asarray([1, 3], dtype=np.int32)})

    assert candidate.dtype == ndx.from_numpy_dtype(np.asarray(mapped).dtype)
    expected = np.asarray(
        [mapped, default], dtype=object if scalar_type is np.str_ else None
    )
    assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1, 2], [False, True, True, False]),
        ([np.nan, 1, 2], [False, True, True, False]),
        ([np.nan], [False, False, False, False]),
    ],
)
def test_isin_numpy_float16_input_execution(values, expected):
    x = ndx.argument(shape=(4,), dtype=ndx.float16)
    candidate = nda.isin(x, [np.float16(value) for value in values])

    actual = run(
        ndx.build({"x": x}, {"y": candidate}),
        {"x": np.asarray([np.nan, 1, 2, 3], dtype=np.float16)},
    )["y"]

    assert_array_equal(actual, np.asarray(expected, dtype=np.bool_))


@pytest.mark.parametrize(
    "np_arr, test_items, desired",
    [
        (
            np.array(["hello", "foo", "baz", "!"]),
            ["foo", "bar", "baz"],
            [False, True, True, False],
        ),
        (np.array(["🚀", "🔴", "hi🟡"]), ["🔴", "🟡", "🟢"], [False, True, False]),
        # Optimizations for 0 and 1 test_items
        (np.array(["hello", "world"]), ["hello"], [True, False]),
        (np.array(["hello", "world"]), [], [False, False]),
        # Note: this is a breaking change in the "typed array"
        # refactor, but follows NumPy semantics
        (np.asarray([np.nan, 1]), [np.nan], [False, False]),
        (np.asarray([np.nan, 1]), [np.nan, 1], [False, True]),
        (
            np.ma.MaskedArray([0, 1, 1], mask=[False, True, False]),
            [0, 1],
            [True, False, True],
        ),
        (
            np.ma.MaskedArray([np.nan, 1, 1], mask=[False, True, False]),
            [np.nan, 1],
            [False, False, True],
        ),
    ],
)
def test_isin(np_arr, test_items, desired):
    arr = ndx.asarray(np_arr)
    actual = nda.isin(arr, test_items)

    np.testing.assert_equal(actual.unwrap_numpy(), desired)


@pytest.mark.parametrize("scalar_type", [float, np.float16, np.float32, np.float64])
@pytest.mark.parametrize("values", [[np.nan], [np.nan, 1], [np.nan, 1, 2]])
def test_isin_numpy_floating_nan(scalar_type, values):
    array = np.asarray([np.nan, 1, 2, 3], dtype=np.float32)
    items = [scalar_type(value) for value in values]

    candidate = nda.isin(ndx.asarray(array), items)

    np.testing.assert_array_equal(
        candidate.unwrap_numpy(), np.isin(array, items), strict=True
    )


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
    expected = np.ma.masked_array([1, 2, 3], mask, dtype.unwrap_numpy())
    assert_array_equal(result.unwrap_numpy(), expected)


def test_is_integer_dtype_excludes_boolean():
    assert not nda.is_integer_dtype(ndx.bool)
