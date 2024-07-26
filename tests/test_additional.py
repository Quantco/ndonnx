# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import sys

import numpy as np
import pytest

import ndonnx as ndx
import ndonnx.additional as nda

from .utils import run


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


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="ORT 1.18 not registering LabelEncoder(4) only on Windows.",
)
def test_static_map():
    a = ndx.array(shape=(3,), dtype=ndx.int64)
    b = nda.static_map(a, {1: 2, 2: 3})

    model = ndx.build({"a": a}, {"b": b})
    np.testing.assert_equal([0, 2, 3], run(model, {"a": np.array([0, 1, 2])})["b"])

    # nans are mapped by static_map
    a = ndx.array(shape=("N",), dtype=ndx.float64)
    b = nda.static_map(a, {1.0: 2, np.nan: 3, 3.0: 42}, default=-1)

    model = ndx.build({"a": a}, {"b": b})
    np.testing.assert_equal(
        [-1, -1, 42, 3],
        run(model, {"a": np.array([0.0, 2.0, 3.0, np.nan])})["b"],
    )


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="ORT 1.18 not registering LabelEncoder(4) only on Windows.",
)
def test_isin():
    a = ndx.array(shape=("N",), dtype=ndx.utf8)
    b = nda.isin(a, ["foo", "bar", "baz"])

    model = ndx.build({"a": a}, {"b": b})
    np.testing.assert_equal(
        [False, True, True, False],
        run(model, dict(a=np.array(["hello", "foo", "baz", "!"])))["b"],
    )

    a = ndx.array(shape=("N",), dtype=ndx.utf8)
    b = nda.isin(a, ["🔴", "🟡", "🟢"])

    model = ndx.build({"a": a}, {"b": b})
    np.testing.assert_equal(
        [False, True, False],
        run(model, dict(a=np.array(["🚀", "🔴", "hi🟡"])))["b"],
    )

    a = ndx.asarray(["hello", "world"])
    np.testing.assert_equal([True, False], nda.isin(a, ["hello"]).to_numpy())
