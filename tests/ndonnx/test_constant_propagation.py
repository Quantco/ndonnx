# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import numpy as np
import pytest

import ndonnx as ndx


def test_add():
    candidate = ndx.asarray([1]) + 1
    np.testing.assert_array_equal(candidate.to_numpy(), np.array([2]), strict=True)


def indexing_model(mode: Literal["lazy", "constant"]):
    if mode == "constant":
        a = ndx.asarray([0, 1, 2, 3])
    else:
        a = ndx.array(
            shape=("N",),
            dtype=ndx.int64,
        )
    b = ndx.array(shape=("M",), dtype=ndx.int64)
    return ndx.build(
        {"a": a, "b": b} if mode == "lazy" else {"b": b}, {"y": a[1:3] * 2}
    )


def arithmetic_model(mode: Literal["lazy", "constant"]):
    if mode == "constant":
        a = ndx.asarray([0, 1, 2, 3])
    else:
        a = ndx.array(
            shape=("N",),
            dtype=ndx.int64,
        )
    b = ndx.array(
        shape=("N",),
        dtype=ndx.int64,
    )
    return ndx.build({"a": a, "b": b} if mode == "lazy" else {"b": b}, {"y": a * 2 + b})


def dynamic_masking_model(mode: Literal["lazy", "constant"]):
    if mode == "constant":
        a = ndx.asarray([0, 1, 2, 3], dtype=ndx.int64)
    else:
        a = ndx.array(
            shape=("N",),
            dtype=ndx.int64,
        )
    b = ndx.array(
        shape=("M",),
        dtype=ndx.int64,
    )
    c = ndx.array(
        shape=("N",),
        dtype=ndx.bool,
    )
    inputs = {"b": b, "c": c}
    if mode == "lazy":
        inputs["a"] = a
    return ndx.build(inputs, {"y": a[c] * 2 + b})


def constant_indexing_model(mode: Literal["lazy", "constant"]):
    if mode == "constant":
        a = ndx.asarray([0, 1, 2, 3], dtype=ndx.int64)
    else:
        a = ndx.array(
            shape=("N",),
            dtype=ndx.int64,
        )
    b = ndx.asarray([5, 7, 8, 8, 9, 9, 234], dtype=ndx.int64)
    idx = ndx.asarray([1, 3, 5, 0])
    result = a * b[idx]
    return ndx.build({"a": a} if mode == "lazy" else {}, {"y": result})


@pytest.mark.parametrize(
    "model_func, expected_operators_constant",
    [
        (
            indexing_model,
            {"Identity", "Constant"},
        ),
        (
            arithmetic_model,
            {"Identity", "Add", "Constant"},
        ),
        (
            # We should be blocked from any real constant propagation
            # here since we are indexing on an eager value.
            dynamic_masking_model,
            {
                "Identity",
                "Constant",
                "Mul",
                "Add",
                "Compress",
            },
        ),
        (
            constant_indexing_model,
            {"Constant", "Identity"},
        ),
    ],
)
def test_model_constant_folds(model_func, expected_operators_constant):
    operators_used_const = {node.op_type for node in model_func("constant").graph.node}
    assert operators_used_const == expected_operators_constant


@pytest.mark.parametrize(
    "a, b, op, expected_operators",
    [
        (
            ndx.array(shape=("N",), dtype=ndx.bool),
            ndx.array(shape=("N",), dtype=ndx.bool),
            ndx.logical_or,
            {"Identity", "Or"},
        ),
        (
            ndx.asarray([False], dtype=ndx.bool),
            ndx.array(shape=("N",), dtype=ndx.bool),
            ndx.logical_or,
            {"Identity"},
        ),
        (
            ndx.asarray([False, False], dtype=ndx.bool),
            ndx.array(shape=("N",), dtype=ndx.bool),
            ndx.logical_or,
            {"Identity", "Or", "Constant"},
        ),
        (
            ndx.array(shape=("N",), dtype=ndx.bool),
            ndx.array(shape=("N",), dtype=ndx.bool),
            ndx.logical_and,
            {"Identity", "And"},
        ),
        (
            ndx.asarray([True], dtype=ndx.bool),
            ndx.array(shape=("N",), dtype=ndx.bool),
            ndx.logical_and,
            {"Identity"},
        ),
        (
            ndx.asarray([True, True], dtype=ndx.bool),
            ndx.array(shape=("N",), dtype=ndx.bool),
            ndx.logical_and,
            {"Identity", "And", "Constant"},
        ),
    ],
)
def test_logical_folding(a, b, op, expected_operators):
    out = op(a, b)
    inputs = {}
    if a.to_numpy() is None:
        inputs["a"] = a
    if b.to_numpy() is None:
        inputs["b"] = b
    model_proto = ndx.build(inputs, {"out": out})
    operators_used_const = {node.op_type for node in model_proto.graph.node}
    assert operators_used_const == expected_operators


@pytest.mark.parametrize(
    "cond, x, y, expected_operators",
    [
        (
            ndx.asarray([True, True]),
            ndx.asarray(["a", "b"]),
            ndx.asarray(["c", "d"]),
            {"Identity", "Constant"},
        ),
        (
            ndx.array(shape=("N",), dtype=ndx.bool),
            ndx.asarray(["a", "b"]),
            ndx.asarray(["c", "d"]),
            {"Identity", "Constant", "Where"},
        ),
        (
            ndx.asarray([False, False]),
            ndx.asarray(["a", "b"]),
            ndx.asarray(["c", "d"]),
            {"Identity", "Constant"},
        ),
        (
            ndx.asarray([True, False]),
            ndx.asarray(["a", "b"]),
            ndx.asarray(["a", "b"]),
            {"Identity", "Constant"},
        ),
        (
            ndx.asarray([True, False]),
            ndx.asarray(["a", "a"]),
            ndx.asarray(["a"]),
            {"Identity", "Constant"},
        ),
        (
            ndx.array(shape=("N",), dtype=ndx.bool),
            ndx.asarray([np.nan]),
            ndx.asarray([np.nan]),
            {"Identity", "Constant"},
        ),
        (
            ndx.array(shape=("N",), dtype=ndx.bool),
            ndx.asarray(["a", "a"]),
            ndx.asarray(["a"]),
            {"Identity", "Constant"},
        ),
        (
            ndx.array(shape=("N",), dtype=ndx.bool),
            ndx.asarray(np.ma.masked_array(["a", "a"], mask=[True, False])),
            ndx.asarray(["a"]),
            {"Identity", "Constant", "And", "Xor", "Not", "Where"},
        ),
        (
            ndx.array(shape=("N",), dtype=ndx.bool),
            ndx.asarray(np.ma.masked_array(["a", "a"], mask=[True, False])),
            ndx.asarray(np.ma.masked_array(["a", "a"], mask=[True, False])),
            {"Identity", "Constant", "And", "Xor", "Not", "Where"},
        ),
    ],
)
def test_where_folding(cond, x, y, expected_operators):
    out = ndx.where(cond, x, y)
    inputs = {}
    if cond.to_numpy() is None:
        inputs["cond"] = cond
    if x.to_numpy() is None:
        inputs["x"] = x
    if y.to_numpy() is None:
        inputs["y"] = y
    model_proto = ndx.build(inputs, {"out": out})
    operators_used_const = {node.op_type for node in model_proto.graph.node}
    assert operators_used_const == expected_operators
