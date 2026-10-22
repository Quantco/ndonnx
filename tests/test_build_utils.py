# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause
import json
from pathlib import Path

import numpy as np
import pytest
from onnx import ModelProto

import ndonnx as ndx
import ndonnx._typed_array as tydx
from ndonnx.extensions import fill_null, get_mask

from .utils import run


@pytest.mark.parametrize(
    "dtype",
    [
        ndx.int8,
        ndx.int16,
        ndx.int32,
        ndx.int64,
        ndx.float32,
        ndx.float64,
        ndx.utf8,
        ndx.bool,
        ndx.uint8,
        ndx.uint16,
        ndx.uint32,
        ndx.uint64,
    ],
)
def test_no_namemangling_for_standard_types(dtype):
    a = ndx.argument(shape=("N",), dtype=dtype)
    model_proto = ndx.build({"input": a}, {"output": a})
    assert [node.name for node in model_proto.graph.input] == ["input"]
    assert [node.name for node in model_proto.graph.output] == ["output"]
    a = ndx.argument(shape=("N",), dtype=tydx.masked_onnx.to_nullable_dtype(dtype))
    model_proto = ndx.build({"input": a}, {"output": a})
    assert {node.name for node in model_proto.graph.input} == {
        "input_values",
        "input_null",
    }
    assert {node.name for node in model_proto.graph.output} == {
        "output_values",
        "output_null",
    }


@pytest.mark.parametrize(
    "dtype",
    [
        ndx.bool,
        ndx.float32,
        ndx.float64,
        ndx.int16,
        ndx.int32,
        ndx.int64,
        ndx.int8,
        ndx.nbool,
        ndx.nfloat32,
        ndx.nfloat64,
        ndx.nint16,
        ndx.nint32,
        ndx.nint64,
        ndx.nint8,
        ndx.nuint16,
        ndx.nuint32,
        ndx.nuint64,
        ndx.nuint8,
        ndx.nutf8,
        ndx.uint16,
        ndx.uint32,
        ndx.uint64,
        ndx.uint8,
        ndx.utf8,
    ],
)
def test_schema_against_snapshots(dtype, update_schema_snapshots):
    # We must not break backwards compatibility. We test every type we
    # support that it keeps producing the same schema.

    a = ndx.argument(shape=("N",), dtype=dtype)
    b = a[0]  # make the build non-trivial

    mp = ndx.build({"a": a}, {"b": b})

    dtype = str(dtype).lower()
    fname = Path(__file__).parent / f"schemas/{dtype}.json"

    if update_schema_snapshots:
        # Ensure a stable order
        metadata = sorted(mp.metadata_props, key=lambda el: el.key)
        with open(fname, "w+") as f:
            json.dump(
                json.loads({el.key: el.value for el in metadata}["ndonnx_schema"]),
                f,
                indent=4,
            )
            f.write("\n")  # Avoid pre-commit complaint about missing new lines

    with open(fname) as f:
        expected_schemas = json.load(f)
    candidate_schemas = json.loads(
        {el.key: el.value for el in mp.metadata_props}["ndonnx_schema"]
    )

    assert expected_schemas == candidate_schemas

    # test json round trip of schema data
    assert candidate_schemas["input_schema"]["a"] == a.dtype.__ndx_infov1__.__dict__
    assert candidate_schemas["output_schema"]["b"] == b.dtype.__ndx_infov1__.__dict__


def _get_input_schema(model_proto: ModelProto):
    for entry in model_proto.metadata_props:
        if entry.key == "ndonnx_schema":
            return json.loads(entry.value)["input_schema"]
    raise KeyError("'ndonnx_schema' not present")


def test_unused_inputs_not_in_schema():
    # Case 1: simple graph with an unused input.
    a = ndx.argument(shape=("N",), dtype=ndx.float64)
    b = ndx.argument(shape=("N",), dtype=ndx.float64)
    unused = ndx.argument(shape=("N",), dtype=ndx.float64)

    c = a + b

    no_drop = ndx.build(
        inputs={"a": a, "b": b, "unused": unused}, outputs={"c": c}, drop_unused=False
    )

    assert "unused" in _get_input_schema(no_drop)

    drop = ndx.build(
        inputs={"a": a, "b": b, "unused": unused}, outputs={"c": c}, drop_unused=True
    )
    assert "unused" not in _get_input_schema(drop)

    # Case 2: part of a nullable input is not used.
    a = ndx.argument(shape=("N",), dtype=ndx.nint64)

    b = ndx.isnan(a)

    drop = ndx.build(inputs={"a": a}, outputs={"b": b}, drop_unused=True)

    assert "a" in _get_input_schema(drop)


_NULLABLE_SAMPLE_VALUES = {
    ndx.nint64: np.array([1, 2], np.int64),
    ndx.nfloat64: np.array([1.0, 2.0], np.float64),
    ndx.nbool: np.array([True, False]),
    ndx.nutf8: np.array(["a", "b"], np.str_),
}


@pytest.mark.parametrize("dtype", list(_NULLABLE_SAMPLE_VALUES))
@pytest.mark.parametrize("prefix", ["", "in_"])
def test_drop_unused_keeps_all_components_if_only_mask_is_used(dtype, prefix):
    a = ndx.argument(shape=("N",), dtype=dtype)
    mask = get_mask(a)
    assert mask is not None
    y = ndx.logical_not(mask)

    mp = ndx.build({"a": a}, {"y": y}, drop_unused=True, input_prefix=prefix)

    assert [i.name for i in mp.graph.input] == [f"{prefix}a_values", f"{prefix}a_null"]
    assert list(_get_input_schema(mp)) == [f"{prefix}a"]

    (res,) = run(
        mp,
        {
            f"{prefix}a_values": _NULLABLE_SAMPLE_VALUES[dtype],
            f"{prefix}a_null": np.array([False, True]),
        },
    ).values()
    np.testing.assert_array_equal(res, [True, False])


def test_drop_unused_keeps_all_components_if_only_values_are_used():
    a = ndx.argument(shape=("N",), dtype=ndx.nint64)
    y = fill_null(a, 0)

    mp = ndx.build({"a": a}, {"y": y}, drop_unused=True)

    assert [i.name for i in mp.graph.input] == ["a_values", "a_null"]
    assert list(_get_input_schema(mp)) == ["a"]


def test_drop_unused_drops_all_components_of_unused_array():
    a = ndx.argument(shape=("N",), dtype=ndx.float64)
    u = ndx.argument(shape=("N",), dtype=ndx.nint64)

    mp = ndx.build({"a": a, "u": u}, {"y": a + 1}, drop_unused=True)

    assert [i.name for i in mp.graph.input] == ["a"]
    assert list(_get_input_schema(mp)) == ["a"]


def test_drop_unused_preserves_input_order():
    a = ndx.argument(shape=("N",), dtype=ndx.nint64)
    b = ndx.argument(shape=("N",), dtype=ndx.float64)
    c = ndx.argument(shape=("N",), dtype=ndx.nint64)
    d = ndx.argument(shape=("N",), dtype=ndx.int64)
    mask = get_mask(a)
    assert mask is not None

    # Use inputs in reverse order of their declaration
    y = ndx.where(mask, d, b)

    mp = ndx.build({"a": a, "b": b, "c": c, "d": d}, {"y": y}, drop_unused=True)

    assert [i.name for i in mp.graph.input] == ["a_values", "a_null", "b", "d"]
    assert list(_get_input_schema(mp)) == ["a", "b", "d"]


def test_build_with_prefixes():
    a = ndx.argument(shape=("N",), dtype=ndx.float64)
    b = ndx.argument(shape=("N",), dtype=ndx.float64)
    c = a + b

    mp = ndx.build(
        inputs={"a": a, "b": b},
        outputs={"c": c},
        input_prefix="in_",
        output_prefix="out_",
    )

    # Check graph inputs and outputs names
    assert {node.name for node in mp.graph.input} == {"in_a", "in_b"}
    assert [node.name for node in mp.graph.output] == ["out_c"]

    # Check schema stored in metadata
    schema = None
    for entry in mp.metadata_props:
        if entry.key == "ndonnx_schema":
            schema = json.loads(entry.value)
            break
    assert schema is not None
    assert schema["input_prefix"] == "in_"
    assert schema["output_prefix"] == "out_"
    assert "in_a" in schema["input_schema"]
    assert "in_b" in schema["input_schema"]
    assert "out_c" in schema["output_schema"]
