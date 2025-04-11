# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
import json
from pathlib import Path

import pytest

import ndonnx as ndx
import ndonnx._typed_array as tydx


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
