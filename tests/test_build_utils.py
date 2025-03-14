# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import json
from pathlib import Path

import pytest

import ndonnx as ndx
from ndonnx import _data_types as dtypes


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
def test_input_output_name_backwards_compatibility(dtype):
    a = ndx.array(shape=("N",), dtype=dtype)
    model_proto = ndx.build({"input": a}, {"output": a})
    assert [node.name for node in model_proto.graph.input] == ["input"]
    assert [node.name for node in model_proto.graph.output] == ["output"]
    a = ndx.array(shape=("N",), dtype=ndx._data_types.into_nullable(dtype))
    model_proto = ndx.build({"input": a}, {"output": a})
    assert [node.name for node in model_proto.graph.input] == [
        "input_values",
        "input_null",
    ]
    assert [node.name for node in model_proto.graph.output] == [
        "output_values",
        "output_null",
    ]


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
    a = ndx.array(shape=("N",), dtype=dtype)
    model_proto = ndx.build({"input": a}, {"output": a})
    assert [node.name for node in model_proto.graph.input] == ["input"]
    assert [node.name for node in model_proto.graph.output] == ["output"]
    a = ndx.array(shape=("N",), dtype=dtypes.into_nullable(dtype))
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

    a = ndx.array(shape=("N",), dtype=dtype)
    b = a[0]  # make the build non-trivial

    mp = ndx.build({"a": a}, {"b": b})

    # These files should not be update automatically!
    # File names follow NumPy's __str__ semantics for primitive data
    # types and generally uses lower case. ndonnx does not have the
    # same __str__ semantics today.
    if dtype == ndx.bool:
        dtype = "bool"
    elif dtype == ndx.nbool:
        dtype = "nbool"
    else:
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
