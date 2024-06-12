# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import ndonnx as ndx


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
    a = ndx.array(shape=("N",), dtype=ndx._data_types.promote_nullable(dtype))
    model_proto = ndx.build({"input": a}, {"output": a})
    assert [node.name for node in model_proto.graph.input] == [
        "input_values",
        "input_null",
    ]
    assert [node.name for node in model_proto.graph.output] == [
        "output_values",
        "output_null",
    ]
