# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json

import numpy as np
import onnx
import onnxruntime as ort

import ndonnx._data_types as dtypes
from ndonnx._build import (
    _assemble_outputs,
    _deconstruct_inputs,
    _extract_output_names,
    _get_dtype,
)

Dtype = dtypes.StructType | dtypes.CoreType


def run(
    model_proto: onnx.ModelProto, inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    # Extract input and output schemas
    input_dtypes, output_dtypes = _get_dtypes(model_proto)
    session = _make_session(model_proto)
    flattened_inputs = _deconstruct_inputs(inputs, input_dtypes)
    output_names = tuple(_extract_output_names(output_dtypes))
    outputs = dict(zip(output_names, session.run(output_names, flattened_inputs)))
    return _assemble_outputs(outputs, output_dtypes)


def _make_session(model_proto: onnx.ModelProto) -> ort.InferenceSession:
    session_options = ort.SessionOptions()
    return ort.InferenceSession(
        model_proto.SerializeToString(), sess_options=session_options
    )


def _get_dtypes(
    model_proto: onnx.ModelProto,
) -> tuple[dict[str, Dtype], dict[str, Dtype]]:
    encoding = json.loads(
        [prop for prop in model_proto.metadata_props if prop.key == "ndonnx_schema"][
            0
        ].value
    )
    version = encoding["version"]
    input_schema = {}
    for name, schema in encoding["input_schema"].items():
        if schema["author"] != "ndonnx":
            raise ValueError(f"Unknown author {schema}")
        input_schema[name] = _get_dtype(schema["type_name"], version)

    output_schema = {}
    for name, schema in encoding["output_schema"].items():
        if schema["author"] != "ndonnx":
            raise ValueError(f"Unknown author {schema}")
        output_schema[name] = _get_dtype(schema["type_name"], version)

    return input_schema, output_schema
