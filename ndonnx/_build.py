# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
import json

import numpy as np
import onnx
import spox

import ndonnx as ndx
import ndonnx._data_types as dtypes

from ._array import Array
from ._corearray import _CoreArray


def build(
    inputs: dict[str, Array],
    outputs: dict[str, Array],
) -> onnx.ModelProto:
    """Build utility to produce an ONNX model. Requires a dictionary of named input and
    output arrays.

    Parameters
    ----------
    inputs: dict[str, Array]
        Inputs of the model
    outputs: dict[str, Array]
        Outputs of the model

    Returns
    -------
    out: onnx.ModelProto
        ONNX model
    """

    def collect_vars(name: str, arr: _CoreArray | Array):
        if isinstance(arr, _CoreArray):
            return {name: arr.var}
        elif isinstance(arr.dtype, ndx.CoreType):
            return {name: arr.spox_var()}
        else:
            constituent_vars: dict[str, spox.Var] = {}
            for field_name, field in arr._fields.items():
                constituent_vars |= collect_vars(f"{name}_{field_name}", field)
            return constituent_vars

    input_vars: dict[str, spox.Var] = functools.reduce(
        lambda acc, arr: acc | collect_vars(*arr), inputs.items(), {}
    )
    output_vars: dict[str, spox.Var] = functools.reduce(
        lambda acc, arr: acc | collect_vars(*arr), outputs.items(), {}
    )
    model_proto = spox.build(input_vars, output_vars)
    input_schema = {
        name: dataclasses.asdict(input.dtype._schema())
        for name, input in inputs.items()
    }
    output_schema = {
        name: dataclasses.asdict(output.dtype._schema())
        for name, output in outputs.items()
    }
    model_proto.metadata_props.append(
        onnx.StringStringEntryProto(
            key="ndonnx_schema",
            value=json.dumps(
                {
                    "version": 1,
                    "input_schema": input_schema,
                    "output_schema": output_schema,
                }
            ),
        )
    )
    return model_proto


def _get_dtype(dtype: str, version: int) -> dtypes.StructType | dtypes.CoreType:
    """Get the ndonnx data type given a string representation and version number, as
    found in the produced `onnx.ModelProto`."""
    if version == 1:
        return _v1_dtypes[dtype]
    else:
        raise ValueError(f"Unknown schema version {version}")


def _extract_output_names(
    output_dtypes: dict[str, dtypes.StructType | dtypes.CoreType],
) -> dict[str, ndx.CoreType]:
    """Given a dictionary mapping output names to their data types, extract the
    underlying fully qualified names and their CoreTypes.

    Parameters
    ----------
    output_dtypes: dict[str, ndonnx.StructType | ndonnx.CoreType]
        Dictionary mapping output names to their data types

    Returns
    -------
    out: dict[str, ndonnx.CoreType]
        Dictionary mapping fully qualified output names to their CoreTypes
    """

    def extract_output_name(name, type):
        if isinstance(type, ndx.CoreType):
            return {name: type}
        else:
            names: dict[str, ndx.CoreType] = {}
            for field_name, field_type in type._fields().items():
                names |= extract_output_name(f"{name}_{field_name}", field_type)
            return names

    return functools.reduce(
        lambda acc, arr: acc | extract_output_name(*arr), output_dtypes.items(), {}
    )


def _flatten(input_dict, dtype: dtypes.CoreType | dtypes.StructType, field_name: str):
    if isinstance(dtype, ndx.CoreType):
        return {field_name: input_dict["data"]}
    else:
        flattened: dict[str, np.ndarray] = {}
        for field, field_type in dtype._fields().items():
            flattened |= _flatten(
                input_dict[field], field_type, f"{field_name}_{field}"
            )
        return flattened


def _deconstruct_inputs(
    inputs: dict[str, np.ndarray],
    input_schema: dict[str, dtypes.CoreType | dtypes.StructType],
) -> dict[str, np.ndarray]:
    """Given a dictionary of named inputs and a schema mapping these input names to
    their ndonnx data types, return the disassembled inputs as valid inputs for
    inference.

    Parameters
    ----------
    inputs: dict[str, np.ndarray]
        Dictionary of named inputs
    input_schema: dict[str, ndonnx.CoreType | ndonnx.StructType]
        Schema mapping input names to their ndonnx data types

    Returns
    -------
    out: dict[str, np.ndarray]
        Disassembled inputs with fully qualified names of core constituent fields of the inputs, based on the data type of the input in `input_schema`, based on the data type's `parse_input` method.
    """

    def parse_input(name, type):
        structured_fields = type._parse_input(inputs[name])
        flattened_fields = _flatten(structured_fields, input_schema[name], name)
        return flattened_fields

    return functools.reduce(
        lambda acc, arr: acc | parse_input(*arr), input_schema.items(), {}
    )


def _assemble_outputs(
    output_data: dict[str, np.ndarray],
    output_schema: dict[str, dtypes.CoreType | dtypes.StructType],
) -> dict[str, np.ndarray]:
    """Given a dictionary of named outputs and a schema mapping these output names to
    their ndonnx data types, return the outputs assembled into values that reflect their
    data type.

    Parameters
    ----------
    output_data: dict[str, np.ndarray]
        Dictionary of named outputs from inference
    output_schema: dict[str, ndonnx.CoreType | ndonnx.StructType]
        Schema mapping output names to their ndonnx data types

    Returns
    -------
    out: dict[str, np.ndarray]
        Assembled outputs with data type that is meaningful based on the `output_schema` entry, based on the data type's `assemble_output` method.
    """

    def _assemble_output(name, type):
        # We want to recurse down the type here and recursively reassemble the output
        def helper(cur_type, prefix):
            if isinstance(cur_type, ndx.CoreType):
                return cur_type._assemble_output({"data": output_data[prefix]})
            elif isinstance(cur_type, dtypes.StructType):
                inputs = {
                    field: helper(field_type, f"{prefix}_{field}")
                    for field, field_type in cur_type._fields().items()
                }
                return cur_type._assemble_output(inputs)
            else:
                raise TypeError

        return {name: helper(type, name)}

    return functools.reduce(
        lambda acc, arr: acc | _assemble_output(*arr), output_schema.items(), {}
    )


_v1_dtypes: dict[str, dtypes.CoreType | dtypes.StructType] = {
    "UInt8": dtypes.uint8,
    "UInt16": dtypes.uint16,
    "UInt32": dtypes.uint32,
    "UInt64": dtypes.uint64,
    "Int8": dtypes.int8,
    "Int16": dtypes.int16,
    "Int32": dtypes.int32,
    "Int64": dtypes.int64,
    "Float32": dtypes.float32,
    "Float64": dtypes.float64,
    "Utf8": dtypes.utf8,
    "Boolean": dtypes.bool,
    "NUInt8": dtypes.nuint8,
    "NUInt16": dtypes.nuint16,
    "NUInt32": dtypes.nuint32,
    "NUInt64": dtypes.nuint64,
    "NInt8": dtypes.nint8,
    "NInt16": dtypes.nint16,
    "NInt32": dtypes.nint32,
    "NInt64": dtypes.nint64,
    "NFloat32": dtypes.nfloat32,
    "NFloat64": dtypes.nfloat64,
    "NUtf8": dtypes.nutf8,
    "NBoolean": dtypes.nbool,
}
