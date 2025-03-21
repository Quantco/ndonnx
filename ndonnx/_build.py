# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause


import onnx
from spox import Var
from spox import build as spox_build

from ._array import Array
from ._schema import SchemaV1


def build(
    inputs: dict[str, Array], outputs: dict[str, Array], drop_unused=False
) -> onnx.ModelProto:
    """Build and ONNX model from the provided argument-Arrays and outputs.

    Parameters
    ----------
    inputs
        Inputs of the model
    outputs
        Outputs of the model

    Returns
    -------
    onnx.ModelProto
        ONNX model
    """
    ins = _arrays_to_vars(inputs)
    outs = _arrays_to_vars(outputs)

    mp = spox_build(ins, outs, drop_unused_inputs=drop_unused)

    schema_v1 = {
        "ndonnx_schema": SchemaV1(
            input_schema={k: v.dtype.__ndx_infov1__ for k, v in inputs.items()},
            output_schema={k: v.dtype.__ndx_infov1__ for k, v in outputs.items()},
            version=1,
        ).to_json()
    }
    onnx.helper.set_model_props(mp, schema_v1)

    return mp


def _arrays_to_vars(dct_of_arrs: dict[str, Array]) -> dict[str, Var]:
    # TODO: Use a different separator for the public name and the nested components?
    public_separator = "_"
    out = {}
    for k, v in dct_of_arrs.items():
        components = v._tyarray.disassemble()
        if isinstance(components, Var):
            out[k] = components
            continue
        for k_inner, v_inner in components.items():
            out[f"{k}{public_separator}{k_inner}"] = v_inner
    return out
