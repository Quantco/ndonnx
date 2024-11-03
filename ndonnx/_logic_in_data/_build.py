# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause


import onnx
from spox import Var
from spox import build as spox_build

from ._array import Array
from ._schema import SchemaV1


def build(arguments: dict[str, Array], results: dict[str, Array]) -> onnx.ModelProto:
    ins = _arrays_to_vars(arguments)
    outs = _arrays_to_vars(results)

    mp = spox_build(ins, outs, drop_unused_inputs=True)

    schema_v1 = {
        "ndonnx_schema": SchemaV1(
            input_schema={k: v.dtype._infov1 for k, v in arguments.items()},
            output_schema={k: v.dtype._infov1 for k, v in results.items()},
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
        components = v._data.disassemble()
        if isinstance(components, Var):
            out[k] = components
            continue
        for k_inner, v_inner in components.items():
            out[f"{k}{public_separator}{k_inner}"] = v_inner
    return out
