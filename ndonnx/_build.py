# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

import onnx
from spox import Var
from spox import build as spox_build

from ._array import Array
from ._schema import SchemaV1


def build(
    inputs: dict[str, Array],
    outputs: dict[str, Array],
    drop_unused: bool = False,
    *,
    input_prefix: str = "",
    output_prefix: str = "",
) -> onnx.ModelProto:
    """Build and ONNX model from the provided argument-Arrays and outputs.

    Parameters
    ----------
    inputs
        Inputs of the model
    outputs
        Outputs of the model
    drop_unused
        Drop input arrays that are entirely unused from the model. Arrays
        consisting of multiple components are kept or dropped as a whole.
    input_prefix
        Prefix to prepend to all input names in the graph.
    output_prefix
        Prefix to prepend to all output names in the graph.

    Returns
    -------
    onnx.ModelProto
        ONNX model
    """
    inputs = {f"{input_prefix}{k}": v for k, v in inputs.items()}
    outputs = {f"{output_prefix}{k}": v for k, v in outputs.items()}

    ins = _arrays_to_vars(inputs)
    outs = _arrays_to_vars(outputs)

    if drop_unused:
        # Determine which components are used, but prune entire arrays
        # rather than individual components (e.g. ``_values``/``_null``).
        used = {
            inp.name
            for inp in spox_build(ins, outs, drop_unused_inputs=True).graph.input
        }
        inputs = {
            name: arr
            for name, arr in inputs.items()
            if any(n in used for n in _disassemble_named_array(name, arr))
        }
        ins = _arrays_to_vars(inputs)

    mp = spox_build(ins, outs, drop_unused_inputs=False)

    schema_v1 = {
        "ndonnx_schema": SchemaV1(
            input_schema={k: v.dtype.__ndx_infov1__ for k, v in inputs.items()},
            output_schema={k: v.dtype.__ndx_infov1__ for k, v in outputs.items()},
            version=1,
            input_prefix=input_prefix,
            output_prefix=output_prefix,
        ).to_json()
    }
    onnx.helper.set_model_props(mp, schema_v1)

    return mp


def _arrays_to_vars(dct_of_arrs: dict[str, Array]) -> dict[str, Var]:
    out: dict[str, Var] = {}
    for name, arr in dct_of_arrs.items():
        out |= _disassemble_named_array(name, arr)
    return out


def _disassemble_named_array(name: str, arr: Array) -> dict[str, Var]:
    # Take an Array, and create a map of its component parts prefixed with a name.
    # TODO: Use a different separator for the public name and the nested components?
    public_separator = "_"
    out: dict[str, Var] = {}
    components = arr._tyarray.disassemble()
    if isinstance(components, Var):
        out[name] = components
    else:
        for k, v in components.items():
            out[f"{name}{public_separator}{k}"] = v
    return out
