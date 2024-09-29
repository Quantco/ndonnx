# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import json

import onnx
from spox import Var
from spox import build as spox_build

from ._array import Array
from ._schema import flatten_components


def build(arguments: dict[str, Array], results: dict[str, Array]) -> onnx.ModelProto:
    ins = _arrays_to_vars(arguments)
    outs = _arrays_to_vars(results)

    mp = spox_build(ins, outs, drop_unused_inputs=True)

    metadata = {
        "schemas": json.dumps(
            {
                "arguments": json.loads(_json_schema(arguments)),
                "results": json.loads(_json_schema(results)),
            }
        ),
        "ndonnx_schema_version": "1",
    }
    onnx.helper.set_model_props(mp, metadata)

    return mp


def _arrays_to_vars(dct_of_arrs: dict[str, Array]) -> dict[str, Var]:
    return flatten_components(
        {k: v._data.disassemble()[0] for k, v in dct_of_arrs.items()}
    )


def _json_schema(dct_of_arrs: dict[str, Array]) -> str:
    return json.dumps(
        {k: v._data.disassemble()[1] for k, v in dct_of_arrs.items()}, default=vars
    )
