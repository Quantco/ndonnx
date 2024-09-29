# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import onnx

if TYPE_CHECKING:
    from spox import Var
    from typing_extensions import Self

    Json = dict[str, "Json"] | list["Json"] | str | int | float | bool | None
    PrimitiveComponent = Literal[
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bool",
        "str",
    ]
    StructComponent = dict[str, "Schema"]
    Components = dict[str, Var]


@dataclass
class DTypeInfo:
    defining_library: str
    version: int
    dtype: str
    dtype_state: Json


@dataclass
class Schema:
    dtype_info: DTypeInfo
    components: PrimitiveComponent | StructComponent

    @classmethod
    def from_json(cls, val: dict[str, Any], ndonnx_schema_version: int) -> Self:
        if ndonnx_schema_version != 1:
            raise ValueError(f"unsupported schema version `{ndonnx_schema_version}`")
        info = DTypeInfo(**val["dtype_info"])
        if isinstance(val["components"], dict):
            comps = {
                k: Schema.from_json(v, ndonnx_schema_version)
                for k, v in val["components"].items()
            }
        else:
            comps = val["components"]

        return cls(dtype_info=info, components=comps)


class ModelSchema:
    inputs: dict[str, Schema]
    outputs: dict[str, Schema]


def var_to_primitive(var: Var) -> PrimitiveComponent:
    dtype = var.unwrap_tensor().dtype
    if dtype == np.int8:
        return "int8"
    if dtype == np.int16:
        return "int16"
    if dtype == np.int32:
        return "int32"
    if dtype == np.int64:
        return "int64"

    if dtype == np.uint8:
        return "uint8"
    if dtype == np.uint16:
        return "uint16"
    if dtype == np.uint32:
        return "uint32"
    if dtype == np.uint64:
        return "uint64"

    if dtype == np.float16:
        return "float16"
    if dtype == np.float32:
        return "float32"
    if dtype == np.float64:
        return "float64"

    if dtype == np.str_:
        return "str"
    if dtype == np.bool:
        return "bool"

    raise ValueError(f"unexpected data type of 'var': `{dtype}`")


@dataclass
class Schemas:
    arguments: dict[str, Schema]
    results: dict[str, Schema]


def get_schemas(mp: onnx.ModelProto) -> Schemas:
    metadict = {el.key: el.value for el in mp.metadata_props}
    schema_dict = json.loads(metadict["schemas"])

    arguments = {k: Schema.from_json(v, 1) for k, v in schema_dict["arguments"].items()}
    results = {k: Schema.from_json(v, 1) for k, v in schema_dict["results"].items()}

    return Schemas(arguments=arguments, results=results)


def flatten_components(comps: dict[str, Components]) -> Components:
    return {
        f"{k}__{inner_k}": inner_v
        for k, inner in comps.items()
        for inner_k, inner_v in inner.items()
    }
