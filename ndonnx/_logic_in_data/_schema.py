# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import onnx
from spox import Var
from typing_extensions import Self

Json = dict[str, "Json"] | list["Json"] | str | int | float | bool | None
"""A JSON serializable object."""

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
"""Primitive type with an equivalent type in the ONNX standard."""

StructComponent = dict[str, "Schema"]
"""A composite type consisting of other ``StructComponents`` or ``PrimitiveTypes``."""

Components = dict[str, Var]
"""A flattened representation of the components of an array with arbitrary data type."""


@dataclass
class DTypeInfo:
    """Class returned by ``DType._info`` describing the respective data type."""

    defining_library: str
    # Version of this particular data type
    version: int
    dtype: str
    dtype_state: Json


@dataclass
class Schema:
    """Schema describing a data type.

    The names are suffixes of the names ultimately used in the model API.
    """

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
    return _get_schemas(metadict)


def _get_schemas(metadata: dict[str, str]) -> Schemas:
    """Get ``Schemas`` from metadata dict.

    This function is factored out from ``get_schemas`` for easier testing.
    """
    schema_dict = json.loads(metadata["ndonnx"])["schemas"]

    arguments = {k: Schema.from_json(v, 1) for k, v in schema_dict["arguments"].items()}
    results = {k: Schema.from_json(v, 1) for k, v in schema_dict["results"].items()}

    return Schemas(arguments=arguments, results=results)


def flatten_components(comps: dict[str, Components]) -> Components:
    return {
        f"{k}__{inner_k}": inner_v
        for k, inner in comps.items()
        for inner_k, inner_v in inner.items()
    }
