# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

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


@dataclass
class DTypeInfo:
    """Class returned by ``DType._info`` describing the respective data type."""

    defining_library: str
    # Version of this particular data type
    version: int
    dtype: str
    dtype_state: Json


@dataclass
class DTypeInfoV1:
    """Class returned by ``DType._info`` describing the respective data type."""

    author: str
    type_name: str
    meta: Json


@dataclass
class SchemaV1:
    input_schema: dict[str, DTypeInfoV1]
    output_schema: dict[str, DTypeInfoV1]
    version: Literal[1]
    input_prefix: str | None = None
    output_prefix: str | None = None

    @classmethod
    def parse_json(cls, s: str, /) -> SchemaV1:
        parsed = json.loads(s)

        return cls(
            input_schema={
                k: DTypeInfoV1(**v) for k, v in parsed["input_schema"].items()
            },
            output_schema={
                k: DTypeInfoV1(**v) for k, v in parsed["output_schema"].items()
            },
            version=1,
            input_prefix=parsed.get("input_prefix"),
            output_prefix=parsed.get("output_prefix"),
        )

    def to_json(self) -> str:
        def default(o):
            if isinstance(o, SchemaV1):
                return {k: v for k, v in vars(o).items() if v not in (None, "")}
            return vars(o)

        return json.dumps(self, default=default)
