# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TypedDict

from ._dtypes import DType
from ._typed_array import onnx

DefaultDataTypes = TypedDict(
    "DefaultDataTypes",
    {
        "real floating": DType,
        "complex floating": DType,
        "integral": DType,
        "indexing": DType,
    },
)


class DataTypes(TypedDict, total=False):
    bool: DType
    float32: DType
    float64: DType
    int8: DType
    int16: DType
    int32: DType
    int64: DType
    uint8: DType
    uint16: DType
    uint32: DType
    uint64: DType


Capabilities = TypedDict(
    "Capabilities",
    {
        "boolean indexing": bool,
        "data-dependent shapes": bool,
        "max rank": None | int,
    },
)


class Info:
    """Namespace returned by `__array_namespace_info__`."""

    def capabilities(self) -> Capabilities:
        return {
            "boolean indexing": True,
            "data-dependent shapes": True,
            "max rank": None,
        }

    def default_device(self) -> Device:
        return device

    def default_dtypes(self, *, device: None | Device = None) -> DefaultDataTypes:
        # TODO: We are not standard compliant until we support complex numbers
        return {  # type: ignore
            "real floating": onnx.float64,
            # "complex floating": onnx.complex128,
            "integral": onnx.int64,
            "indexing": onnx.int64,
        }

    def devices(self) -> list[Device]:
        return [device]

    def dtypes(
        self, *, device: None | Device = None, kind: None | str | tuple[str, ...]
    ) -> DataTypes:
        return {
            "bool": onnx.bool_,
            "float32": onnx.float32,
            "float64": onnx.float64,
            # "complex64": DType,
            # "complex128": DType,
            "int8": onnx.int8,
            "int16": onnx.int16,
            "int32": onnx.int32,
            "int64": onnx.int64,
            "uint8": onnx.uint8,
            "uint16": onnx.uint16,
            "uint32": onnx.uint32,
            "uint64": onnx.uint64,
        }


class Device: ...


device = Device()


def __array_namespace_info__() -> Info:
    return Info()


__all__ = ["__array_namespace_info__", "device"]
