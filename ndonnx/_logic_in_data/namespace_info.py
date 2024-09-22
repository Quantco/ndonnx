# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from typing import TypedDict

from . import dtypes

DefaultDataTypes = TypedDict(
    "DefaultDataTypes",
    {
        "real floating": dtypes.DType,
        "complex floating": dtypes.DType,
        "integral": dtypes.DType,
        "indexing": dtypes.DType,
    },
)


class DataTypes(TypedDict, total=False):
    bool: dtypes.DType
    float32: dtypes.DType
    float64: dtypes.DType
    int8: dtypes.DType
    int16: dtypes.DType
    int32: dtypes.DType
    int64: dtypes.DType
    uint8: dtypes.DType
    uint16: dtypes.DType
    uint32: dtypes.DType
    uint64: dtypes.DType


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

    def default_device(self) -> None:
        raise ValueError("ndonnx does not define a default device")
        ...

    def default_dtypes(self, *, device: None) -> DefaultDataTypes:
        # TODO: We are not standard compliant until we support complex numbers
        return {  # type: ignore
            "real floating": dtypes.default_float,
            # "complex floating": dtypes.complex128,
            "integral": dtypes.default_int,
            "indexing": dtypes.default_int,
        }

    def devices(self) -> list[None]:
        raise ValueError("ndonnx does not define devices")

    def dtypes(self, *, device: None, kind: None | str | tuple[str, ...]) -> DataTypes:
        return {
            "bool": dtypes.bool_,
            "float32": dtypes.float32,
            "float64": dtypes.float64,
            # "complex64": dtypes.DType,
            # "complex128": dtypes.DType,
            "int8": dtypes.int8,
            "int16": dtypes.int16,
            "int32": dtypes.int32,
            "int64": dtypes.int64,
            "uint8": dtypes.uint8,
            "uint16": dtypes.uint16,
            "uint32": dtypes.uint32,
            "uint64": dtypes.uint64,
        }


def __array_namespace_info__() -> Info:
    return Info()


__all__ = ["__array_namespace_info__"]
