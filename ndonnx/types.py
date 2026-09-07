# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from types import EllipsisType
from typing import TYPE_CHECKING, Literal, TypeAlias, Union

import numpy as np

if TYPE_CHECKING:
    from ._array import Array


StrictShape = tuple[int, ...]
OnnxShape = tuple[int | str | None, ...]

GetItemKey: TypeAlias = Union[
    int | slice | EllipsisType | None,
    tuple[int | slice | EllipsisType | None, ...],
    "Array",
]

SetitemKey: TypeAlias = Union[
    int | slice | EllipsisType, tuple[int | slice | EllipsisType, ...], "Array"
]

PyScalar = bool | int | float | str
# Use concrete classes so this alias remains valid in isinstance. The longlong
# classes can be distinct from the fixed-width aliases on supported platforms.
NumpyScalar = (
    np.bool_
    | np.int8
    | np.int16
    | np.int32
    | np.int64
    | np.longlong
    | np.uint8
    | np.uint16
    | np.uint32
    | np.uint64
    | np.ulonglong
    | np.float16
    | np.float32
    | np.float64
    | np.str_
)
Scalar = PyScalar | NumpyScalar
NestedSequence = Sequence["Scalar | NestedSequence"]

DTypeAlias: TypeAlias = (
    (type[bool] | type[int] | type[float])
    | Literal["bool", "int", "float"]
    | Literal["int8", "int16", "int32", "int64"]
    | Literal["uint8", "uint16", "uint32", "uint64"]
    | Literal["float16", "float32", "float64"]
)

__all__ = [
    "StrictShape",
    "OnnxShape",
    "GetItemKey",
    "SetitemKey",
    "PyScalar",
    "NestedSequence",
]
