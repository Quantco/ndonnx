# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from types import EllipsisType
from typing import TYPE_CHECKING, Literal, TypeAlias, Union

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
NestedSequence = Sequence["PyScalar | NestedSequence"]

DTypeAlias: TypeAlias = (
    type[bool]
    | type[int]
    | type[float]
    | Literal["bool"]
    | Literal["int"]
    | Literal["float"]
)


__all__ = [
    "StrictShape",
    "OnnxShape",
    "GetItemKey",
    "SetitemKey",
    "PyScalar",
    "NestedSequence",
]
