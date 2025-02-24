# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from types import EllipsisType
from typing import TYPE_CHECKING, TypeAlias, Union

if TYPE_CHECKING:
    from ._array import Array


StrictShape = tuple[int, ...]
StandardShape = int | tuple[int | None, ...]
OnnxShape = tuple[int | str | None, ...]

GetitemIndex: TypeAlias = Union[
    int | slice | EllipsisType | None,
    tuple[int | slice | EllipsisType | None, ...],
    "Array",
]

SetitemIndex: TypeAlias = Union[
    int | slice | EllipsisType, tuple[int | slice | EllipsisType, ...], "Array"
]

PyScalar = bool | int | float | str
NestedSequence = Sequence["PyScalar | NestedSequence"]
