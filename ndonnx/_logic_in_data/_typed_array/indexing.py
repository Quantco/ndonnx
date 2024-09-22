# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from types import EllipsisType
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .core import TyArrayInteger


GetitemIndexStatic = Union[
    int | slice | EllipsisType | None, tuple[int | slice | EllipsisType | None, ...]
]
GetitemIndex = Union[GetitemIndexStatic, "TyArrayInteger"]

SetitemIndexStatic = Union[
    int | slice | EllipsisType, tuple[int | slice | EllipsisType, ...]
]
SetitemIndex = Union[SetitemIndexStatic, "TyArrayInteger"]
