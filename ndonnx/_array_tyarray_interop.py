# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, overload

from ._array import Array

if TYPE_CHECKING:
    from ._typed_array import TyArrayBase


PY_SCALAR = TypeVar("PY_SCALAR", int, float, str, bool)


@overload
def unwrap_tyarray(x: Array) -> TyArrayBase: ...


@overload
def unwrap_tyarray(x: PY_SCALAR) -> PY_SCALAR: ...


@overload
def unwrap_tyarray(x: None) -> None: ...


def unwrap_tyarray(x: Array | PY_SCALAR | None) -> TyArrayBase | PY_SCALAR | None:
    """Unwrap an ``Array`` in a union with Python scalars."""
    if isinstance(x, Array):
        return x._tyarray
    return x
