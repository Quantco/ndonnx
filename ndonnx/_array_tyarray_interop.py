# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, overload

import numpy as np

import ndonnx as ndx
from ndonnx._typed_array.types import NpScalar

from ._array import Array

if TYPE_CHECKING:
    from ._typed_array import TyArrayBase


PY_SCALAR_NOT_FLOAT = TypeVar("PY_SCALAR_NOT_FLOAT", int, str, bool)


# np.float64 is a subclass of float (the np.generic where that is the
# case). This creates some issues here. where we can't reliably say
# that any float (subclass) will always be returned as a float
# (subclass).
@overload
def unwrap_tyarray(x: PY_SCALAR_NOT_FLOAT) -> PY_SCALAR_NOT_FLOAT: ...


@overload
def unwrap_tyarray(x: Array | NpScalar) -> TyArrayBase: ...


@overload
def unwrap_tyarray(x: float) -> TyArrayBase | float: ...


@overload
def unwrap_tyarray(x: None) -> None: ...


def unwrap_tyarray(
    x: Array | NpScalar | PY_SCALAR_NOT_FLOAT | float | None,
) -> TyArrayBase | PY_SCALAR_NOT_FLOAT | float | None:
    """Unwrap an ``Array`` in a union with Python scalars."""
    if isinstance(x, np.generic):
        return ndx.asarray(np.asarray(x))._tyarray
    if isinstance(x, Array):
        return x._tyarray
    return x
