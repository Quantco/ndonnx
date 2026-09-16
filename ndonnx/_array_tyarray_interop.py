# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, overload

from ._array import Array
from .types import NumpyScalar, PyScalar, Scalar

if TYPE_CHECKING:
    from ._typed_array import TyArrayBase


PY_SCALAR = TypeVar("PY_SCALAR", int, float, str, bool)


@overload
def unwrap_tyarray(x: Array) -> TyArrayBase: ...


@overload
def unwrap_tyarray(x: NumpyScalar) -> TyArrayBase: ...  # type: ignore[overload-overlap]


@overload
def unwrap_tyarray(x: PY_SCALAR) -> PY_SCALAR: ...


@overload
def unwrap_tyarray(x: None) -> None: ...


def unwrap_tyarray(x: Array | Scalar | None) -> TyArrayBase | PyScalar | None:
    """Unwrap arrays and strongly typed NumPy scalar operands."""
    if isinstance(x, Array):
        return x._tyarray
    if isinstance(x, NumpyScalar):
        from ._typed_array import funcs as tyfuncs

        return tyfuncs.astyarray(x)
    return x
