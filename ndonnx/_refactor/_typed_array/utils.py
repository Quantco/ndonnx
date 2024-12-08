# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import NotImplementedType
from typing import TYPE_CHECKING, TypeVar, overload

if TYPE_CHECKING:
    from .._dtypes import DType
    from . import TyArray, TyMaArray
    from .py_scalars import _ArrayPyScalar
    from .typed_array import TyArrayBase

T = TypeVar("T")


@overload
def promote(lhs: TyArray, *others: TyArray) -> tuple[TyArray, ...]: ...


@overload
def promote(
    lhs: TyArray | _ArrayPyScalar, *others: TyArray | _ArrayPyScalar
) -> tuple[TyArray, ...]: ...


@overload
def promote(
    lhs: TyArray | TyMaArray | _ArrayPyScalar,
    *others: TyArray | TyMaArray | _ArrayPyScalar,
) -> tuple[TyArray | TyMaArray, ...]: ...


@overload
def promote(lhs: TyArrayBase, *others: TyArrayBase) -> tuple[TyArrayBase, ...]: ...


def promote(lhs: TyArrayBase, *others: TyArrayBase) -> tuple[TyArrayBase, ...]:
    acc: DType = lhs.dtype
    for other in others:
        updated = acc._result_type(other.dtype)
        if isinstance(updated, NotImplementedType):
            updated = other.dtype._result_type(acc)
        if isinstance(updated, NotImplementedType):
            raise TypeError("Failed to promote into common data type.")
        acc = updated
    return tuple([lhs.astype(acc)] + [other.astype(acc) for other in others])


def safe_cast(ty: type[T], a: TyArrayBase | bool) -> T:
    # The union with bool is due to mypy thinking that __eq__ always
    # returns bool.
    if isinstance(a, ty):
        return a
    raise TypeError(f"Expected `{ty}` found `{type(a)}`")


def normalize_axes_tuple(axes: int | tuple[int, ...], rank: int) -> tuple[int, ...]:
    if isinstance(axes, int):
        axes = (axes,)

    return tuple(el if el >= 0 else rank + el for el in axes)