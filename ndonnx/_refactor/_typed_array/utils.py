# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, overload

if TYPE_CHECKING:
    from . import TyArrayBase
    from .masked_onnx import TyMaArray
    from .onnx import TyArray

T = TypeVar("T")
_PyScalar = bool | int | float | str


@overload
def promote(lhs: TyArray, *others: TyArray) -> tuple[TyArray, ...]: ...


@overload
def promote(lhs: TyArray, *others: TyArray | _PyScalar) -> tuple[TyArray, ...]: ...


@overload
def promote(
    lhs: TyArray | TyMaArray,
    *others: TyArray | TyMaArray | _PyScalar,
) -> tuple[TyArray | TyMaArray, ...]: ...


@overload
def promote(
    lhs: TyArrayBase, *others: TyArrayBase | _PyScalar
) -> tuple[TyArrayBase, ...]: ...


def promote(
    lhs: TyArrayBase, *others: TyArrayBase | _PyScalar
) -> tuple[TyArrayBase, ...]:
    from .funcs import astyarray, result_type

    res_type = result_type(lhs, *others)
    return tuple(
        [lhs.astype(res_type)] + [astyarray(other, dtype=res_type) for other in others]
    )


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
