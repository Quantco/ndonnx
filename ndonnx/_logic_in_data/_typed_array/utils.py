# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import NotImplementedType
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from ..dtypes import DType
    from .core import _ArrayCoreType
    from .typed_array import _TypedArray


@overload
def promote(
    lhs: _ArrayCoreType, *others: _ArrayCoreType
) -> tuple[_ArrayCoreType, ...]: ...


@overload
def promote(
    lhs: _TypedArray, *others: _TypedArray
) -> tuple[_TypedArray[DType], ...]: ...


def promote(lhs: _TypedArray, *others: _TypedArray) -> tuple[_TypedArray[DType], ...]:
    acc: DType = lhs.dtype
    for other in others:
        updated = acc._result_type(other.dtype)
        if isinstance(updated, NotImplementedType):
            updated = other.dtype._rresult_type(acc)
        if isinstance(updated, NotImplementedType):
            raise TypeError("Failed to promote into common data type.")
        acc = updated
    return tuple([lhs.astype(acc)] + [other.astype(acc) for other in others])
