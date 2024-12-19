# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
"""Functions for backward-compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias
from warnings import warn

from ._array import Array
from ._dtypes import DType
from ._funcs import asarray
from ._typed_array.masked_onnx import (
    NCoreDTypes,
    NCoreFloatingDTypes,
    NCoreIntegerDTypes,
)
from ._typed_array.onnx import DTypes, FloatingDTypes, IntegerDTypes

if TYPE_CHECKING:
    from spox import Var


def array(
    *,
    shape: tuple[int | str | None, ...],
    dtype: DType,
) -> Array:
    return Array(shape=shape, dtype=dtype)


def from_spox_var(var: Var) -> Array:
    warn("'from_spox_var' is deprecated in favor of 'asarray'")
    return asarray(var)


class UnsupportedOperationError(TypeError): ...


Nullable = NCoreDTypes
Floating = FloatingDTypes
Integral = IntegerDTypes
Numerical = Floating | Integral
CoreType: TypeAlias = DTypes

NullableFloating = NCoreFloatingDTypes
NullableIntegral = NCoreIntegerDTypes
