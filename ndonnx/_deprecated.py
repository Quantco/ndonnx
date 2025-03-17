# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
"""Functions for backward-compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias
from warnings import warn

from . import _typed_array as tydx
from ._array import Array
from ._dtypes import DType
from ._funcs import asarray

if TYPE_CHECKING:
    from spox import Var

    from ndonnx.types import OnnxShape


def array(
    *,
    shape: OnnxShape,
    dtype: DType,
) -> Array:
    return Array(shape=shape, dtype=dtype)


def from_spox_var(var: Var) -> Array:
    warn("'from_spox_var' is deprecated in favor of 'asarray'")
    return asarray(var)


Floating = tydx.onnx.FloatingDTypes
Integral = tydx.onnx.IntegerDTypes
Numerical = tydx.onnx.NumericDTypes
CoreType: TypeAlias = tydx.onnx.DTypes

Nullable = tydx.masked_onnx.DTypes
NullableFloating = tydx.masked_onnx.FloatDTypes
NullableIntegral = tydx.masked_onnx.IntegerDTypes
