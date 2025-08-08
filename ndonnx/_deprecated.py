# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
"""Functions for backward-compatibility."""

from __future__ import annotations

from typing import TypeAlias
from warnings import warn

from spox import Var

import ndonnx as ndx
from ndonnx import _typed_array as tydx


def from_spox_var(var: Var) -> ndx.Array:
    warn(
        "'from_spox_var' is deprecated in favor of 'asarray'",
        DeprecationWarning,
        stacklevel=2,
    )
    return ndx.asarray(var)


Floating = tydx.onnx.FloatingDTypes
Integral = tydx.onnx.IntegerDTypes
Numerical = tydx.onnx.NumericDTypes
CoreType: TypeAlias = tydx.onnx.DTypes

Nullable = tydx.masked_onnx.DTypes
NullableFloating = tydx.masked_onnx.FloatDTypes
NullableIntegral = tydx.masked_onnx.IntegerDTypes
