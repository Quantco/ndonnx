# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
"""Functions for backward-compatibility."""

from ._array import Array
from ._dtypes import DType


def array(
    *,
    shape: tuple[int | str | None, ...],
    dtype: DType,
) -> Array:
    return Array(shape=shape, dtype=dtype)
