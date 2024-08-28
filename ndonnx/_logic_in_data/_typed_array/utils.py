# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

# from ._py_scalars import _ArrayPyFloat, _ArrayPyInt
# from ._array_ma import NInt8Data, NInt16Data, NInt32Data, NInt64Data, NBoolData, NFloat16Data, NFloat32Data, NFloat64Data
# from ._core_types import Int8Data, Int16Data, Int32Data, Int64Data, BoolData, Float16Data, Float32Data, Float64Data
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dtypes import DType
    from .typed_array import DTYPE, _TypedArray

# from .. import dtypes
# from . import py_scalars

# dtype_to_typed_array = {
#     dtypes._PyFloat: py_scalars._ArrayPyFloat,
#     dtypes._PyInt: py_scalars._ArrayPyInt,
# }


def promote(
    lhs: _TypedArray[DTYPE], *others: _TypedArray[DTYPE]
) -> list[_TypedArray[DType]]:
    from types import NotImplementedType

    acc: DType = lhs.dtype
    for other in others:
        updated = acc._result_type(other.dtype)
        if isinstance(updated, NotImplementedType):
            updated = other.dtype._rresult_type(acc)
        if isinstance(updated, NotImplementedType):
            raise TypeError("Failed to promote into common data type.")
        acc = updated
    return [lhs.astype(acc)] + [other.astype(acc) for other in others]
