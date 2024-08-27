# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

# from ._py_scalars import _ArrayPyFloat, _ArrayPyInt
# from ._array_ma import NInt8Data, NInt16Data, NInt32Data, NInt64Data, NBoolData, NFloat16Data, NFloat32Data, NFloat64Data
# from ._core_types import Int8Data, Int16Data, Int32Data, Int64Data, BoolData, Float16Data, Float32Data, Float64Data

from .. import dtypes
from . import py_scalars

dtype_to_typed_array = {
    dtypes._PyFloat: py_scalars._ArrayPyFloat,
    dtypes._PyInt: py_scalars._ArrayPyInt,
}
