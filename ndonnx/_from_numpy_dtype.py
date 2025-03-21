# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ndonnx._typed_array import datetime, onnx


def from_numpy_dtype(
    np_dtype: np.dtype,
) -> onnx.DTypes | datetime.TimeDelta64DType | datetime.DateTime64DType:
    # We must import lazily since the onnx module is using this function
    from ndonnx._typed_array import datetime, onnx

    # Ensure that this also works with np.generic such as np.int64
    np_dtype = np.dtype(np_dtype)

    if np_dtype.kind == "M":
        unit_str = np.datetime_data(np_dtype)[0]
        unit = datetime.validate_unit(unit_str)
        return datetime.DateTime64DType(unit=unit)

    if np_dtype.kind == "m":
        unit_str = np.datetime_data(np_dtype)[0]
        unit = datetime.validate_unit(unit_str)
        return datetime.TimeDelta64DType(unit=unit)

    # We don't support "T" ("Text" is the kind used for `StringDType` in numpy >= 2), yet.
    # See https://numpy.org/neps/nep-0055-string_dtype.html#python-api-for-stringdtype
    if np_dtype.kind in ["U", "i", "b", "f", "u"]:
        return onnx.from_numpy_dtype(np_dtype)

    raise ValueError(f"'{np_dtype}' does not have a corresponding ndonnx data type")
