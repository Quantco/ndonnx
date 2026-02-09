# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from ndonnx._typed_array import datetime, onnx, string


def from_numpy_dtype(
    np_dtype: np.dtype,
) -> (
    onnx.DTypes
    | datetime.TimeDelta64DType
    | datetime.DateTime64DType
    | string.StringDType
):
    # Ensure that this also works with np.generic such as np.int64
    np_dtype = np.dtype(np_dtype)

    if np_dtype.kind in "Mm":
        return datetime.from_numpy_dtype(np_dtype)

    # We don't support "T" ("Text" is the kind used for `StringDType` in numpy >= 2), yet.
    # See https://numpy.org/neps/nep-0055-string_dtype.html#python-api-for-stringdtype
    if np_dtype.kind in ["U", "i", "b", "f", "u"]:
        return onnx.from_numpy_dtype(np_dtype)

    if np_dtype.kind == "T":
        return string.from_numpy_dtype(np_dtype)

    raise ValueError(f"'{np_dtype}' does not have a corresponding ndonnx data type")
