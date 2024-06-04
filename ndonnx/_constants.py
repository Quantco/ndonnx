# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

import numpy as np

from ndonnx._funcs import asarray

e = asarray(np.e)
inf = asarray(np.inf)
nan = asarray(np.nan)
pi = asarray(np.pi)
newaxis = None

__all__ = [
    "e",
    "inf",
    "nan",
    "pi",
    "newaxis",
]
