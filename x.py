# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import ndonnx._refactor as ndx

x = ndx.asarray(["abc", "def"], dtype=ndx.nutf8)
x = ndx.asarray(["abc", "def"], dtype=ndx.utf8)
x = ndx.asarray(["abc", "def"])
x = ndx.asarray(np.asarray(["abc", "def"]))
x = ndx.asarray(np.ma.masked_array(["abc", "def"], mask=[1, 0]))
x = ndx.asarray(
    np.asarray(["1990-01-01", "2023-01-31", "2000-01-01"], dtype="datetime64[s]")
)

x = ndx.asarray(["abc", "def", "kw"], dtype=ndx.object_dtype)
# illustrates why we need to be able to do the obove statement. This will crash since, appropriately, None is not a string.
y = ndx.asarray([None, "ghi", np.nan], dtype=ndx.object_dtype)
print(x)
print(y)
z = x + y
print(z)
