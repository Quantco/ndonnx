# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from ndonnx._logic_in_data import Array


def test_radd():
    shape = ("N",)
    res = 1 + Array(shape, np.dtype("float32"))
    assert res._data.shape == shape
    assert res.shape == (None,)
