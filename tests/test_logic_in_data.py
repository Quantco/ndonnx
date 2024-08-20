# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause


from ndonnx._logic_in_data import Array, dtypes


def test_radd():
    shape = ("N",)
    res = 1 + Array(shape, dtypes.int32)
    assert res.dtype == dtypes.int32  # Do not cast Python scalar to default int64
    assert res._data.shape == shape
    assert res.shape == (None,)
