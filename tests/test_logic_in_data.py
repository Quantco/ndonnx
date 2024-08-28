# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from ndonnx._logic_in_data import Array, dtypes


@pytest.mark.parametrize(
    "scalar, dtype, res_dtype",
    [
        (1, dtypes.int32, dtypes.int32),
        (1.0, dtypes.int32, dtypes.float64),
        (1.0, dtypes.float32, dtypes.float32),
        (1.0, dtypes.float64, dtypes.float64),
    ],
)
def test_radd_pyscalar(scalar, dtype, res_dtype):
    shape = ("N",)
    res = scalar + Array(shape, dtype)

    assert res.dtype == res_dtype
    assert res._data.shape == shape
    assert res.shape == (None,)
