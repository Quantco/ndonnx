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
        (1.0, dtypes.nfloat64, dtypes.nfloat64),
        (1.0, dtypes.nint32, dtypes.nfloat64),
    ],
)
def test_radd_pyscalar(scalar, dtype, res_dtype):
    shape = ("N",)
    res = scalar + Array(shape, dtype)

    assert res.dtype == res_dtype
    assert res._data.shape == shape
    assert res.shape == (None,)


@pytest.mark.parametrize(
    "dtype1, dtype2, res_dtype",
    [
        (dtypes.int32, dtypes.int32, dtypes.int32),
        (dtypes.int64, dtypes.int32, dtypes.int64),
        (dtypes.float64, dtypes.int32, dtypes.float64),
        (dtypes.nint32, dtypes.int32, dtypes.nint32),
    ],
)
def test_core_add(dtype1, dtype2, res_dtype):
    shape = ("N",)
    res = Array(shape, dtype1) + Array(shape, dtype2)

    assert res.dtype == res_dtype
    assert res._data.shape == shape
    assert res.shape == (None,)


@pytest.mark.parametrize(
    "dtype1, dtype2, res_dtype",
    [
        (dtypes.int32, dtypes.int32, dtypes.int32),
        (dtypes.int64, dtypes.int32, dtypes.int64),
        (dtypes.bool_, dtypes.bool_, dtypes.bool_),
    ],
)
def test_core_or(dtype1, dtype2, res_dtype):
    shape = ("N",)
    res = Array(shape, dtype1) | Array(shape, dtype2)

    assert res.dtype == res_dtype
    assert res._data.shape == shape
    assert res.shape == (None,)
