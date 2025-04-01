# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import operator as ops
import sys

import numpy as np
import pytest

import ndonnx as ndx

BINARY_ARITHMETIC_OPS = [ops.add, ops.mul, ops.sub]
NUMERICAL_DTYPES = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
]


@pytest.mark.parametrize("op", BINARY_ARITHMETIC_OPS)
@pytest.mark.parametrize("dtype", NUMERICAL_DTYPES)
@pytest.mark.parametrize("reflected", [True, False])
def test_bool_arithmetics(op, dtype, reflected):
    def do(npx, reflected: bool):
        a = npx.asarray([True])
        b = npx.asarray([2], dtype=getattr(npx, dtype))
        if reflected:
            return op(b, a)
        return op(a, b)

    np.testing.assert_array_equal(
        do(ndx, reflected).unwrap_numpy(), do(np, reflected), strict=True
    )


@pytest.mark.parametrize("op", BINARY_ARITHMETIC_OPS)
@pytest.mark.parametrize("dtype", NUMERICAL_DTYPES)
@pytest.mark.parametrize("reflected", [True, False])
def test_bool_arithmetics_with_python_scalar(op, dtype, reflected):
    def do(npx, reflected: bool):
        a = npx.asarray([True])
        b = 2
        if reflected:
            return op(b, a)
        return op(a, b)

    strict = not (sys.platform.startswith("win") and np.__version__ < "2")

    np.testing.assert_array_equal(
        do(ndx, reflected).unwrap_numpy(), do(np, reflected), strict=strict
    )
