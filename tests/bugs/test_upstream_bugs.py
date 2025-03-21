# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import spox.opset.ai.onnx.v21 as op


@pytest.mark.parametrize(
    "dtype, succeeds",
    [
        (np.int8, True),
        (np.int16, True),
        (np.int32, False),
        (np.int64, False),
        (np.uint8, True),
        (np.uint16, True),
        (np.uint32, True),
        (np.uint64, True),
        (np.float32, True),
        (np.float64, True),
        (np.bool_, True),
        (np.str_, True),
    ],
)
def test_concat(dtype, succeeds):
    # Concat appears to have a bug in the type/shape inference that
    # raises an exception for some data types.

    a = op.const([[]], np.uint64)
    op.concat([a, a], axis=0)

    a = op.const([[]], dtype)
    op.concat([a, a], axis=0)

    a = op.const([], np.uint64)
    op.concat([a, a], axis=0)

    a = op.const([1], dtype)
    op.concat([a, a], axis=0)

    # fails for some data types
    a = op.const([], dtype)
    if succeeds:
        op.concat([a, a], axis=0)
    else:
        with pytest.raises(Exception, match=r"axis must be in \[-rank, rank-1\]"):
            op.concat([a, a], axis=0)


@pytest.mark.skip(reason="Well, it segfaults...")
def test_segfault_mul():
    # Works:
    op.mul(op.const([[]], np.int64), op.const([1], np.int64))
    # Segfaults
    op.mul(op.const([], np.int64), op.const([1], np.int64))
