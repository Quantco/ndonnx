# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import spox.opset.ai.onnx.v21 as op


@pytest.mark.skip(reason="Well, it segfaults...")
def test_segfault_mul():
    # Works:
    op.mul(op.const([[]], np.int64), op.const([1], np.int64))
    # Segfaults
    op.mul(op.const([], np.int64), op.const([1], np.int64))
