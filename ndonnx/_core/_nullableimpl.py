# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import ndonnx as ndx

from ._interface import OperationsBlock
from ._utils import validate_core


class NullableOperationsImpl(OperationsBlock):
    @validate_core
    def fill_null(self, x, value):
        value = ndx.asarray(value)
        if value.dtype != x.values.dtype:
            value = value.astype(x.values.dtype)
        return ndx.where(x.null, value, x.values)
