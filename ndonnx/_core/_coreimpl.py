# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from spox import Tensor, argument

import ndonnx as ndx
import ndonnx._data_types as dtypes
import ndonnx.additional as nda
from ndonnx._corearray import _CoreArray

from ._interface import OperationsBlock
from ._utils import validate_core


class CoreOperationsImpl(OperationsBlock):
    def make_array(self, shape, dtype, eager_value=None):
        return ndx.Array._from_fields(
            dtype,
            data=_CoreArray(
                dtype._parse_input(eager_value)["data"]
                if eager_value is not None
                else argument(Tensor(dtype.to_numpy_dtype(), shape))
            ),
        )

    @validate_core
    def make_nullable(self, x, null):
        if null.dtype != ndx.bool:
            raise TypeError("'null' must be a boolean array")

        return ndx.Array._from_fields(
            dtypes.into_nullable(x.dtype),
            values=x.copy(),
            null=ndx.reshape(null, nda.shape(x)),
        )
