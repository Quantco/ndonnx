# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import overload

import numpy as np
import spox.opset.ai.onnx.v21 as op
from spox import Tensor, argument

from .data import Data, ascoredata
from .dtypes import DType, as_numpy

StrictShape = tuple[int, ...]
StandardShape = tuple[int | None, ...]
OnnxShape = tuple[int | str | None, ...]


def _get_data(val: int | float | Array) -> Data:
    raise NotImplementedError


class Array:
    _data: Data

    @overload
    def __init__(self, shape: OnnxShape, dtype: DType): ...

    @overload
    def __init__(self, value: np.ndarray): ...

    def __init__(self, shape=None, dtype=None, value=None, var=None):
        # TODO: Validation
        # if value is not None:
        #     np.ma.isMaskedArray(value)
        #     data = NullableCoreData()
        if shape is not None and dtype is not None:
            var = argument(Tensor(as_numpy(dtype), shape))
            self._data = ascoredata(var)
            return
        raise NotImplementedError

    @classmethod
    def _from_data(cls, data: Data) -> Array:
        inst = cls.__new__(cls)
        inst._data = data
        return inst

    def __add__(self, rhs: int | float | Array) -> Array:
        rhs_data = _get_data(rhs)
        data = self._data + rhs_data
        if data is not NotImplemented:
            return Array._from_data(data)
        return NotImplemented

    def __radd__(self, lhs: int | float | Array) -> Array:
        # This is called for instance when doing int + Array
        lhs_data = _get_data(lhs)
        data = lhs_data + self._data
        if data is not NotImplemented:
            return Array._from_data(data)
        return NotImplemented

    @property
    def shape(self) -> StandardShape:
        shape = self._data.shape
        return tuple(None if isinstance(item, str) else item for item in shape)

    @property
    def dtype(self) -> DType:
        return self._data.dtype


def asarray(obj: int | float | bool | str | Array) -> Array:
    if isinstance(obj, Array):
        return obj
    data = ascoredata(op.const(obj))
    return Array._from_data(data)


# def where(cond: Array, a: Array, b: Array) -> Array:
#     from .dtypes import bool_, nbool

#     if cond.dtype not in [bool_, nbool]:
#         raise ValueError
#     ret = cond._data._where(*a._data.promote(b._data))
#     if ret == NotImplemented:
#         ret


def add(a: Array, b: Array) -> Array:
    return a + b
