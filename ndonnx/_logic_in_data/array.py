# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import overload

import numpy as np
import spox.opset.ai.onnx.v21 as op

from ._typed_array import _ArrayPyFloat, _ArrayPyInt, _TypedArray, ascoredata
from .dtypes import DType

StrictShape = tuple[int, ...]
StandardShape = tuple[int | None, ...]
OnnxShape = tuple[int | str | None, ...]


class Array:
    """User-facing objects that makes no assumption about any data type related
    logic."""

    _data: _TypedArray

    @overload
    def __init__(self, shape: OnnxShape, dtype: DType): ...

    @overload
    def __init__(self, value: np.ndarray): ...

    def __init__(self, shape=None, dtype=None, value=None, var=None):
        # TODO: Validation
        # if value is not None:
        #     np.ma.isMaskedArray(value)
        #     data = NullableCoreData()
        if isinstance(shape, tuple) and isinstance(dtype, DType):
            self._data = dtype._data_class.as_argument(shape)
            return
        raise NotImplementedError

    @classmethod
    def _from_data(cls, data: _TypedArray) -> Array:
        inst = cls.__new__(cls)
        inst._data = data
        return inst

    def astype(self, dtype: DType):
        return self._data.astype(dtype)

    def __add__(self, rhs: int | float | Array) -> Array:
        rhs_data = _as_typed_array(rhs)
        data = self._data + rhs_data
        if data is not NotImplemented:
            return Array._from_data(data)
        return NotImplemented

    def __radd__(self, lhs: int | float | Array) -> Array:
        # This is called for instance when doing int + Array
        lhs_data = _as_typed_array(lhs)
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


def _as_typed_array(val: int | float | Array) -> _TypedArray:
    if isinstance(val, Array):
        return val._data
    if isinstance(val, int):
        return _ArrayPyInt(val)
    if isinstance(val, float):
        return _ArrayPyFloat(val)

    raise ValueError


def add(a: Array, b: Array) -> Array:
    return a + b
