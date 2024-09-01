# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import operator as std_ops
from collections.abc import Callable
from types import EllipsisType, NotImplementedType
from typing import overload

import numpy as np
import spox.opset.ai.onnx.v21 as op
from spox import Var

from ._typed_array import _TypedArray, ascoredata
from ._typed_array.funcs import astypedarray
from .dtypes import DType

StrictShape = tuple[int, ...]
StandardShape = tuple[int | None, ...]
OnnxShape = tuple[int | str | None, ...]

ScalarIndex = int | bool | slice | EllipsisType | None
Index = ScalarIndex | tuple[ScalarIndex, ...]


class Array:
    """User-facing objects that makes no assumption about any data type related
    logic."""

    _data: _TypedArray

    @overload
    def __init__(self, shape: OnnxShape, dtype: DType): ...

    @overload
    def __init__(self, value: np.ndarray | float | int | bool): ...

    def __init__(self, shape=None, dtype=None, value=None, var=None):
        # TODO: Validation
        (is_shape, is_dtype, is_value, is_var) = (
            item is not None for item in [shape, dtype, value, var]
        )
        if not (
            (True, True, False, False) == (is_shape, is_dtype, is_value, is_var)
            or (False, False, True, False) == (is_shape, is_dtype, is_value, is_var)
            or (False, False, False, True) == (is_shape, is_dtype, is_value, is_var)
        ):
            raise ValueError("Invalid arguments.")

        if isinstance(shape, tuple) and isinstance(dtype, DType):
            self._data = dtype._tyarr_class.as_argument(shape)
            return
        if isinstance(value, np.ndarray):
            raise NotImplementedError
        if isinstance(value, int | float):
            ty_arr = astypedarray(value, use_py_scalars=False, dtype=dtype)
            self._data = ty_arr
            return

        raise NotImplementedError

    @classmethod
    def _from_data(cls, data: _TypedArray) -> Array:
        inst = cls.__new__(cls)
        inst._data = data
        return inst

    @property
    def shape(self) -> StandardShape:
        shape = self._data.shape
        return tuple(None if isinstance(item, str) else item for item in shape)

    @property
    def dtype(self) -> DType:
        return self._data.dtype

    def astype(self, dtype: DType):
        return self._data.astype(dtype)

    def unwrap_numpy(self) -> np.ndarray:
        """Return the propagated value as a NumPy array if available.

        Raises
        ------
        ValueError:
            If no propagated value is available.
        """
        return self._data.to_numpy()

    def __getitem__(self, index: Index) -> Array:
        data = self._data[index]
        return type(self)._from_data(data)

    # __r*__ are needed for interacting with Python scalars
    # (e.g. doing 1 + Array(...)). These functions are _NOT_ used to
    # dispatch between different `_TypedArray` subclasses.

    def __add__(self, rhs: int | float | Array) -> Array:
        return _apply_op(self, rhs, std_ops.add)

    def __radd__(self, lhs: int | float | Array) -> Array:
        return _apply_op(lhs, self, std_ops.add)

    def __or__(self, rhs: int | float | Array) -> Array:
        return _apply_op(self, rhs, std_ops.or_)

    def __ror__(self, lhs: int | float | Array) -> Array:
        return _apply_op(lhs, self, std_ops.or_)


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


def _as_array(
    val: int | float | Array | Var | np.ndarray, use_py_scalars=False
) -> Array:
    if isinstance(val, Array):
        return val
    ty_arr = astypedarray(val, use_py_scalars=use_py_scalars)
    return Array._from_data(ty_arr)


@overload
def _apply_op(
    lhs: Array,
    rhs: int | float | Array,
    op: Callable[[_TypedArray, _TypedArray], _TypedArray],
) -> Array | NotImplementedType: ...


@overload
def _apply_op(
    lhs: int | float | Array,
    rhs: Array,
    op: Callable[[_TypedArray, _TypedArray], _TypedArray],
) -> Array | NotImplementedType: ...


def _apply_op(
    lhs: int | float | Array,
    rhs: int | float | Array,
    op: Callable[[_TypedArray, _TypedArray], _TypedArray],
) -> Array | NotImplementedType:
    lhs = _as_array(lhs, use_py_scalars=True)
    rhs = _as_array(rhs, use_py_scalars=True)
    data = op(lhs._data, rhs._data)
    if data is not NotImplemented:
        return Array._from_data(data)

    breakpoint()
    return NotImplemented
