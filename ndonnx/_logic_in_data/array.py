# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import operator as std_ops
from collections.abc import Callable
from types import EllipsisType, NotImplementedType
from typing import Any, Optional, Union, overload

import numpy as np
import spox.opset.ai.onnx.v21 as op
from spox import Var

from ._typed_array import TyArrayBase, ascoredata
from ._typed_array.funcs import astypedarray
from .dtypes import CoreDTypes, DType

StrictShape = tuple[int, ...]
StandardShape = int | tuple[int | None, ...]
OnnxShape = tuple[int | str | None, ...]

GetitemIndex = Union[
    int | slice | EllipsisType | None,
    tuple[int | slice | EllipsisType | None, ...],
    "Array",
]

SetitemIndex = Union[
    int | slice | EllipsisType, tuple[int | slice | EllipsisType, ...], "Array"
]


class Array:
    """User-facing objects that makes no assumption about any data type related
    logic."""

    _data: TyArrayBase

    @overload
    def __init__(self, shape: OnnxShape, dtype: DType): ...

    @overload
    def __init__(self, value: np.ndarray | float | int | bool): ...

    def __init__(self, shape=None, dtype=None, value=None, var=None):
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
            self._data = dtype._argument(shape)
            return
        if isinstance(value, np.ndarray):
            raise NotImplementedError
        if isinstance(value, int | float):
            ty_arr = astypedarray(value, use_py_scalars=False, dtype=dtype)
            self._data = ty_arr
            return

        raise NotImplementedError

    @classmethod
    def _from_data(cls, data: TyArrayBase) -> Array:
        if not isinstance(data, TyArrayBase):
            raise TypeError(f"expected '_TypedArrayBase', found `{type(data)}`")
        inst = cls.__new__(cls)
        inst._data = data
        return inst

    @property
    def shape(self) -> tuple[int | None, ...]:
        shape = self._data.shape
        return tuple(None if isinstance(item, str) else item for item in shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dynamic_shape(self) -> Array:
        shape = self._data.dynamic_shape
        return Array._from_data(shape)

    @property
    def dtype(self) -> DType:
        return self._data.dtype

    def astype(self, dtype: DType) -> Array:
        new_data = self._data.astype(dtype)
        return Array._from_data(new_data)

    def unwrap_numpy(self) -> np.ndarray:
        """Return the propagated value as a NumPy array if available.

        Raises
        ------
        ValueError:
            If no propagated value is available.
        """
        return self._data.unwrap_numpy()

    def __getitem__(self, key: GetitemIndex, /) -> Array:
        from ._typed_array.core import TyArrayBool, TyArrayInteger
        from ._typed_array.indexing import GetitemIndexStatic

        if isinstance(key, Array):
            if key.ndim > self.ndim:
                raise IndexError(
                    "Rank of 'key' must be less or equal to rank of 'self'"
                )
            elif not all(ks in (xs, 0) for xs, ks in zip(self.shape, key.shape)):
                raise IndexError("Shape of 'key' is incompatible with shape of 'self'")
            if not isinstance(key._data, TyArrayInteger | TyArrayBool):
                raise TypeError(
                    f"indexing array must have integer or boolean data type; found `{key.dtype}`"
                )
            idx: GetitemIndexStatic | TyArrayInteger | TyArrayBool = key._data
        else:
            idx = key
        data = self._data[idx]
        return type(self)._from_data(data)

    def __setitem__(
        self: Array,
        key: SetitemIndex,
        value: Union[int, float, bool, Array],
        /,
    ) -> None:
        from ._typed_array.core import TyArrayBool, TyArrayInteger
        from ._typed_array.indexing import SetitemIndexStatic

        if isinstance(key, Array):
            if not isinstance(key._data, TyArrayInteger | TyArrayBool):
                raise TypeError(
                    f"indexing array must have integer or boolean data type; found `{key.dtype}`"
                )
            idx: SetitemIndexStatic | TyArrayInteger | TyArrayBool = key._data
        else:
            idx = key

        # Specs say that the data type of self must not be changed.
        self._data[idx] = asarray(value, dtype=self.dtype)._data

    def __bool__(self: Array, /) -> bool:
        return bool(self.unwrap_numpy())

    def __float__(self: Array, /) -> float:
        return float(self.unwrap_numpy())

    def __index__(self: Array, /) -> int:
        return self.unwrap_numpy().__index__()

    def __int__(self: Array, /) -> int:
        return int(self.unwrap_numpy())

    ##################################################################
    # __r*__ are needed for interacting with Python scalars          #
    # (e.g. doing 1 + Array(...)). These functions are _NOT_ used to #
    # dispatch between different `_TypedArray` subclasses.           #
    ##################################################################

    def __abs__(self: Array, /) -> Array:
        data = self._data.__abs__()
        return Array._from_data(data)

    def __add__(self: Array, rhs: int | float | Array, /) -> Array:
        return _apply_op(self, rhs, std_ops.add)

    def __radd__(self, lhs: int | float | Array, /) -> Array:
        return _apply_op(lhs, self, std_ops.add)

    def __and__(self: Array, other: Union[int, bool, Array], /) -> Array:
        raise NotImplementedError

    def __array_namespace__(
        self: Array, /, *, api_version: Optional[str] = None
    ) -> Any:
        raise NotImplementedError

    def __complex__(self: Array, /) -> complex:
        raise NotImplementedError

    def __eq__(self: Array, other: Union[int, float, bool, Array], /) -> Array:  # type: ignore
        return _apply_op(self, other, std_ops.eq)

    def __floordiv__(self: Array, other: Union[int, float, Array], /) -> Array:
        return _apply_op(self, other, std_ops.floordiv)

    def __ge__(self: Array, other: Union[int, float, Array], /) -> Array:
        return _apply_op(self, other, std_ops.ge)

    def __gt__(self: Array, other: Union[int, float, Array], /) -> Array:
        return _apply_op(self, other, std_ops.gt)

    def __invert__(self: Array, /) -> Array:
        raise NotImplementedError

    def __le__(self: Array, other: Union[int, float, Array], /) -> Array:
        raise NotImplementedError

    def __lshift__(self: Array, other: Union[int, Array], /) -> Array:
        raise NotImplementedError

    def __lt__(self: Array, other: Union[int, float, Array], /) -> Array:
        raise NotImplementedError

    def __matmul__(self: Array, other: Array, /) -> Array:
        raise NotImplementedError

    def __mod__(self: Array, other: Union[int, float, Array], /) -> Array:
        raise NotImplementedError

    def __mul__(self, rhs: int | float | Array) -> Array:
        return _apply_op(self, rhs, std_ops.mul)

    def __rmul__(self, lhs: int | float | Array) -> Array:
        return _apply_op(lhs, self, std_ops.mul)

    def __ne__(self: Array, other: Union[int, float, bool, Array], /) -> Array:  # type: ignore
        raise NotImplementedError

    def __neg__(self: Array, /) -> Array:
        raise NotImplementedError

    def __or__(self, rhs: int | bool | Array, /) -> Array:
        return _apply_op(self, rhs, std_ops.or_)

    def __ror__(self, lhs: int | float | Array) -> Array:
        return _apply_op(lhs, self, std_ops.or_)

    def __pos__(self: Array, /) -> Array:
        raise NotImplementedError

    def __pow__(self: Array, other: Union[int, float, Array], /) -> Array:
        raise NotImplementedError

    def __rshift__(self: Array, other: Union[int, Array], /) -> Array:
        raise NotImplementedError

    def __sub__(self, rhs: int | float | Array) -> Array:
        return _apply_op(self, rhs, std_ops.sub)

    def __rsub__(self, lhs: int | float | Array) -> Array:
        return _apply_op(lhs, self, std_ops.sub)

    def __truediv__(self: Array, other: Union[int, float, Array], /) -> Array:
        raise NotImplementedError

    def __xor__(self: Array, other: Union[int, bool, Array], /) -> Array:
        raise NotImplementedError

    def __repr__(self) -> str:
        # TODO: Visualize if we have a constant or something lazy for non-core types
        val = None
        if isinstance(self.dtype, CoreDTypes):
            try:
                val = self.unwrap_numpy().tolist()
            except ValueError:
                pass
        if val is not None:
            return f"array({val}, shape={self.shape}, dtype={self.dtype})"
        return f"array(shape={self.shape}, dtype={self.dtype})"


def asarray(
    obj: Array | bool | int | float | np.ndarray,
    /,
    *,
    dtype: DType | None = None,
    device=None,
    copy: bool | None = None,
) -> Array:
    if isinstance(obj, Array):
        return obj
    data: TyArrayBase = ascoredata(op.const(obj))
    if dtype:
        data = data.astype(dtype)
    return Array._from_data(data)


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
    op: Callable[[TyArrayBase, TyArrayBase], TyArrayBase],
) -> Array | NotImplementedType: ...


@overload
def _apply_op(
    lhs: int | float | Array,
    rhs: Array,
    op: Callable[[TyArrayBase, TyArrayBase], TyArrayBase],
) -> Array | NotImplementedType: ...


def _apply_op(
    lhs: int | float | Array,
    rhs: int | float | Array,
    op: Callable[[TyArrayBase, TyArrayBase], TyArrayBase],
) -> Array | NotImplementedType:
    lhs = _as_array(lhs, use_py_scalars=True)
    rhs = _as_array(rhs, use_py_scalars=True)
    data = op(lhs._data, rhs._data)
    if data is not NotImplemented:
        return Array._from_data(data)

    return NotImplemented
