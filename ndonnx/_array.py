# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import operator as std_ops
from collections.abc import Callable
from enum import Enum
from types import EllipsisType, NotImplementedType
from typing import TYPE_CHECKING, Any
from warnings import warn

import numpy as np
from spox import Var

from ndonnx import DType

from ._namespace_info import Device, device
from ._typed_array import TyArrayBase, onnx
from ._typed_array import funcs as tyfuncs
from ._typed_array.masked_onnx import TyMaArray
from .extensions import get_mask

if TYPE_CHECKING:
    from .types import GetItemKey, OnnxShape, PyScalar, SetitemKey


_BinaryOp = Callable[["Array", "int | bool | str | float | Array"], "Array"]
_Axisparam = int | tuple[int, ...] | None


def _make_binary(
    tyarr_op: Callable[[TyArrayBase | PyScalar, TyArrayBase | PyScalar], TyArrayBase],
) -> tuple[_BinaryOp, _BinaryOp]:
    def binary_op_forward(self, other):
        return _apply_op(self, other, tyarr_op)

    def binary_op_backward(self, other):
        return _apply_op(other, self, tyarr_op)

    return binary_op_forward, binary_op_backward


class Array:
    """User-facing objects that makes no assumption about any data type related
    logic."""

    _tyarray: TyArrayBase

    def __init__(self, *args, **kwargs) -> None:
        raise TypeError(
            "'Array' cannot be instantiated directly. Use the 'ndonnx.array' or 'ndonnx.asarray' functions instead"
        )

    @classmethod
    def _argument(cls, /, *, shape: OnnxShape, dtype: DType) -> Array:
        inst = cls.__new__(cls)
        inst._tyarray = dtype.__ndx_argument__(shape)
        return inst

    @classmethod
    def _constant(
        cls, /, *, value: PyScalar | np.ndarray, dtype: DType | None
    ) -> Array:
        return cls._from_tyarray(tyfuncs.astyarray(value, dtype=dtype))

    @classmethod
    def _from_tyarray(cls, tyarray: TyArrayBase, /) -> Array:
        if not isinstance(tyarray, TyArrayBase):
            raise TypeError(f"expected 'TypedArrayBase', found `{type(tyarray)}`")
        inst = cls.__new__(cls)
        inst._tyarray = tyarray
        return inst

    @property
    def device(self) -> Device:
        return device

    @property
    def dtype(self) -> DType:
        return self._tyarray.dtype

    @property
    def dynamic_shape(self) -> Array:
        """Runtime shape of this array as a 1D int64 tensor."""
        shape = self._tyarray.dynamic_shape
        return Array._from_tyarray(shape)

    @property
    def mT(self) -> Array:  # noqa: N802
        return Array._from_tyarray(self._tyarray.mT)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> tuple[int | None, ...]:
        shape = self._tyarray.shape
        return tuple(None if isinstance(item, str) else item for item in shape)

    @property
    def size(self) -> int | None:
        static_dims = []
        for el in self.shape:
            if el is None:
                return None
            static_dims.append(el)

        return math.prod(static_dims)

    @property
    def dynamic_size(self) -> Array:
        """Return the size of an array as scalar array.

        Contrary to `Array.size` this function also works on dynamically sized arrays.
        """
        # Special cases that allow for shortcuts
        if self.ndim == 0:
            Array._from_tyarray(onnx.const(1, dtype=onnx.int64))
        if self.ndim == 1:
            return self.dynamic_shape

        return Array._from_tyarray(self.dynamic_shape._tyarray.prod())

    @property
    def T(self) -> Array:  # noqa: N802
        return Array._from_tyarray(self._tyarray.T)

    @property
    def null(self) -> None | Array:
        warn(
            "'Array.null' is deprecated in favor of 'ndonnx.extensions.get_mask'",
            DeprecationWarning,
        )
        return get_mask(self)

    @property
    def values(self) -> Array:
        warn(
            "'Array.values' is deprecated in favor of 'ndonnx.extensions.get_data'",
            DeprecationWarning,
        )

        if isinstance(self._tyarray, TyMaArray):
            return Array._from_tyarray(self._tyarray.data)
        if isinstance(self._tyarray, onnx.TyArray):
            return Array._from_tyarray(self._tyarray)
        raise ValueError(f"`{self.dtype}` is not a nullable built-in type")

    def astype(self, dtype: DType, *, copy=True) -> Array:
        new_data = self._tyarray.astype(dtype, copy=copy)
        return Array._from_tyarray(new_data)

    def copy(self) -> Array:
        return Array._from_tyarray(self._tyarray.copy())

    def to_numpy(self) -> np.ndarray | None:
        warn(
            "'Array.to_numpy' is deprecated in favor of 'Array.unwrap_numpy'",
            DeprecationWarning,
        )
        try:
            return self.unwrap_numpy()
        except ValueError:
            return None

    def spox_var(self) -> Var:
        """Unwrap the underlying ``spox.Var`` object if ``self`` is of primitive data
        type.

        Otherwise, raise an exception.
        """
        warn(
            "'Array.spox_var' is deprecated in favor of 'Array.disassemble' or 'Array.unwrap_spox'",
            DeprecationWarning,
        )
        return self.unwrap_spox()

    def unwrap_spox(self) -> Var:
        """Unwrap the underlying ``spox.Var`` object if ``self`` is of primitive data
        type.

        Otherwise, raise an exception.
        """
        if isinstance(self._tyarray, onnx.TyArray):
            return self._tyarray.disassemble()

        raise TypeError(
            "cannot safely unwrap underlying 'spox.Var' object(s) "
            f"from array of data type `{self.dtype}`"
        )

    def to_device(self, device: Any, /, *, stream: int | Any | None = None) -> Array:
        raise ValueError("ONNX provides no control over the used device")

    def unwrap_numpy(self) -> np.ndarray:
        """Return the propagated value as a NumPy array if available.

        Raises
        ------
        ValueError:
            If no propagated value is available.
        """
        return self._tyarray.unwrap_numpy()

    def disassemble(self) -> dict[str, Var] | Var:
        """Disassemble into the constituent ``spox.Var`` objects.

        The particular layout depends on the data type.
        """
        return self._tyarray.disassemble()

    def __dlpack__(
        self,
        *,
        stream: int | Any | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[Enum, int] | None = None,
        copy: bool | None = None,
    ) -> Any:
        raise BufferError("ndonnx does not support the export of array data")

    def __dlpack_device__(self) -> tuple[Enum, int]:
        raise ValueError("ONNX provides no control over the used device")

    def __iter__(self):
        try:
            n, *_ = self.shape
        except IndexError:
            raise ValueError("iteration over 0-d array")
        if isinstance(n, int):
            return (self[i, ...] for i in range(n))
        raise ValueError(
            "iteration requires dimension of static length, but dimension 0 is dynamic"
        )

    def __getitem__(self, key: GetItemKey, /) -> Array:
        idx = _normalize_arrays_in_key(key)
        data = self._tyarray[idx]
        return type(self)._from_tyarray(data)

    def __setitem__(
        self,
        key: SetitemKey,
        value: str | int | float | bool | Array,
        /,
    ) -> None:
        # Specs say that the data type of self must not be changed.
        updates = (
            value._tyarray.astype(self.dtype)
            if isinstance(value, Array)
            else tyfuncs.astyarray(value, dtype=self.dtype)
        )

        if isinstance(key, Array):
            if not isinstance(key._tyarray, onnx.TyArrayInteger | onnx.TyArrayBool):
                raise TypeError(
                    f"indexing array must have integer or boolean data type; found `{key.dtype}`"
                )
            self._tyarray[key._tyarray] = updates
        else:
            self._tyarray[key] = updates

    def __bool__(self, /) -> bool:
        return bool(self.unwrap_numpy())

    def __complex__(self, /) -> complex:
        return self.unwrap_numpy().__complex__()

    def __float__(self, /) -> float:
        return float(self.unwrap_numpy())

    def __index__(self, /) -> int:
        return self.unwrap_numpy().__index__()

    def __int__(self, /) -> int:
        return int(self.unwrap_numpy())

    def __array_namespace__(self, /, *, api_version: str | None = None) -> Any:
        import ndonnx as ndx

        return ndx

    # We spell out __eq__ and __ne__ so that mypy may pick up the
    # change in return type (Array rather than bool)
    def __eq__(self, other) -> Array:  # type: ignore[override]
        return _apply_op(self, other, std_ops.eq)

    def __ne__(self, other) -> Array:  # type: ignore[override]
        return _apply_op(self, other, std_ops.ne)

    ##################################################################
    # __r*__ are needed for interacting with Python scalars          #
    # (e.g. doing 1 + Array(...)). These functions are _NOT_ used to #
    # dispatch between different `_TypedArray` subclasses.           #
    ##################################################################

    __add__, __radd__ = _make_binary(std_ops.add)
    __and__, __rand__ = _make_binary(std_ops.and_)
    __floordiv__, __rfloordiv__ = _make_binary(std_ops.floordiv)
    __ge__, _ = _make_binary(std_ops.ge)
    __gt__, _ = _make_binary(std_ops.gt)
    __le__, _ = _make_binary(std_ops.le)
    __lshift__, __rlshift__ = _make_binary(std_ops.lshift)
    __lt__, _ = _make_binary(std_ops.lt)
    __matmul__, __rmatmul__ = _make_binary(std_ops.matmul)
    __mod__, __rmod__ = _make_binary(std_ops.mod)
    __mul__, __rmul__ = _make_binary(std_ops.mul)
    __or__, __ror__ = _make_binary(std_ops.or_)
    __pow__, __rpow__ = _make_binary(std_ops.pow)
    __rshift__, __rrshift__ = _make_binary(std_ops.rshift)
    __sub__, __rsub__ = _make_binary(std_ops.sub)
    __truediv__, __rtruediv__ = _make_binary(std_ops.truediv)
    __xor__, __rxor__ = _make_binary(std_ops.xor)

    def __abs__(self, /) -> Array:
        data = self._tyarray.__abs__()
        return Array._from_tyarray(data)

    def __invert__(self, /) -> Array:
        return Array._from_tyarray(~self._tyarray)

    def __neg__(self, /) -> Array:
        return Array._from_tyarray(-self._tyarray)

    def __pos__(self, /) -> Array:
        return Array._from_tyarray(+self._tyarray)

    def __repr__(self) -> str:
        value_repr = ", ".join(
            [f"{k}: {v}" for k, v in self._tyarray.__ndx_value_repr__().items()]
        )
        # We only add shape information if we don't have a constant value to show
        shape_info = (
            "" if self._tyarray.is_constant else f" shape={self._tyarray.shape},"
        )
        return f"array({value_repr},{shape_info} dtype={self.dtype})"

    # Non-standard functions exposed by NumPy and ndonnx <=0.9
    def sum(self, axis: _Axisparam = 0, keepdims: bool = False) -> Array:
        """See :py:func:`ndonnx.sum` for documentation."""
        return Array._from_tyarray(self._tyarray.sum(axis=axis, keepdims=keepdims))

    def prod(self, axis: _Axisparam = 0, keepdims: bool = False) -> Array:
        """See :py:func:`ndonnx.prod` for documentation."""
        return Array._from_tyarray(self._tyarray.prod(axis=axis, keepdims=keepdims))

    def max(self, axis: _Axisparam = 0, keepdims: bool = False) -> Array:
        """See :py:func:`ndonnx.max` for documentation."""
        return Array._from_tyarray(self._tyarray.max(axis=axis, keepdims=keepdims))

    def min(self, axis: _Axisparam = 0, keepdims: bool = False) -> Array:
        """See :py:func:`ndonnx.min` for documentation."""
        return Array._from_tyarray(self._tyarray.min(axis=axis, keepdims=keepdims))

    def all(self, axis: _Axisparam = 0, keepdims: bool = False) -> Array:
        """See :py:func:`ndonnx.all` for documentation."""
        return Array._from_tyarray(self._tyarray.all(axis=axis, keepdims=keepdims))

    def any(self, axis: _Axisparam = 0, keepdims: bool = False) -> Array:
        """See :py:func:`ndonnx.any` for documentation."""
        return Array._from_tyarray(self._tyarray.any(axis=axis, keepdims=keepdims))


def _astyarray_or_pyscalar(
    val: int | float | str | Array | Var | np.ndarray,
) -> TyArrayBase | int | float | str:
    if isinstance(val, Array):
        return val._tyarray
    if isinstance(val, int | float | str):
        return val
    return tyfuncs.astyarray(val)


def _apply_op(
    lhs: PyScalar | Array,
    rhs: PyScalar | Array,
    op: Callable[[TyArrayBase | PyScalar, TyArrayBase | PyScalar], TyArrayBase],
) -> Array | NotImplementedType:
    lhs_ = _astyarray_or_pyscalar(lhs)
    rhs_ = _astyarray_or_pyscalar(rhs)
    data = op(lhs_, rhs_)
    if data is not NotImplemented:
        return Array._from_tyarray(data)

    return NotImplemented


def _normalize_arrays_in_key(key: GetItemKey) -> onnx.GetitemIndex:
    if isinstance(key, Array):
        if isinstance(key._tyarray.dtype, onnx.Bool):
            return key._tyarray.astype(onnx.bool_)

    if isinstance(key, int | slice | EllipsisType | Array | None):
        return _normalize_key_item(key)

    if isinstance(key, tuple):
        return tuple(_normalize_key_item(el) for el in key)

    raise IndexError(f"unexpected key type: `{type(key)}`")


def _normalize_key_item(
    item: int | slice | EllipsisType | Array | None,
) -> int | slice | EllipsisType | onnx.TyArrayInt64 | None:
    if isinstance(item, int | EllipsisType | None):
        return item

    if isinstance(item, Array):
        if isinstance(item.dtype, onnx.Integer):
            return item._tyarray.astype(onnx.int64)
        raise IndexError(
            f"indexing arrays must be of integer or boolean data type; found `{item.dtype}`"
        )

    def _normalize_slice_arg(el: int | Array | None) -> int | onnx.TyArrayInt64 | None:
        if isinstance(el, int | None):
            return el
        if not isinstance(el.dtype, onnx.Integer):
            IndexError(
                f"arrays in 'slice' objects must be of integer data types; found `{el.dtype}"
            )
        if el.ndim != 0:
            IndexError(f"arrays in 'slice' objects must be rank-0; found `{el.ndim}")
        return el._tyarray.astype(onnx.int64)

    if isinstance(item, slice):
        start = _normalize_slice_arg(item.start)
        stop = _normalize_slice_arg(item.stop)
        step = _normalize_slice_arg(item.step)
        return slice(start, stop, step)

    raise IndexError(f"invalid index key type: `{type(item)}`")
