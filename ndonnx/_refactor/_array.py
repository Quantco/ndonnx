# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import operator as std_ops
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from types import EllipsisType, NotImplementedType
from typing import TYPE_CHECKING, Any, Optional, Union, overload
from warnings import warn

import numpy as np
from spox import Var

from ._dtypes import DType
from ._typed_array import TyArrayBase
from ._typed_array.funcs import astyarray
from ._typed_array.indexing import normalize_getitem_key

if TYPE_CHECKING:
    from ._typed_array.indexing import SetitemIndex as TySetitemIndex

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


_BinaryOp = Callable[["Array", "int | bool | str | float | Array"], "Array"]


def _make_binary(
    tyarr_op: Callable[[TyArrayBase, TyArrayBase], TyArrayBase],
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

    @overload
    def __init__(self, *, shape: OnnxShape, dtype: DType): ...

    @overload
    def __init__(self, value: np.ndarray | float | int | bool): ...

    def __init__(self, value=None, *, shape=None, dtype=None):
        (is_shape, is_dtype, is_value) = (
            item is not None for item in [shape, dtype, value]
        )
        if not (
            (True, True, False) == (is_shape, is_dtype, is_value)
            or (False, False, True) == (is_shape, is_dtype, is_value)
            or (False, False, False) == (is_shape, is_dtype, is_value)
        ):
            raise ValueError("Invalid arguments.")

        if isinstance(shape, tuple) and isinstance(dtype, DType):
            self._tyarray = dtype._argument(shape)
            return
        if isinstance(value, np.ndarray):
            raise NotImplementedError
        if isinstance(value, int | float):
            ty_arr = astyarray(value, use_py_scalars=False, dtype=dtype)
            self._tyarray = ty_arr
            return

        raise NotImplementedError

    @classmethod
    def _from_tyarray(cls, tyarray: TyArrayBase, /) -> Array:
        if not isinstance(tyarray, TyArrayBase):
            raise TypeError(f"expected '_TypedArrayBase', found `{type(tyarray)}`")
        inst = cls.__new__(cls)
        inst._tyarray = tyarray
        return inst

    @property
    def device(self) -> None:
        return None

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
        static_dims = [el for el in self.shape if el is not None]
        if static_dims:
            return math.prod(static_dims)
        return None

    @property
    def T(self) -> Array:  # noqa: N802
        return Array._from_tyarray(self._tyarray.T)

    @property
    def null(self) -> None | Array:
        from .extensions import get_mask

        warn(
            "'Array.null' is deprecated in favor of 'ndonnx.extensions.get_mask'",
            DeprecationWarning,
        )
        return get_mask(self)

    @property
    def values(self) -> Array:
        from ._typed_array.masked_onnx import TyMaArray

        warn(
            "'Array.values' is deprecated in favor of 'ndonnx.extensions.get_data'",
            DeprecationWarning,
        )

        if isinstance(self._tyarray, TyMaArray):
            return Array._from_tyarray(self._tyarray.data)
        raise ValueError(f"`{self.dtype}` is not a nullable built-in type")

    def astype(self, dtype: DType, *, copy=True) -> Array:
        new_data = self._tyarray.astype(dtype, copy=copy)
        return Array._from_tyarray(new_data)

    def copy(self) -> Array:
        return Array._from_tyarray(self._tyarray.copy())

    def to_numpy(self) -> np.ndarray | None:
        from warnings import warn

        warn(
            "'Array.to_numpy' is deprecated in favor of 'Array.unwrap_numpy'",
            DeprecationWarning,
        )
        try:
            return self.unwrap_numpy()
        except ValueError:
            return None

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

    def disassemble(self) -> Mapping[str, Var] | Var:
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
        raise BufferError("ndonnx does not (yet) support the export of array data")

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
            "iteration requires dimension of static length, but dimension 0 is dynamic."
        )

    def __getitem__(self, key: GetitemIndex, /) -> Array:
        idx = normalize_getitem_key(key)
        data = self._tyarray[idx]
        return type(self)._from_tyarray(data)

    def __setitem__(
        self,
        key: SetitemIndex,
        value: Union[int, float, bool, Array],
        /,
    ) -> None:
        from ._typed_array import TyArrayBool, TyArrayInteger

        if isinstance(key, Array):
            if not isinstance(key._tyarray, TyArrayInteger | TyArrayBool):
                raise TypeError(
                    f"indexing array must have integer or boolean data type; found `{key.dtype}`"
                )
            idx: TySetitemIndex = key._tyarray
        else:
            idx = key

        # Specs say that the data type of self must not be changed.
        self._tyarray[idx] = asarray(value, dtype=self.dtype)._tyarray

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

    def __array_namespace__(self, /, *, api_version: Optional[str] = None) -> Any:
        import ndonnx._refactor as ndx

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
    __gt__, __rgt__ = _make_binary(std_ops.gt)
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
        shape = self._tyarray.shape
        return f"array({value_repr}, shape={shape}, dtype={self.dtype})"


NestedSequence = Sequence["Array | bool | int | float | NestedSequence"]


def asarray(
    obj: Array | bool | int | float | str | np.ndarray | NestedSequence | Var,
    /,
    *,
    dtype: DType | None = None,
    device=None,
    copy: bool | None = None,
) -> Array:
    if isinstance(obj, Var | str):
        obj = Array._from_tyarray(astyarray(obj))
        if dtype is None:
            return obj
        return obj.astype(dtype)
    elif isinstance(obj, Array):
        if copy is None:
            # We try to copy if possible.
            # TODO: Implement stricter failures cases according to standard
            copy = False
        if copy:
            obj = Array._from_tyarray(obj._tyarray.copy())
        if dtype:
            return obj.astype(dtype, copy=copy)
        return obj
    elif isinstance(obj, bool | int | float):
        obj = np.asarray(obj)
    elif isinstance(obj, np.ndarray):
        pass
    elif isinstance(obj, Sequence):
        return _asarray_sequence(obj, dtype=dtype, copy=copy)
    else:
        TypeError(f"Unexpected input type: `{type(obj)}`")
    data = astyarray(obj)
    if dtype:
        data = data.astype(dtype)
    return Array._from_tyarray(data)


def _asarray_sequence(
    seq: NestedSequence, dtype: DType | None, copy: bool | None
) -> Array:
    from ._funcs import concat

    arrays = [asarray(el, dtype=dtype, copy=copy) for el in seq]
    unwrapped = []
    for arr in arrays:
        try:
            unwrapped.append(arr.unwrap_numpy())
        except ValueError:
            break
    else:
        return asarray(np.asarray(unwrapped), dtype=dtype, copy=copy)
    return concat(arrays)


def _as_array(
    val: int | float | str | Array | Var | np.ndarray, use_py_scalars=False
) -> Array:
    if isinstance(val, Array):
        return val
    ty_arr = astyarray(val, use_py_scalars=use_py_scalars)
    return Array._from_tyarray(ty_arr)


def _apply_op(
    lhs: int | float | str | Array,
    rhs: int | float | str | Array,
    op: Callable[[TyArrayBase, TyArrayBase], TyArrayBase],
) -> Array | NotImplementedType:
    lhs = _as_array(lhs, use_py_scalars=True)
    rhs = _as_array(rhs, use_py_scalars=True)
    data = op(lhs._tyarray, rhs._tyarray)
    if data is not NotImplemented:
        return Array._from_tyarray(data)

    return NotImplemented