# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import operator
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from copy import copy as std_copy
from types import EllipsisType, NotImplementedType
from typing import TYPE_CHECKING, Literal, TypeGuard, TypeVar, cast, overload

import numpy as np
import spox
import spox._future
import spox.opset.ai.onnx.v21 as op
from spox import Tensor, Var, argument
from typing_extensions import Self

from .._dtypes import TY_ARRAY, DType, from_numpy
from .._schema import DTypeInfoV1
from . import ort_compat
from .indexing import (
    FancySlice,
    _key_to_indices,
    _move_ellipsis_back,
    normalize_getitem_key,
)
from .typed_array import TyArrayBase
from .utils import promote, safe_cast

if TYPE_CHECKING:
    from .._array import OnnxShape
    from .indexing import (
        GetitemIndex,
        GetitemTuple,
        SetitemIndex,
    )

TY_ARRAY_ONNX = TypeVar("TY_ARRAY_ONNX", bound="TyArray")
KEY = TypeVar("KEY", int, float, str)
VALUE = TypeVar("VALUE", int, float, str)


class _OnnxDType(DType[TY_ARRAY_ONNX]):
    """Data types with a direct representation in the ONNX standard."""

    def unwrap_numpy(self) -> np.dtype:
        if self == int8:
            return np.dtype("int8")
        if self == int16:
            return np.dtype("int16")
        if self == int32:
            return np.dtype("int32")
        if self == int64:
            return np.dtype("int64")

        if self == uint8:
            return np.dtype("uint8")
        if self == uint16:
            return np.dtype("uint16")
        if self == uint32:
            return np.dtype("uint32")
        if self == uint64:
            return np.dtype("uint64")

        if self == float16:
            return np.dtype("float16")
        if self == float32:
            return np.dtype("float32")
        if self == float64:
            return np.dtype("float64")

        if self == bool_:
            return np.dtype("bool")

        if self == utf8:
            # TODO: Migrate to numpy.StringDType
            return np.dtype(str)

        # Should never happen
        raise ValueError(f"'{self}' does not have a corresponding NumPy data type")

    @property
    @abstractmethod
    def _tyarr_class(self) -> type[TY_ARRAY_ONNX]: ...

    def __ndx_convert_tyarray__(self, arr: TyArrayBase) -> TY_ARRAY_ONNX:
        if isinstance(arr, TyArray):
            var = op.cast(arr.var, to=self.unwrap_numpy())
            return safe_cast(self._tyarr_class, ascoredata(var))
        return NotImplemented

    def _argument(self, shape: OnnxShape) -> TY_ARRAY_ONNX:
        var = argument(Tensor(self.unwrap_numpy(), shape))
        return self._tyarr_class(var)

    @property
    def _infov1(self) -> DTypeInfoV1:
        return DTypeInfoV1(
            author="ndonnx", type_name=self.__class__.__name__, meta=None
        )

    def _arange(
        self,
        start: int | float,
        stop: int | float,
        step: int | float = 1,
    ) -> TY_ARRAY_ONNX:
        np_dtype = self.unwrap_numpy()
        np_arr = np.array([start, stop, step])

        if np_dtype.kind != np_arr.dtype.kind:
            TypeError(
                f"'start', 'stop', and 'step' have to match data type kind of `{self}`"
            )
        # Get everything onto the same type
        start_, stop_, step_ = np.array([start, stop, step], dtype=np_dtype)

        var = ort_compat.range(
            op.const(start_, np_dtype),
            op.const(stop_, np_dtype),
            op.const(step_, np_dtype),
        )

        return self._tyarr_class(var)

    def _eye(
        self,
        n_rows: int,
        n_cols: int | None = None,
        /,
        *,
        k: int = 0,
    ) -> TY_ARRAY_ONNX:
        n_cols = n_rows if n_cols is None else n_cols
        # There is no adequate ONNX operator
        mat = np.eye(n_rows, n_cols, k=k)
        var = op.const(mat)

        return self._tyarr_class(var)

    def _ones(self, shape: tuple[int, ...] | TyArrayInt64) -> TY_ARRAY_ONNX:
        np_dtype = self.unwrap_numpy()
        scalar = self._tyarr_class(op.const(1, np_dtype))

        return scalar.broadcast_to(shape)

    def _zeros(self, shape: tuple[int, ...] | TyArrayInt64) -> TY_ARRAY_ONNX:
        np_dtype = self.unwrap_numpy()
        scalar = self._tyarr_class(op.const(0, np_dtype))

        return scalar.broadcast_to(shape)


class _Number(_OnnxDType[TY_ARRAY_ONNX]):
    def _result_type(self, rhs: DType) -> DType | NotImplementedType:
        if isinstance(self, NumericDTypes) and isinstance(rhs, NumericDTypes):
            return _result_type_core_numeric(self, rhs)

        return NotImplemented


class Utf8(_OnnxDType["TyArrayUtf8"]):
    def _result_type(self, rhs: DType) -> DType | NotImplementedType:
        if self == rhs:
            return self
        return NotImplemented

    @property
    def _tyarr_class(self) -> type[TyArrayUtf8]:
        return TyArrayUtf8

    def _zeros(self, shape: tuple[int, ...] | TyArrayInt64) -> TyArrayUtf8:
        # NumPy returns empty strings for `zeros` with string data
        # types. We follow that lead.
        np_dtype = self.unwrap_numpy()
        scalar = self._tyarr_class(op.const("", np_dtype))

        return scalar.broadcast_to(shape)


class Boolean(_OnnxDType["TyArrayBool"]):
    def _result_type(self, rhs: DType) -> DType | NotImplementedType:
        if self == rhs:
            return self
        return NotImplemented

    @property
    def _tyarr_class(self) -> type[TyArrayBool]:
        return TyArrayBool


class Integer(_Number[TY_ARRAY_ONNX]): ...


class Floating(_Number[TY_ARRAY_ONNX]): ...


class Int8(Integer["TyArrayInt8"]):
    @property
    def _tyarr_class(self) -> type[TyArrayInt8]:
        return TyArrayInt8


class Int16(Integer["TyArrayInt16"]):
    @property
    def _tyarr_class(self) -> type[TyArrayInt16]:
        return TyArrayInt16


class Int32(Integer["TyArrayInt32"]):
    @property
    def _tyarr_class(self) -> type[TyArrayInt32]:
        return TyArrayInt32


class Int64(Integer["TyArrayInt64"]):
    @property
    def _tyarr_class(self) -> type[TyArrayInt64]:
        return TyArrayInt64


class UInt8(Integer["TyArrayUInt8"]):
    @property
    def _tyarr_class(self) -> type[TyArrayUInt8]:
        return TyArrayUInt8


class UInt16(Integer["TyArrayUInt16"]):
    @property
    def _tyarr_class(self) -> type[TyArrayUInt16]:
        return TyArrayUInt16


class UInt32(Integer["TyArrayUInt32"]):
    @property
    def _tyarr_class(self) -> type[TyArrayUInt32]:
        return TyArrayUInt32


class UInt64(Integer["TyArrayUInt64"]):
    @property
    def _tyarr_class(self) -> type[TyArrayUInt64]:
        return TyArrayUInt64


class Float16(Floating["TyArrayFloat16"]):
    @property
    def _tyarr_class(self) -> type[TyArrayFloat16]:
        return TyArrayFloat16


class Float32(Floating["TyArrayFloat32"]):
    @property
    def _tyarr_class(self) -> type[TyArrayFloat32]:
        return TyArrayFloat32


class Float64(Floating["TyArrayFloat64"]):
    @property
    def _tyarr_class(self) -> type[TyArrayFloat64]:
        return TyArrayFloat64


# Non-nullable Singleton instances
bool_: Boolean = Boolean()

float16: Float16 = Float16()
float32: Float32 = Float32()
float64: Float64 = Float64()

int16: Int16 = Int16()
int32: Int32 = Int32()
int64: Int64 = Int64()
int8: Int8 = Int8()

uint8: UInt8 = UInt8()
uint16: UInt16 = UInt16()
uint32: UInt32 = UInt32()
uint64: UInt64 = UInt64()

utf8: Utf8 = Utf8()

# Union types
#
# Union types are exhaustive and don't create ambiguities with respect to user-defined subtypes.
FloatingDTypes = Float16 | Float32 | Float64
SignedIntegerDTypes = Int8 | Int16 | Int32 | Int64
UnsignedIntegerDTypes = UInt8 | UInt16 | UInt32 | UInt64
IntegerDTypes = SignedIntegerDTypes | UnsignedIntegerDTypes
NumericDTypes = FloatingDTypes | IntegerDTypes
DTypes = NumericDTypes | Utf8 | Boolean


class TyArray(TyArrayBase):
    dtype: _OnnxDType
    var: Var

    def __init__(self, var: Var):
        var_dtype = var.unwrap_tensor().dtype
        if from_numpy(var_dtype) != self.dtype:
            raise ValueError(
                f"data type of 'var' (`{var_dtype}`) is incompatible with `{type(self)}`"
            )
        if var.unwrap_tensor().shape is None:
            raise ValueError("'var' has no shape information")
        self.var = var

    def __ndx_value_repr__(self) -> dict[str, str]:
        try:
            return {"data": str(self.unwrap_numpy().tolist())}
        except ValueError:
            return {"data": "*lazy*"}

    def __getitem__(self, key: GetitemIndex) -> Self:
        key = normalize_getitem_key(key)

        if isinstance(key, bool):
            key = TyArrayBool(op.const(key))
        if not isinstance(key, TyArray):
            return self._getitem_static(key)
        if isinstance(key, TyArrayBool):
            return self._getitem_boolmask(key)
        raise IndexError(f"Cannot index with key `{key}`")

    def _getitem_static(self, key: GetitemTuple) -> Self:
        if isinstance(key, list):
            # Technically not supported, but a common pattern
            key = tuple(key)
        if isinstance(key, int | slice | EllipsisType | None):
            key = (key,)

        # Validation
        cnt_slice_or_idx = len(list(el for el in key if isinstance(el, int | slice)))

        if key.count(...) == 0:
            if self.ndim != cnt_slice_or_idx:
                raise IndexError(
                    "length of 'key' excluding 'None' values must match array's rank or contain an ellipsis (`...`)"
                )
            # Ensure that we always have a tailing ellipsis to have fewer special cases
            key = tuple(list(key) + [...])

        cnt_ellipsis_fills = self.ndim - cnt_slice_or_idx

        ellipsis_pos = key.index(...)
        pre_ellipsis = list(key[:ellipsis_pos])
        post_ellipsis = list(key[ellipsis_pos + 1 :])
        no_more_ellipsis = cast(
            list[int | slice | None],
            pre_ellipsis
            + [slice(None, None) for _ in range(cnt_ellipsis_fills)]
            + post_ellipsis,
        )

        new_axis_positions = [i for i, el in enumerate(no_more_ellipsis) if el is None]

        var = self.var
        if new_axis_positions:
            var = op.unsqueeze(self.var, op.const(new_axis_positions, np.int64))

        # Remove `None` values
        indices_and_slices_only = [
            slice(None) if el is None else el for el in no_more_ellipsis
        ]
        if all(el == slice(None) for el in indices_and_slices_only):
            # Early return if we only expanded some dims
            return type(self)(var)

        fancy_slices = [FancySlice(el) for el in indices_and_slices_only]
        kwargs = _to_slice_kwargs(fancy_slices)
        if kwargs:
            # Some slices are not noops, so we do the slicing
            var = op.slice(var, **kwargs)
            squeeze_positions = [i for i, s in enumerate(fancy_slices) if s.squeeze]

            if squeeze_positions:
                var = op.squeeze(var, op.const(squeeze_positions, np.int64))

        return type(self)(var)

    def _getitem_boolmask(self, key: TyArrayBool) -> Self:
        if not all(ks in (xs, 0) for xs, ks in zip(self.shape, key.shape)):
            raise IndexError("Shape of 'key' is incompatible with shape of 'self'")
        if key.ndim > self.ndim:
            raise IndexError(
                "rank of 'key' (`{key.ndim}`) must be less or equal to rank of 'self' (`{self.ndim}`)"
            )
        if key.ndim == 0:
            key = key[None, ...]
            x = self[None, ...]
            res = op.compress(x.var, key.var, axis=0)
            return type(self)(res)

        # Can't use reshape since the use of -1 for zero dimensional
        # arrays is undefined
        in_shape = (
            self.dynamic_shape[: key.ndim]
            .prod(keepdims=True, dtype=int64)
            .concat([self.dynamic_shape[key.ndim :]], axis=0)
        )
        var = self.reshape(in_shape).var

        res = op.compress(var, key.reshape((-1,)).var, axis=0)
        return type(self)(res)

    def __setitem__(
        self,
        key: SetitemIndex,
        value: Self,
        /,
    ) -> None:
        if key == ():
            self.var = value.broadcast_to(self.dynamic_shape).var
            return

        if isinstance(key, list):
            # Technically not supported, but a common pattern
            key = tuple(key)
        if isinstance(key, int | slice | EllipsisType):
            key = (key,)

        if isinstance(key, TyArrayBool):
            return self._setitem_boolmask(key, value)
        elif isinstance(key, TyArrayInteger):
            raise IndexError("'__setitem__' with integer arrays is not supported.")

        # Shortcut: Remove ellipsis from the front if there is any
        if ... in key and key.index(...) == len(key):
            key = key[:-1]

        if contains_no_ellipsis(key):
            arr_key = _key_to_indices(key, self.dynamic_shape)
            self._setitem_int_array(arr_key, value)
            return

        # The ellipsis is somewhere else in the key. We need to do some rearangment
        tmp_dim_perm, reverse_perm, keys_no_ellips = _move_ellipsis_back(self.ndim, key)
        perm_self = self.permute_dims(tmp_dim_perm)
        perm_self[keys_no_ellips] = value
        self.var = perm_self.permute_dims(reverse_perm).var

    def _setitem_boolmask(self, key: TyArrayBool, value: Self) -> None:
        if self.ndim < key.ndim:
            raise IndexError("provided boolean mask has higher rank than 'self'")
        if self.ndim > key.ndim:
            diff = self.ndim - key.ndim
            # expand to rank of self
            idx = [...] + diff * [None]
            key = key[tuple(idx)]

        self.var = ort_compat.where(key.var, value.var, self.var)
        return

    def _setitem_int_array(self, key: TyArrayInt64, value: Self) -> None:
        # The last dim of shape must be statically known (i.e. the
        # length of the indexing tuple
        idx_tuple_len = key.shape[-1]
        if not isinstance(idx_tuple_len, int):
            raise IndexError("Length of indexing tuple must be statically known")
        value_shape = key.dynamic_shape[:-1].concat(
            [self.dynamic_shape[idx_tuple_len:]], axis=0
        )
        # We may have to broadcast `value` into a higher rank, or we
        # have to reshape it into a lower one
        (target_rank,) = value_shape.shape
        if not isinstance(target_rank, int):
            raise IndexError("target rank must be statically known")
        if target_rank < value.ndim:
            value_correct_shape = value.reshape(value_shape)
        else:
            value_correct_shape = value.broadcast_to(value_shape)
        self.var = op.scatter_nd(self.var, key.var, value_correct_shape.var)
        return

    @property
    def dynamic_shape(self) -> TyArrayInt64:
        from .funcs import astyarray

        # Spox does not attempt value propagation for lazy
        # arrays. Thus, we never get a constant from lazy input, even
        # if the shape is statically known. We therefore do that
        # manually here.
        if all(isinstance(el, int) for el in self.shape):
            return astyarray(np.array(self.shape), dtype=int64)

        var = op.shape(self.var)
        return TyArrayInt64(var)

    @property
    def dynamic_size(self) -> TyArrayInt64:
        var = op.size(self.var)
        return TyArrayInt64(var)

    @property
    def mT(self) -> Self:  # noqa: N802
        if self.ndim < 2:
            raise ValueError(
                f"array must have two or more dimensions, found `{self.ndim}`"
            )
        dims = list(range(self.ndim))
        dims = dims[:-2] + dims[-2:][::-1]
        var = op.transpose(self.var, perm=dims)
        return type(self)(var)

    @property
    def shape(self) -> OnnxShape:
        shape = self.var.unwrap_tensor().shape
        if shape is None:
            raise ValueError("Missing shape information")
        if any(isinstance(el, None | str) for el in shape):
            try:
                np_arr = self.unwrap_numpy()
            except ValueError:
                return shape
            # check that things are compatible
            static_shape = []
            for np_el, ndx_el in zip(np_arr.shape, shape):
                if isinstance(ndx_el, None | str):
                    static_shape.append(np_el)
                    continue
                elif np_el != ndx_el:
                    raise ValueError("propagated shape disagrees with propagated value")
                static_shape.append(np_el)
            return tuple(static_shape)

        return shape

    def unwrap_numpy(self) -> np.ndarray:
        if self.var._value is not None:
            np_arr = np.asarray(self.var._value.value)
            np_arr.astype(np.dtypes.StringDType)
            if np_arr.dtype == self.dtype.unwrap_numpy():
                return np_arr
            if self.dtype.unwrap_numpy().kind == "U" and np_arr.dtype.kind in [
                "U",
                "O",
            ]:
                # String-case
                # TODO: Where does the "object" kind come from?
                # Probably spox; we should have a more predictable
                # upstream string support.
                return np_arr.astype(str)

            # Should not happen
            raise ValueError(f"unexpected value data type: `{np_arr.dtype}`")

        raise ValueError("no propagated value available")

    def all(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> TyArrayBool:
        if axis == ():
            return self.astype(bool_)

        if isinstance(axis, int):
            axis = (axis,)

        bools = self.astype(bool_)

        if bools.ndim == 0:
            if axis:
                ValueError("'axis' were provided but 'self' is a scalar")
            # Nothing left to reduce
            return safe_cast(TyArrayBool, bools)

        axes = op.const(list(axis), np.int64) if axis else None

        # Note: reduce_min, which would support uint8s, appears to be buggy on the onnxruntime
        # side. Thus we use reduce_prod for now.
        var = ort_compat.reduce_prod(
            bools.astype(int32).var, axes=axes, keepdims=keepdims
        )
        return safe_cast(TyArrayBool, TyArrayInt32(var).astype(bool_))

    def any(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> TyArrayBool:
        if axis == ():
            return self.astype(bool_)
        if isinstance(axis, int):
            axis = (axis,)

        bools = self.astype(bool_)

        if bools.ndim == 0:
            if axis:
                ValueError("'axis' were provided but 'self' is a scalar")
            # Nothing left to reduce
            return safe_cast(TyArrayBool, bools)

        axes = op.const(list(axis), np.int64) if axis else None

        # Accumulate in float32 to avoid possible overflowing issues
        var = ort_compat.reduce_sum(
            bools.astype(float32).var, axes=axes, keepdims=keepdims
        )
        return safe_cast(TyArrayBool, TyArrayFloat32(var).astype(bool_))

    def as_core_dtype(self, dtype: DTypes) -> TyArray:
        raise ValueError(f"Casting between `{self.dtype}` and `{dtype}` is undefined")

    def broadcast_to(self, shape: tuple[int, ...] | TyArrayInt64) -> Self:
        from .funcs import astyarray

        if isinstance(shape, tuple):
            shape = astyarray(np.array(shape), dtype=int64)
        if shape.ndim != 1:
            raise ValueError(f"'shape' must be a rank-1 tensor, found `{shape.ndim}`")
        if not isinstance(shape.shape[0], int):
            raise ValueError("'shape' must be a rank-1 tensor with static length.")
        (target_rank,) = shape.shape
        if not isinstance(target_rank, int):
            raise ValueError(
                "target rank of broadcasting operation must be statically known"
            )
        if target_rank < self.ndim:
            raise ValueError("target rank must equal or greater than rank of 'self'")
        var = op.expand(self.var, shape.var)
        return type(self)(var)

    def concat(self, others: Sequence[Self], axis: None | int) -> Self:
        others = list(others)
        arrays = [self] + others
        if axis is None:
            arrays = [a if a.ndim == 1 else a.reshape((-1,)) for a in arrays]
            axis = 0

        seen_ranks = {a.ndim for a in arrays}
        if len(seen_ranks) != 1:
            raise ValueError("concatenated arrays must all have the same rank")
        (rank,) = seen_ranks
        if rank == 0:
            raise ValueError("zero-dimensional arrays cannot be concatenated")

        if len(arrays) == 1:
            return arrays[0].copy()

        # It seems that there is currently a bug(?) in the type/shape
        # inference in ONNX which prohibits us from concatenating
        # empty 1D int32 and int64 arrays (see test case). We therefor
        # do some hacky special-casing here for those types and those
        # types only. Other data types are fine.
        #
        # TODO: File upstream bug; this may also be what caused the
        # segfaults in onnxruntime in the past!
        if self.dtype in (int32, int64) and self.ndim == 1:
            axis_ = axis if axis is None else axis
            dummy_axis = op.const([axis_ + 1], dtype=np.int64)
            vars = [op.unsqueeze(a.var, dummy_axis) for a in arrays]
            var = op.concat(vars, axis=axis_)
            var = op.squeeze(var, dummy_axis)
        else:
            var = op.concat([a.var for a in arrays], axis=0 if axis is None else axis)
        return type(self)(var)

    def copy(self) -> Self:
        return std_copy(self)

    def disassemble(self) -> Var:
        return self.var

    def permute_dims(self, axes: tuple[int, ...]) -> Self:
        var = op.transpose(self.var, perm=axes)
        return type(self)(var)

    def reshape(self, shape: tuple[int, ...] | TyArrayInt64) -> Self:
        if isinstance(shape, tuple | list):
            var = op.reshape(self.var, op.const(shape, np.int64), allowzero=True)
        else:
            var = op.reshape(self.var, shape.var, allowzero=True)
        return type(self)(var)

    def repeat(
        self, repeats: int | TyArrayInt64, /, *, axis: int | None = None
    ) -> Self:
        raise NotImplementedError

    def searchsorted(
        self,
        x2: Self,
        /,
        *,
        side: Literal["left", "right"] = "left",
        sorter: TyArrayInteger | None = None,
    ) -> TyArrayInt64:
        x1 = self
        if x1.ndim != 1:
            raise TypeError("x1 must be a 1-dimensional array")
        if sorter is not None:
            x1 = x1.take(sorter.astype(int64, copy=False))

        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")

        x1 = x1[:, None]
        x2 = x2[None, :]

        if side == "left":
            indices = (x1 < x2).astype(int64).sum(axis=0)
        else:
            indices = (x1 <= x2).astype(int64).sum(axis=0)

        return safe_cast(TyArrayInt64, indices)

    def squeeze(self, /, axis: int | tuple[int, ...]) -> Self:
        if isinstance(axis, int):
            axis = (axis,)
        if axis == ():
            return self.copy()
        try:
            res = op.squeeze(self.var, op.const(axis, np.int64))
        except Exception as e:
            # Error in shape inference caught via static shape parameters
            if "inference error" in str(e).lower():
                raise ValueError(
                    f"cannot squeeze array of shape `{self.shape}` on axis `{axis}`"
                )
            raise e
        return type(self)(res)

    def take(self, indices: TyArrayInt64, /, *, axis: int | None = None) -> Self:
        axis = axis or 0
        var = op.gather(self.var, indices.var, axis=axis)
        return type(self)(var)

    def tile(self, repetitions: tuple[int, ...], /) -> Self:
        n = self.ndim
        m = len(repetitions)
        x = self
        if n > m:
            repetitions = tuple([1] * (n - m) + list(repetitions))
        elif n < m:
            x = self[tuple([None] * (m - n) + [...])]

        var = op.tile(x.var, op.const(repetitions, np.int64))
        return type(self)(var)

    def _tile_array(self, repetitions: TyArrayInt64, /) -> Self:
        if not repetitions.ndim == 1:
            raise ValueError(
                f"'repetitions' has rank `{repetitions.ndim}` but must be '1'"
            )

        try:
            reps = repetitions.unwrap_numpy()
            return self.tile(tuple(reps))
        except ValueError:
            raise NotImplementedError(
                "'tile' does not yet support repetition via dynamic arrays "
            )

    def tril(self, /, *, k: int = 0) -> Self:
        var = ort_compat.trilu(self.var, k=op.const(k), upper=False)
        return type(self)(var)

    def triu(self, /, *, k: int = 0) -> Self:
        var = ort_compat.trilu(self.var, k=op.const(k), upper=True)
        return type(self)(var)

    def unique_all(self) -> tuple[Self, TyArrayInt64, TyArrayInt64, TyArrayInt64]:
        flattened = self.reshape((-1,))
        res = ort_compat.unique(flattened.var, sorted=True)
        values, indices, inverse_indices, counts = (
            type(self)(res[0]),
            TyArrayInt64(res[1]),
            TyArrayInt64(res[2]),
            TyArrayInt64(res[3]),
        )

        inverse_indices = inverse_indices.reshape(self.dynamic_shape)
        counts = counts.reshape(values.dynamic_shape)

        return (values, indices, inverse_indices, counts)

    def __ndx_astype__(self, dtype: DType[TY_ARRAY]) -> TY_ARRAY:
        # We pretend that we don't know about any other data type. We
        # delegate all work to ``DType.__ndx_convert_tyarray__``
        return NotImplemented

    def __eq__(self, other) -> TyArrayBool | NotImplementedType:  # type: ignore[override]
        res = _promote_and_apply_op(
            self, other, operator.eq, ort_compat.equal, forward=True
        )
        if isinstance(res, NotImplementedType):
            return res
        return safe_cast(TyArrayBool, res)

    def _eqcomp(self, other) -> TyArrayBase:
        return _promote_and_apply_op(self, other, operator.eq, ort_compat.equal, True)

    def clip(
        self, min: TyArrayBase | None = None, max: TyArrayBase | None = None
    ) -> Self:
        if min is None and max is None:
            return self.copy()

        # The standard says that min/max must not change the output type.
        min_ = None if min is None else min.astype(self.dtype).var
        max_ = None if max is None else max.astype(self.dtype).var

        var = ort_compat.clip(self.var, min_, max_)

        return type(self)(var)

    def static_map(self, mapping: Mapping[KEY, VALUE], default: VALUE) -> TyArray:
        """Map values in ``self`` based on the static ``mapping``.

        Parameters
        ----------
        mapping
            A mapping from keys to values. The keys must be of the
            same type as the values in ``x``. NaN-keys compare true to
            values in ``self``.
        default
            The default value to use when a key is not found in the mapping.

        Returns
        -------
        A new Array with the values mapped according to the mapping.
        """
        np_dtype = self.dtype.unwrap_numpy()
        keys = np.array(list(mapping.keys()), dtype=np_dtype)
        values = np.array(list(mapping.values()))
        if values.dtype.kind in ("O", "U"):
            values = values.astype(str)
            # Don't use values.dtype to cast the default since it may
            # have a length associated with it!
            default_ = np.array([default], dtype=str)
        else:
            default_ = np.array([default], dtype=values.dtype)

        var = ort_compat.label_encoder(
            self.var, keys_tensor=keys, values_tensor=values, default_tensor=default_
        )

        return ascoredata(var)

    @overload
    def __ndx_maximum__(self, rhs: TyArray, /) -> TyArray | NotImplementedType: ...

    @overload
    def __ndx_maximum__(
        self, rhs: TyArrayBase, /
    ) -> TyArrayBase | NotImplementedType: ...

    def __ndx_maximum__(self, rhs: TyArrayBase, /) -> TyArrayBase | NotImplementedType:
        if isinstance(rhs, TyArray):
            lhs, rhs = promote(self, rhs)
            var = ort_compat.max([lhs.var, rhs.var])
            return type(lhs)(var)

        return NotImplemented

    @overload
    def __ndx_minimum__(self, rhs: TyArray, /) -> TyArray | NotImplementedType: ...

    @overload
    def __ndx_minimum__(
        self, rhs: TyArrayBase, /
    ) -> TyArrayBase | NotImplementedType: ...

    def __ndx_minimum__(self, rhs: TyArrayBase, /) -> TyArrayBase | NotImplementedType:
        if isinstance(rhs, TyArray):
            lhs, rhs = promote(self, rhs)
            var = ort_compat.min([lhs.var, rhs.var])
            return type(lhs)(var)

        return NotImplemented

    @overload
    def __ndx_where__(
        self, cond: TyArrayBool, y: TyArray, /
    ) -> TyArray | NotImplementedType: ...

    @overload
    def __ndx_where__(
        self, cond: TyArrayBool, y: TyArrayBase, /
    ) -> TyArrayBase | NotImplementedType: ...

    def __ndx_where__(
        self, cond: TyArrayBool, y: TyArrayBase, /
    ) -> TyArrayBase | NotImplementedType:
        if isinstance(y, TyArray):
            x, y = promote(self, y)
            var = ort_compat.where(cond.var, x.var, y.var)
            return type(x)(var)

        return NotImplemented


class TyArrayUtf8(TyArray):
    dtype = utf8

    def __add__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.add, op.string_concat, forward=True
        )


class TyArrayNumber(TyArray):
    # We piggyback here on the numeric implementations for booleans
    # even though the standard does not use this relation ship between
    # number/int and bool. (see `isdtype` for details).
    dtype: _Number | Boolean

    def _arg_minmax(
        self, spox_op, /, *, axis: int | None = None, keepdims: bool = False
    ) -> TyArrayInt64:
        x = self
        axis_ = axis
        if axis_ is None:
            x = x.reshape((-1,))
            axis_ = 0
        res = TyArrayInt64(spox_op(x.var, axis=axis_, keepdims=keepdims))
        if axis is None:
            if keepdims:
                res = res.reshape(tuple(1 for _ in range(self.ndim)))
            # else:
            #     res = res.squeeze(0)

        return res

    def argmax(
        self, /, *, axis: int | None = None, keepdims: bool = False
    ) -> TyArrayInt64:
        return self._arg_minmax(ort_compat.arg_max, axis=axis, keepdims=keepdims)

    def argmin(
        self, /, *, axis: int | None = None, keepdims: bool = False
    ) -> TyArrayInt64:
        return self._arg_minmax(ort_compat.arg_min, axis=axis, keepdims=keepdims)

    def argsort(
        self, /, *, axis: int = -1, descending: bool = False, stable: bool = True
    ) -> TyArrayInt64:
        if axis < 0:
            axis += self.ndim
        k = self.dynamic_shape[axis : axis + 1].var  # k must be 1D, thus the slice
        _values, indices = ort_compat.top_k(
            self.var, k, axis=axis, largest=descending, sorted=stable
        )
        return TyArrayInt64(indices)

    def cumulative_sum(
        self,
        /,
        *,
        axis: int | None = None,
        dtype: DType | None = None,
        include_initial: bool = False,
    ) -> TyArrayBase:
        from .funcs import astyarray

        dtype_ = self._accumulation_dtype(dtype)
        if self.dtype != dtype_:
            return self.astype(dtype_).cumulative_sum(
                axis=axis, include_initial=include_initial, dtype=dtype_
            )

        if axis is None:
            if self.ndim > 1:
                raise ValueError(
                    "'axis' parameter must be provided for arrays with more than one dimension."
                )
            axis_ = 0
        else:
            axis_ = axis

        out = safe_cast(
            type(self),
            ascoredata(
                ort_compat.cumsum(
                    self.var,
                    op.const(axis_, np.int64),
                    exclusive=False,
                )
            ),
        )
        if include_initial:
            out_shape = out.dynamic_shape
            out_shape[axis_] = safe_cast(TyArrayInt64, astyarray(1))
            zeros = safe_cast(
                type(self), astyarray(0, out.dtype).broadcast_to(out_shape)
            )
            out = zeros.concat([out], axis=axis_)
        return out

    def _accumulation_dtype(self, dtype: DType | None = None) -> DType:
        if dtype is None:
            if isinstance(self.dtype, SignedIntegerDTypes):
                dtype_: DType = int64
            elif isinstance(self.dtype, UnsignedIntegerDTypes):
                dtype_ = uint64
            else:
                dtype_ = self.dtype
        else:
            dtype_ = dtype
        return dtype_

    def _reduce(
        self,
        pub_name: str,
        var_op: Callable[..., Var],
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
    ) -> TyArrayBase:
        dtype_ = dtype or self._accumulation_dtype(dtype)
        if self.dtype != dtype_:
            member_after_cast = getattr(self.astype(dtype_), pub_name)
            return member_after_cast(axis=axis, keepdims=keepdims, dtype=dtype_)

        # Early return if there is nothing left to do. We deliberately
        # do this after the casting.
        if axis == ():
            return self.copy()

        axis_ = _axis_var(axis, self.ndim)
        var = var_op(
            self.var,
            axis_,
            keepdims=keepdims,
            noop_with_empty_axes=False,
        )
        return ascoredata(var)

    def max(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Self:
        if axis == ():
            # TODO: File bug:
            # `reduce_max`'s ORT implementation differs from onnx-shape
            # inference for keepdims=False AND noop_with_empty_axes=True.
            return self.copy()

        axes = _axis_var(axis, self.ndim)
        return type(self)(
            ort_compat.reduce_max(
                self.var, axes=axes, keepdims=keepdims, noop_with_empty_axes=False
            )
        )

    def min(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Self:
        if axis == ():
            return self.copy()

        axes = _axis_var(axis, self.ndim)
        return type(self)(
            ort_compat.reduce_min(
                self.var, axes=axes, keepdims=keepdims, noop_with_empty_axes=False
            )
        )

    @overload
    def prod(
        self,
        /,
        *,
        dtype: DType[TY_ARRAY],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> TY_ARRAY: ...

    @overload
    def prod(
        self,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
    ) -> TyArrayBase: ...

    def prod(
        self,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
    ) -> TyArrayBase:
        return self._reduce(
            "prod", ort_compat.reduce_prod, axis=axis, dtype=dtype, keepdims=keepdims
        )

    def sort(
        self, /, *, axis: int = -1, descending: bool = False, stable: bool = True
    ) -> Self:
        if axis < 0:
            axis += self.ndim
        k = self.dynamic_shape[axis : axis + 1].var  # k must be 1D, thus the slice
        values, _indices = ort_compat.top_k(
            self.var, k, axis=axis, largest=descending, sorted=stable
        )
        return type(self)(values)

    @overload
    def sum(
        self,
        /,
        *,
        dtype: DType[TY_ARRAY],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> TY_ARRAY: ...

    @overload
    def sum(
        self,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
    ) -> TyArrayBase: ...

    def sum(
        self,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
    ) -> TyArrayBase:
        return self._reduce(
            "sum", ort_compat.reduce_sum, axis=axis, dtype=dtype, keepdims=keepdims
        )

    def __abs__(self) -> Self:
        return type(self)(op.abs(self.var))

    def __neg__(self) -> TyArray:
        return type(self)(op.neg(self.var))

    def __pos__(self) -> Self:
        return self.copy()

    def __add__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.add, ort_compat.add, forward=True
        )

    def __radd__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.add, ort_compat.add, forward=False
        )

    def __ge__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.ge, ort_compat.greater_or_equal, forward=True
        )

    def __gt__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.gt, ort_compat.greater, forward=True
        )

    def __le__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.le, ort_compat.less_or_equal, forward=True
        )

    def __lt__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.lt, ort_compat.less, forward=True
        )

    def __mul__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.mul, ort_compat.mul, forward=True
        )

    def __rmul__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.mul, ort_compat.mul, forward=False
        )

    def __pow__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.pow, ort_compat.pow, forward=True
        )

    def __rpow__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.pow, ort_compat.pow, forward=False
        )

    def __sub__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.sub, ort_compat.sub, forward=True
        )

    def __rsub__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.sub, ort_compat.sub, forward=False
        )

    def __truediv__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.truediv, ort_compat.div, forward=True
        )

    def __rtruediv__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.truediv, ort_compat.div, forward=False
        )

    def __floordiv__(self, other) -> TyArrayBase:
        promo_result, _ = promote(self, other)
        return (self / other).floor().astype(promo_result.dtype)

    def __rfloordiv__(self, other) -> TyArrayBase:
        promo_result, _ = promote(self, other)
        return (other / self).floor().astype(promo_result.dtype)

    def nonzero(self) -> tuple[TyArrayInt64, ...]:
        if self.ndim == 0:
            raise ValueError("'nonzero' is not defined for scalar arrays")

        res = TyArrayInt64(ort_compat.non_zero(self.var))
        out = []
        for i in range(self.ndim):
            out.append(res[i, :])
        return tuple(out)

    def sign(self) -> Self:
        return type(self)(ort_compat.sign(self.var))

    def sqrt(self) -> Self:
        return type(self)(ort_compat.sqrt(self.var))

    def square(self) -> TyArray:
        from .funcs import astyarray

        two = astyarray(2, use_py_scalars=True)
        return safe_cast(TyArray, self**two)


class TyArrayInteger(TyArrayNumber):
    dtype: Integer

    def __truediv__(self, rhs, /) -> TyArrayBase:
        # Casting rules are implementation defined. We default to float64 like NumPy
        if isinstance(rhs, TyArrayInteger):
            return self.astype(float64) / rhs.astype(float64)
        return super().__truediv__(rhs)

    def __rtruediv__(self, lhs, /) -> TyArrayBase:
        # Casting rules are implementation defined. We default to float64 like NumPy
        if isinstance(lhs, TyArrayInteger):
            return lhs.astype(float64) / self.astype(float64)
        return super().__rtruediv__(lhs)

    def __and__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.and_, op.bitwise_and, forward=True
        )

    def __rand__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.and_, op.bitwise_and, forward=False
        )

    def __or__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.or_, op.bitwise_or, forward=True
        )

    def __ror__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.or_, op.bitwise_or, forward=False
        )

    def __mod__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.mod, lambda a, b: op.mod(a, b, fmod=0), forward=True
        )

    def __rmod__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.mod, lambda a, b: op.mod(a, b, fmod=0), forward=False
        )

    def __xor__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.xor, op.bitwise_xor, forward=True
        )

    def __rxor__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.xor, op.bitwise_xor, forward=False
        )

    def __lshift__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self,
            other,
            operator.lshift,
            lambda a, b: ort_compat.bit_shift(a, b, direction="LEFT"),
            forward=True,
        )

    def __rlshift__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self,
            other,
            operator.lshift,
            lambda a, b: ort_compat.bit_shift(a, b, direction="LEFT"),
            forward=False,
        )

    def __rshift__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self,
            other,
            operator.rshift,
            lambda a, b: ort_compat.bit_shift(a, b, direction="RIGHT"),
            forward=True,
        )

    def __rrshift__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self,
            other,
            operator.rshift,
            lambda a, b: ort_compat.bit_shift(a, b, direction="RIGHT"),
            forward=False,
        )

    def __invert__(self) -> Self:
        res = op.bitwise_not(self.var)
        return type(self)(res)

    def ceil(self) -> Self:
        return self.copy()

    def floor(self) -> Self:
        return self.copy()

    def round(self) -> Self:
        return self.copy()

    def isfinite(self) -> TyArrayBool:  # type: ignore
        return TyArrayBool(op.const(True)).broadcast_to(self.dynamic_shape)

    def isnan(self) -> TyArrayBool:  # type: ignore
        return TyArrayBool(op.const(False)).broadcast_to(self.dynamic_shape)

    def isinf(self) -> TyArrayBool:  # type: ignore
        return TyArrayBool(op.const(False)).broadcast_to(self.dynamic_shape)

    def trunc(self) -> Self:
        return self.copy()


class TyArraySignedInteger(TyArrayInteger): ...


class TyArrayUnsignedInteger(TyArrayInteger):
    def __abs__(self) -> Self:
        return self.copy()

    def __pos__(self) -> Self:
        return self.copy()

    def __neg__(self) -> TyArray:
        if isinstance(self.dtype, UInt8):
            x: TyArray = self.astype(int8)
        elif isinstance(self.dtype, UInt16):
            x = self.astype(int16)
        elif isinstance(self.dtype, UInt32):
            x = self.astype(int32)
        elif isinstance(self.dtype, UInt64):
            x = self.astype(int64)
        else:
            raise TypeError(f"unexpected data type `{self.dtype}`")
        return safe_cast(TyArray, (-x).astype(self.dtype))


class TyArrayFloating(TyArrayNumber):
    dtype: Floating

    def __mod__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.mod, lambda a, b: op.mod(a, b, fmod=1), forward=True
        )

    def __rmod__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(
            self, other, operator.mod, lambda a, b: op.mod(a, b, fmod=1), forward=False
        )

    def ceil(self) -> Self:
        return type(self)(op.ceil(self.var))

    def floor(self) -> Self:
        return type(self)(op.floor(self.var))

    def round(self) -> Self:
        return type(self)(op.round(self.var))

    def max(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Self:
        # ONNX standard does not define nan handling so we have to do
        # some special handling for floating points
        from .funcs import astyarray, where

        res = super().max(axis=axis, keepdims=keepdims)

        is_nan = self.isnan().any(axis=axis, keepdims=keepdims)
        return safe_cast(
            type(self), where(is_nan, astyarray(float("nan"), dtype=res.dtype), res)
        )

    def min(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Self:
        # ONNX standard does not define nan handling so we have to do
        # some special handling for floating points
        from .funcs import astyarray, where

        res = super().min(axis=axis, keepdims=keepdims)

        is_nan = self.isnan().any(axis=axis, keepdims=keepdims)
        return safe_cast(
            type(self), where(is_nan, astyarray(float("nan"), dtype=res.dtype), res)
        )

    def mean(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Self:
        if axis == ():
            return self.copy()
        axes = _axis_var(axis, self.ndim)

        summed = type(self)(
            # TODO: File bug:
            # ONNX is unclear about the semantics for `reduce_mean` for empty arrays
            ort_compat.reduce_sum(
                self.var, axes=axes, keepdims=keepdims, noop_with_empty_axes=False
            )
        )
        in_shape = self.dynamic_shape
        axis_ = _axis_array(axis, in_shape.ndim)
        if axis_ is None:
            n_elements = in_shape.prod()
        else:
            n_elements = in_shape.take(axis_).prod()
        return (summed / n_elements).astype(self.dtype)

    def std(
        self,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        correction: int | float = 0.0,
        keepdims: bool = False,
    ) -> Self:
        res = self.variance(axis=axis, correction=correction, keepdims=keepdims).sqrt()
        return res

    def variance(
        self,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        correction: int | float = 0.0,
        keepdims: bool = False,
    ) -> Self:
        from .funcs import astyarray, where

        if axis is None:
            size = self.dynamic_size
            means = self.mean()
        else:
            means = self.mean(axis=axis, keepdims=True)
            if isinstance(axis, int):
                axis_ = op.const([axis], np.int64)
            else:
                axis_ = op.const(axis, np.int64)

            ax_indices = TyArrayInt64(axis_).reshape((-1,))
            size = self.dynamic_shape.take(ax_indices).prod(dtype=int64)

        nom = (self - means).square().sum(axis=axis, keepdims=keepdims)

        if correction != 0:
            size = safe_cast(TyArrayInt64, size - astyarray(correction, dtype=int64))
        res = (nom / size).astype(self.dtype)

        # There is an uncanny corner case here: If self is 0-sized,
        # means is NaN, however, the sum over a zero-sized array is
        # `0`! I.e. we lose the NaN information. We recover it with
        # the `where` call later.
        nan = astyarray(np.nan, dtype=self.dtype)
        is_zero_sized = self.dynamic_shape.prod(dtype=int64) == astyarray(
            0, dtype=int64
        )
        return safe_cast(type(self), where(is_zero_sized, nan, res))

    def isfinite(self) -> TyArrayBool:
        return safe_cast(TyArrayBool, ~(self.isinf() | self.isnan()))

    def isinf(self) -> TyArrayBool:
        return TyArrayBool(op.isinf(self.var))

    def isnan(self) -> TyArrayBool:
        return TyArrayBool(op.isnan(self.var))

    # Element-wise for floating point
    def acos(self) -> Self:
        return type(self)(ort_compat.acos(self.var))

    def acosh(self) -> Self:
        return type(self)(ort_compat.acosh(self.var))

    def asin(self) -> Self:
        return type(self)(ort_compat.asin(self.var))

    def asinh(self) -> Self:
        return type(self)(ort_compat.asinh(self.var))

    def atan(self) -> Self:
        return type(self)(ort_compat.atan(self.var))

    def atanh(self) -> Self:
        return type(self)(ort_compat.atanh(self.var))

    def cos(self) -> Self:
        return type(self)(ort_compat.cos(self.var))

    def cosh(self) -> Self:
        return type(self)(ort_compat.cosh(self.var))

    def exp(self) -> Self:
        return type(self)(op.exp(self.var))

    def log(self) -> Self:
        return type(self)(op.log(self.var))

    def log2(self) -> Self:
        from .py_scalars import TyArrayPyFloat

        res = self.log() / TyArrayPyFloat(float(np.log(2)))
        return safe_cast(type(self), res)

    def log10(self) -> Self:
        from .py_scalars import TyArrayPyFloat

        res = self.log() / TyArrayPyFloat(float(np.log(10)))
        return safe_cast(type(self), res)

    def sin(self) -> Self:
        return type(self)(ort_compat.sin(self.var))

    def sinh(self) -> Self:
        return type(self)(ort_compat.sinh(self.var))

    def tan(self) -> Self:
        return type(self)(ort_compat.tan(self.var))

    def tanh(self) -> Self:
        return type(self)(ort_compat.tanh(self.var))

    def trunc(self) -> Self:
        from .funcs import astyarray, where

        zero = astyarray(0, use_py_scalars=True)
        return safe_cast(
            type(self),
            where(safe_cast(TyArrayBool, self < zero), self.ceil(), self.floor()),
        )


class TyArrayBool(TyArray):
    dtype = bool_

    def __or__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(self, other, operator.or_, op.or_, forward=True)

    def __ror__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(self, other, operator.or_, op.or_, forward=False)

    def __xor__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(self, other, operator.xor, op.xor, forward=True)

    def __rxor__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(self, other, operator.xor, op.xor, forward=False)

    def __and__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(self, other, operator.and_, op.and_, forward=True)

    def __rand__(self, other) -> TyArrayBase:
        return _promote_and_apply_op(self, other, operator.and_, op.and_, forward=False)

    def __invert__(self) -> Self:
        return type(self)(op.not_(self.var))

    def __ndx_logical_and__(self, rhs: TyArrayBase, /) -> Self:
        if isinstance(rhs, TyArrayBool):
            return type(self)(op.and_(self.var, rhs.var))
        return NotImplemented

    def __ndx_rlogical_and__(self, lhs: TyArrayBase, /) -> TyArrayBase:
        if isinstance(lhs, TyArrayBool):
            return type(self)(op.and_(lhs.var, self.var))
        return NotImplemented

    def __ndx_logical_xor__(self, rhs: TyArrayBase, /) -> Self:
        if isinstance(rhs, TyArrayBool):
            return type(self)(op.xor(self.var, rhs.var))
        return NotImplemented

    def __ndx_rlogical_xor__(self, lhs: TyArrayBase, /) -> TyArrayBase:
        if isinstance(lhs, TyArrayBool):
            return type(self)(op.xor(lhs.var, self.var))
        return NotImplemented

    def __ndx_logical_or__(self, rhs: TyArrayBase, /) -> Self:
        if isinstance(rhs, TyArrayBool):
            return type(self)(op.or_(self.var, rhs.var))
        return NotImplemented

    def __ndx_rlogical_or__(self, lhs: TyArrayBase, /) -> TyArrayBase:
        if isinstance(lhs, TyArrayBool):
            return type(self)(op.or_(lhs.var, self.var))
        return NotImplemented

    def nonzero(self) -> tuple[TyArrayInt64, ...]:
        # Use numeric implementation
        return self.astype(uint8).nonzero()

    def logical_not(self) -> Self:
        return ~self

    def static_map(self, mapping: Mapping[KEY, VALUE], default: VALUE) -> TyArray:
        from .funcs import astyarray, where

        mapping_ = {bool(k): v for k, v in mapping.items()}

        true_val = mapping_.get(True, default)
        false_val = mapping_.get(False, default)

        return safe_cast(
            TyArray, where(self, astyarray(true_val), astyarray(false_val))
        )


class TyArrayInt8(TyArraySignedInteger):
    dtype = int8


class TyArrayInt16(TyArraySignedInteger):
    dtype = int16


class TyArrayInt32(TyArraySignedInteger):
    dtype = int32


class TyArrayInt64(TyArraySignedInteger):
    dtype = int64


class TyArrayUInt8(TyArrayUnsignedInteger):
    dtype = uint8


class TyArrayUInt16(TyArrayUnsignedInteger):
    dtype = uint16


class TyArrayUInt32(TyArrayUnsignedInteger):
    dtype = uint32


class TyArrayUInt64(TyArrayUnsignedInteger):
    dtype = uint64


class TyArrayFloat16(TyArrayFloating):
    dtype = float16


class TyArrayFloat32(TyArrayFloating):
    dtype = float32


class TyArrayFloat64(TyArrayFloating):
    dtype = float64


def ascoredata(var: Var) -> TyArray:
    dtype = from_numpy(var.unwrap_tensor().dtype)

    return dtype._tyarr_class(var)


def is_sequence_of_core_data(
    seq: Sequence[TyArrayBase],
) -> TypeGuard[Sequence[TyArray]]:
    return all(isinstance(d, TyArray) for d in seq)


def all_items_are_int(
    seq: Sequence,
) -> TypeGuard[Sequence[int]]:
    return all(isinstance(d, int) for d in seq)


def contains_no_ellipsis(
    seq: tuple[int | slice | EllipsisType, ...],
) -> TypeGuard[tuple[int | slice, ...]]:
    return all(el != ... for el in seq)


T = TypeVar("T", bound=TyArray)


def _element_wise(x: T, op: Callable[[Var], Var], via: _OnnxDType | None = None) -> T:
    """Apply an element wise operations possible with an onnxruntime workaround ``via``
    a different data type.

    The workaround is only applied if
    ``spox._value_prop._VALUE_PROP_BACKEND==spox._future.ValuePropBackend.ONNXRUNTIME``.
    """
    target_ort = (
        spox._value_prop._VALUE_PROP_BACKEND
        == spox._future.ValuePropBackend.ONNXRUNTIME
    )
    if via is not None and target_ort:
        res = ascoredata(op(x.astype(via).var))
        return safe_cast(type(x), res.astype(x.dtype))
    return type(x)(op(x.var))


def _promote_and_apply_op(
    this: TyArray,
    other: TyArrayBase,
    arr_op: Callable[[TyArrayBase, TyArrayBase], TyArrayBase],
    spox_op: Callable[[Var, Var], Var],
    forward: bool,
) -> TyArrayBase | NotImplementedType:
    """Promote and apply an operation by passing it through to the data member."""
    if isinstance(other, TyArray):
        if this.dtype != other.dtype:
            a, b = promote(this, other)
            return arr_op(a, b) if forward else arr_op(b, a)

        # Data is core & integer
        var = spox_op(this.var, other.var) if forward else spox_op(other.var, this.var)
        return ascoredata(var)
    return NotImplemented


#####################
# Conversion tables #
#####################

# Promotion tables taken from:
# https://data-apis.org/array-api/draft/API_specification/type_promotion.html#type-promotion
# and
# https://numpy.org/neps/nep-0050-scalar-promotion.html#motivation-and-scope
_signed_signed: dict[tuple[NumericDTypes, NumericDTypes], NumericDTypes] = {
    (int8, int8): int8,
    (int16, int8): int16,
    (int32, int8): int32,
    (int64, int8): int64,
    (int8, int16): int16,
    (int16, int16): int16,
    (int32, int16): int32,
    (int64, int16): int64,
    (int8, int32): int32,
    (int16, int32): int32,
    (int32, int32): int32,
    (int64, int32): int64,
    (int8, int64): int64,
    (int16, int64): int64,
    (int32, int64): int64,
    (int64, int64): int64,
}
_unsigned_unsigned: dict[tuple[NumericDTypes, NumericDTypes], NumericDTypes] = {
    (uint8, uint8): uint8,
    (uint16, uint8): uint16,
    (uint32, uint8): uint32,
    (uint64, uint8): uint64,
    (uint8, uint16): uint16,
    (uint16, uint16): uint16,
    (uint32, uint16): uint32,
    (uint64, uint16): uint64,
    (uint8, uint32): uint32,
    (uint16, uint32): uint32,
    (uint32, uint32): uint32,
    (uint64, uint32): uint64,
    (uint8, uint64): uint64,
    (uint16, uint64): uint64,
    (uint32, uint64): uint64,
    (uint64, uint64): uint64,
}
_mixed_integers: dict[tuple[NumericDTypes, NumericDTypes], NumericDTypes] = {
    (int8, uint8): int16,
    (int16, uint8): int16,
    (int32, uint8): int32,
    (int64, uint8): int64,
    (int8, uint16): int32,
    (int16, uint16): int32,
    (int32, uint16): int32,
    (int64, uint16): int64,
    (int8, uint32): int64,
    (int16, uint32): int64,
    (int32, uint32): int64,
    (int64, uint32): int64,
    # NOTE: Standard does not define interaction with uint64!
}
_mixed_integers |= {(k[1], k[0]): v for k, v in _mixed_integers.items()}

_floating_floating: dict[tuple[NumericDTypes, NumericDTypes], NumericDTypes] = {
    (float16, float16): float16,
    (float16, float32): float32,
    (float32, float16): float32,
    (float16, float64): float64,
    (float64, float16): float64,
    (float32, float32): float32,
    (float32, float64): float64,
    (float64, float32): float64,
    (float64, float64): float64,
}

# Non-standard interactions
_non_standard: dict[tuple[NumericDTypes, NumericDTypes], NumericDTypes] = {
    (int8, uint64): float64,
    (int16, uint64): float64,
    (int32, uint64): float64,
    (int64, uint64): float64,
}

# Mixed integers and floating point numbers are not
# strictly defined.
_int_to_floating: dict[NumericDTypes, NumericDTypes] = {
    int8: float32,
    uint8: float32,
    int16: float32,
    uint16: float32,
    int32: float64,
    uint32: float64,
    int64: float64,
    uint64: float64,
}


def _result_type_core_numeric(a: NumericDTypes, b: NumericDTypes) -> NumericDTypes:
    # Attempt promotion between known types. The implementation is not
    # using `isinstance` to avoid subclassing issues.
    if ret := _signed_signed.get((a, b)):
        return ret
    if ret := _unsigned_unsigned.get((a, b)):
        return ret
    if ret := _mixed_integers.get((a, b)):
        return ret
    if ret := _floating_floating.get((a, b)):
        return ret
    if ret := _non_standard.get((a, b)):
        return ret
    if a_floating := _int_to_floating.get(a):
        return _result_type_core_numeric(a_floating, b)
    if b_floating := _int_to_floating.get(b):
        return _result_type_core_numeric(a, b_floating)

    raise ValueError(f"No promotion between `{a}` and `{b}` is defined.")


def _axis_array(axis: int | tuple[int, ...] | None, rank: int) -> TyArrayInt64 | None:
    """Construct an `axis` array for reduction operations such as `mean` and normalize
    it to positive values."""
    if axis is None:
        return None
    else:
        if isinstance(axis, int):
            axis = (axis,)
        axes = [ax if ax >= 0 else ax + rank for ax in axis]

    return TyArrayInt64(op.const(axes, np.int64))


def _axis_var(axis: int | tuple[int, ...] | None, rank: int) -> Var | None:
    res = _axis_array(axis, rank)
    if res is None:
        return None
    return res.var


def _to_slice_kwargs(slices: list[FancySlice]) -> dict[str, Var]:
    if all(el.is_noop() for el in slices):
        return {}

    transposed = zip(
        *[(el.start, el.stop, el.step) for el in slices if not el.is_noop()]
    )

    def concat(lst: tuple[TyArrayInt64, ...]) -> TyArrayInt64:
        # This concatenation is a bit of a hot spot in the test suite.
        # Thus, we use some hack constant folding here to skip some
        # value prop overhead.
        try:
            # fast path
            np_arrays = [el.unwrap_numpy() for el in lst]
            return TyArrayInt64(op.const(np.asarray(np_arrays)))
        except ValueError:
            pass
        arrays_1d = [el.reshape((-1,)) for el in lst]
        return arrays_1d[0].concat(arrays_1d[1:], axis=0)

    # We can't concat 0-D arrays, thus the reshape
    kw = {k: concat(lst).var for k, lst in zip(["starts", "ends", "steps"], transposed)}

    kw["axes"] = TyArrayInt64(
        op.const([i for i, el in enumerate(slices) if not el.is_noop()], np.int64)
    ).var
    return kw