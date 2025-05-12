# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from copy import copy as std_copy
from types import EllipsisType, NotImplementedType
from typing import Literal, TypeAlias, TypeGuard, TypeVar, cast, overload

import numpy as np
from spox import Tensor, Var, argument
from typing_extensions import Self

from ndonnx import DType
from ndonnx.types import NestedSequence, OnnxShape, PyScalar

from .._schema import DTypeInfoV1
from . import TyArrayBase, safe_cast
from . import ort_compat as op
from .dtype_independent_funcs import maximum, minimum, where
from .indexing import FancySlice

_ScalarInt: TypeAlias = "TyArrayInteger"
"""Alias signaling that this must be a rank-0 integer tensor."""
_BoolMask: TypeAlias = "TyArrayBool"
"""Alias signaling that this must be a rank-1 boolean tensor."""
SetitemItem: TypeAlias = "int | slice | EllipsisType | _ScalarInt"
"""A single item; i.e. not a tuple nor a boolean mask.

This does not include `None`.
"""
GetitemItem: TypeAlias = "int | slice | EllipsisType | _ScalarInt | None"
"""A single item (; i.e. not a tuple nor a boolean mask) for __getitem__.

This includes `None`.
"""
SetitemIndex: TypeAlias = "SetitemItem | tuple[SetitemItem, ...] | _BoolMask"
GetitemIndex: TypeAlias = "GetitemItem | tuple[GetitemItem, ...] | _BoolMask"

TY_ARRAY = TypeVar("TY_ARRAY", bound="TyArray")
TY_ARRAY_co = TypeVar("TY_ARRAY_co", bound="TyArray", covariant=True)

TY_ARRAY_BASE = TypeVar("TY_ARRAY_BASE", bound="TyArrayBase")
TY_ARRAY_BASE_co = TypeVar("TY_ARRAY_BASE_co", bound="TyArrayBase", covariant=True)

TY_ARRAY_NUMBER = TypeVar("TY_ARRAY_NUMBER", bound="TyArrayNumber")

KEY = TypeVar("KEY", int, float, str)
VALUE = TypeVar("VALUE", int, float, str)


class _OnnxDType(DType[TY_ARRAY_co]):
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

    @abstractmethod
    def _build(self, var: Var) -> TY_ARRAY_co: ...

    def __str__(self) -> str:
        return type(self).__name__.lower()

    def __ndx_cast_from__(self, arr: TyArrayBase) -> TY_ARRAY_co:
        if isinstance(arr, TyArray):
            var = op.cast(arr._var, to=self.unwrap_numpy())
            return self._build(var)
        return NotImplemented

    def __ndx_result_type__(self, rhs: DType | PyScalar) -> DType | NotImplementedType:
        if isinstance(rhs, _OnnxDType | PyScalar):
            return result_type(self, rhs)

        return NotImplemented

    def __ndx_create__(
        self, val: PyScalar | np.ndarray | TyArrayBase | Var | NestedSequence
    ) -> TY_ARRAY_co | NotImplementedType:
        if isinstance(val, Var):
            return _var_to_tyarray(val).astype(self)
        elif isinstance(val, PyScalar | np.ndarray | np.generic):
            return const(val, dtype=self)
        elif isinstance(val, Sequence):
            return self.__ndx_create__(np.asarray(val))
        elif isinstance(val, TyArrayBase):
            return val.copy().astype(self)
        return NotImplemented

    def __ndx_argument__(self, shape: OnnxShape) -> TY_ARRAY_co:
        var = argument(Tensor(self.unwrap_numpy(), shape))
        return self._build(var)

    @property
    def __ndx_infov1__(self) -> DTypeInfoV1:
        # TODO: Follow numpy naming scheme for v2!
        return DTypeInfoV1(
            author="ndonnx", type_name=self.__class__.__name__, meta=None
        )

    def __ndx_arange__(
        self,
        start: int | float,
        stop: int | float,
        step: int | float = 1,
    ) -> TY_ARRAY_co:
        np_dtype = self.unwrap_numpy()
        np_arr = np.array([start, stop, step])

        if np_dtype.kind != np_arr.dtype.kind:
            TypeError(
                f"'start', 'stop', and 'step' have to match data type kind of `{self}`"
            )
        # Get everything onto the same type
        start_, stop_, step_ = np.array([start, stop, step], dtype=np_dtype)

        var = op.range(
            op.const(start_, np_dtype),
            op.const(stop_, np_dtype),
            op.const(step_, np_dtype),
        )

        return self._build(var)

    def __ndx_eye__(
        self,
        n_rows: int,
        n_cols: int | None = None,
        /,
        *,
        k: int = 0,
    ) -> TY_ARRAY_co:
        n_cols = n_rows if n_cols is None else n_cols
        # There is no adequate ONNX operator
        mat = np.eye(n_rows, n_cols, k=k, dtype=self.unwrap_numpy())
        var = op.const(mat)

        return self._build(var)

    def __ndx_ones__(self, shape: tuple[int, ...] | TyArrayInt64) -> TY_ARRAY_co:
        np_dtype = self.unwrap_numpy()
        scalar = self._build(op.const(1, np_dtype))

        return scalar.broadcast_to(shape)

    def __ndx_zeros__(self, shape: tuple[int, ...] | TyArrayInt64) -> TY_ARRAY_co:
        np_dtype = self.unwrap_numpy()
        scalar = self._build(op.const(0, np_dtype))

        return scalar.broadcast_to(shape)


class _Number(_OnnxDType[TY_ARRAY_co]): ...


class Utf8(_OnnxDType["TyArrayUtf8"]):
    def _build(self, var: Var) -> TyArrayUtf8:
        return TyArrayUtf8(var)

    def __ndx_zeros__(self, shape: tuple[int, ...] | TyArrayInt64) -> TyArrayUtf8:
        # NumPy returns empty strings for `zeros` with string data
        # types. We follow that lead.
        np_dtype = self.unwrap_numpy()
        scalar = self._build(op.const("", np_dtype))

        return scalar.broadcast_to(shape)


class Bool(_OnnxDType["TyArrayBool"]):
    def _build(self, var: Var) -> TyArrayBool:
        return TyArrayBool(var)

    @property
    def __ndx_infov1__(self) -> DTypeInfoV1:
        # Override to be compatible with the existing schema
        return DTypeInfoV1(author="ndonnx", type_name="Boolean", meta=None)


class Integer(_Number[TY_ARRAY_co]): ...


class Floating(_Number[TY_ARRAY_co]): ...


class Int8(Integer["TyArrayInt8"]):
    def _build(self, var: Var) -> TyArrayInt8:
        return TyArrayInt8(var)


class Int16(Integer["TyArrayInt16"]):
    def _build(self, var: Var) -> TyArrayInt16:
        return TyArrayInt16(var)


class Int32(Integer["TyArrayInt32"]):
    def _build(self, var: Var) -> TyArrayInt32:
        return TyArrayInt32(var)


class Int64(Integer["TyArrayInt64"]):
    def _build(self, var: Var) -> TyArrayInt64:
        return TyArrayInt64(var)


class UInt8(Integer["TyArrayUInt8"]):
    def _build(self, var: Var) -> TyArrayUInt8:
        return TyArrayUInt8(var)


class UInt16(Integer["TyArrayUInt16"]):
    def _build(self, var: Var) -> TyArrayUInt16:
        return TyArrayUInt16(var)


class UInt32(Integer["TyArrayUInt32"]):
    def _build(self, var: Var) -> TyArrayUInt32:
        return TyArrayUInt32(var)


class UInt64(Integer["TyArrayUInt64"]):
    def _build(self, var: Var) -> TyArrayUInt64:
        return TyArrayUInt64(var)


class Float16(Floating["TyArrayFloat16"]):
    def _build(self, var: Var) -> TyArrayFloat16:
        return TyArrayFloat16(var)


class Float32(Floating["TyArrayFloat32"]):
    def _build(self, var: Var) -> TyArrayFloat32:
        return TyArrayFloat32(var)


class Float64(Floating["TyArrayFloat64"]):
    def _build(self, var: Var) -> TyArrayFloat64:
        return TyArrayFloat64(var)


# Non-nullable Singleton instances
bool_: Bool = Bool()

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
DTypes = NumericDTypes | Utf8 | Bool


class TyArray(TyArrayBase):
    _var: Var

    def __init__(self, var: Var):
        var_dtype = var.unwrap_tensor().dtype
        if from_numpy_dtype(var_dtype) != self.dtype:
            raise ValueError(
                f"data type of 'var' (`{var_dtype}`) is incompatible with `{type(self)}`"
            )
        if var.unwrap_tensor().shape is None:
            raise ValueError("'var' has no shape information")
        self._var = var

    def __ndx_value_repr__(self) -> dict[str, str]:
        if self.is_constant:
            # TODO: should this really be "data"?
            return {"data": str(self.unwrap_numpy().tolist())}
        else:
            return {"data": "*lazy*"}

    def __getitem__(self, key: GetitemIndex) -> Self:
        # Dispatch cases roughly in line with how the standard describes them here
        # https://data-apis.org/array-api/2024.12/API_specification/indexing.html

        # rank-0 indexing
        if self.ndim == 0 and key == () or key == ...:
            return self.copy()
        # Boolean indexing
        if isinstance(key, TyArrayBool | bool):
            if isinstance(key, bool):
                # NOTE: This is not mandated by the standard. Is this needed?
                key = const(key, dtype=bool_)
            return self._getitem_boolmask(key)

        # Single-axis indexing
        if _is_single_axis_indexing(key):
            return self._getitem_multi_axis_indexing((key,))

        # Multi-axis indexing
        if _is_multi_axis_indexing(key):
            return self._getitem_multi_axis_indexing(key)

        # Integer-array indexing
        if _is_integer_array_indexing(key):
            return self._getitem_integer_array_indexing(key)

        raise IndexError(f"cannot index with key `{key}`")

    def _getitem_multi_axis_indexing(self, key: tuple[GetitemItem, ...]) -> Self:
        if isinstance(key, list):
            # Technically not supported, but a common pattern
            key = tuple(key)
        if isinstance(key, int | slice | EllipsisType | None):
            key = (key,)

        # Validation
        cnt_slice_or_idx = len(
            list(el for el in key if isinstance(el, int | slice | TyArrayInteger))
        )

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

        var = self._var
        if new_axis_positions:
            var = op.unsqueeze(self._var, op.const(new_axis_positions, np.int64))

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
        for xs, ks in zip(self.shape, key.shape):
            if ks in (xs, 0, None) or isinstance(ks, str):
                continue
            raise IndexError(
                f"shape of 'key' (`{key.shape}`) is incompatible with shape of 'self' (`{self.shape}`)"
            )
        if key.ndim > self.ndim:
            raise IndexError(
                f"rank of 'key' (`{key.ndim}`) must be less or equal to rank of 'self' (`{self.ndim}`)"
            )
        if self.ndim == key.ndim:
            # Optimization
            arr = self
            key_arr = key
            if self.ndim == key.ndim == 0:
                # Output will be 1D; compress cannot handle scalars
                arr = self.reshape((1,))
                key_arr = key.reshape((1,))
            else:
                arr = self.reshape((-1,))
                key_arr = key.reshape((-1,))

            return type(self)(op.compress(arr._var, key_arr._var, axis=0))

        if key.ndim == 0:
            key = key[None, ...]
            x = self[None, ...]
            res = op.compress(x._var, key._var, axis=0)
            return type(self)(res)

        # Can't use reshape since the use of -1 for zero dimensional
        # arrays is undefined
        in_shape = (
            self.dynamic_shape[: key.ndim]
            .prod(keepdims=True, dtype=int64)
            .concat([self.dynamic_shape[key.ndim :]], axis=0)
        )
        var = self.reshape(in_shape)._var

        res = op.compress(var, key.reshape((-1,))._var, axis=0)
        return type(self)(res)

    def _getitem_integer_array_indexing(
        self, key: tuple[int | TyArrayInt64, ...]
    ) -> Self:
        key_arrays = []
        for el in key:
            if isinstance(el, int):
                key_arrays.append(const(np.asarray([el], np.int64), dtype=int64))
            else:
                key_arrays.append(el)
        key_arrays_tuple = broadcast_arrays(*[el[..., None] for el in key_arrays])
        key_array = key_arrays_tuple[0].concat(key_arrays_tuple[1:], axis=-1)

        var = op.gather_nd(self._var, key_array._var)
        return type(self)(var)

    def __setitem__(
        self,
        key: SetitemIndex,
        value: Self,
        /,
    ) -> None:
        if key == ():
            self._var = value.broadcast_to(self.dynamic_shape)._var
            return

        if isinstance(key, list):
            # Technically not supported, but a common pattern
            key = tuple(key)
        if isinstance(key, int | slice | EllipsisType):
            key = (key,)

        if isinstance(key, TyArrayBool):
            return self._setitem_boolmask(key, value)
        elif isinstance(key, TyArrayInteger):
            raise IndexError("'__setitem__' with integer arrays is not supported")

        # Shortcut: Remove ellipsis from the front if there is any
        if ... in key and key.index(...) == len(key):
            key = key[:-1]

        if _contains_no_ellipsis(key):
            arr_key = _key_to_indices(key, self.dynamic_shape)
            self._setitem_int_array(arr_key, value)
            return

        # The ellipsis is somewhere else in the key. We need to do some rearangment
        tmp_dim_perm, reverse_perm, keys_no_ellips = _move_ellipsis_back(self.ndim, key)
        perm_self = self.permute_dims(tmp_dim_perm)
        perm_self[keys_no_ellips] = value
        self._var = perm_self.permute_dims(reverse_perm)._var

    def _setitem_boolmask(self, key: TyArrayBool, value: Self) -> None:
        if self.ndim < key.ndim:
            raise IndexError("provided boolean mask has higher rank than 'self'")
        if self.ndim > key.ndim:
            diff = self.ndim - key.ndim
            # expand to rank of self
            idx = [...] + diff * [None]
            key = key[tuple(idx)]

        if value.ndim == 0:
            # We can be sure that we don't run into broadcasting
            # issues with a zero-sized value if the value is a
            # scalar. This allows us to use `where`.
            self._var = op.where(key._var, value._var, self._var)
            return

        int_keys = safe_cast(
            TyArrayInt64,
            const(1, int64).broadcast_to(key.dynamic_shape).cumulative_sum() - 1,
        )[key]
        arr = self.copy().reshape((-1,))
        arr.put(int_keys, value.reshape((-1,)))
        arr.reshape(self.dynamic_shape)
        self._var = arr._var

    def _setitem_int_array(self, key: TyArrayInt64, value: Self) -> None:
        # The last dim of shape must be statically known (i.e. the
        # length of the indexing tuple
        idx_tuple_len = key.shape[-1]
        if not isinstance(idx_tuple_len, int):
            raise IndexError("length of indexing tuple must be statically known")
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
        self._var = op.scatter_nd(self._var, key._var, value_correct_shape._var)

    def put(
        self,
        key: TyArrayInt64,
        value: Self,
        /,
    ) -> None:
        data = self.reshape((-1,))
        if key.ndim == 1:
            key = key.reshape((-1, 1))
        data._setitem_int_array(key, value)
        self._var = data.reshape(self.dynamic_shape)._var

    @property
    def dtype(self) -> _OnnxDType:
        # This property is overwritten downstream. We define it here
        # to be able to use `safe_cast` with TyArray (MyPy does not accept an
        # abstract `T` in `type[T]`
        # https://github.com/python/mypy/issues/4717)
        raise NotImplementedError

    @property
    def dynamic_shape(self) -> TyArrayInt64:
        # Spox does not attempt value propagation for lazy
        # arrays. Thus, we never get a constant from lazy input, even
        # if the shape is statically known. We therefore do that
        # manually here.
        if all(isinstance(el, int) for el in self.shape):
            return const(np.array(self.shape), dtype=int64)

        var = op.shape(self._var)
        return TyArrayInt64(var)

    @property
    def dynamic_size(self) -> TyArrayInt64:
        var = op.size(self._var)
        return TyArrayInt64(var)

    @property
    def mT(self) -> Self:  # noqa: N802
        if self.ndim < 2:
            raise ValueError(
                f"array must have two or more dimensions, found `{self.ndim}`"
            )
        dims = list(range(self.ndim))
        dims = dims[:-2] + dims[-2:][::-1]
        var = op.transpose(self._var, perm=dims)
        return type(self)(var)

    @property
    def shape(self) -> OnnxShape:
        shape = self._var.unwrap_tensor().shape
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

    @property
    def is_constant(self) -> bool:
        return self._var._value is not None

    def unwrap_numpy(self) -> np.ndarray:
        if self._var._value is not None:
            np_arr = np.asarray(self._var._value.value)
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
        var = op.reduce_prod(bools.astype(int32)._var, axes=axes, keepdims=keepdims)
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
        var = op.reduce_sum(bools.astype(float32)._var, axes=axes, keepdims=keepdims)
        return safe_cast(TyArrayBool, TyArrayFloat32(var).astype(bool_))

    def as_core_dtype(self, dtype: DTypes) -> TyArray:
        raise ValueError(f"casting between `{self.dtype}` and `{dtype}` is undefined")

    def broadcast_to(self, shape: tuple[int, ...] | TyArrayInt64) -> Self:
        if isinstance(shape, tuple):
            shape = const(np.array(shape), dtype=int64)
        if shape.ndim != 1:
            raise ValueError(f"'shape' must be a rank-1 tensor, found `{shape.ndim}`")
        if not isinstance(shape.shape[0], int):
            raise ValueError("'shape' must be a rank-1 tensor with static length")
        (target_rank,) = shape.shape
        if not isinstance(target_rank, int):
            raise ValueError(
                "target rank of broadcasting operation must be statically known"
            )
        if target_rank < self.ndim:
            raise ValueError("target rank must be equal or greater than rank of 'self'")
        var = op.expand(self._var, shape._var)
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
            dummy_axis = op.const([axis + 1], dtype=np.int64)
            vars = [op.unsqueeze(a._var, dummy_axis) for a in arrays]
            var = op.concat(vars, axis=axis)
            var = op.squeeze(var, dummy_axis)
        else:
            var = op.concat([a._var for a in arrays], axis=0 if axis is None else axis)
        return type(self)(var)

    def copy(self) -> Self:
        return std_copy(self)

    def disassemble(self) -> Var:
        return self._var

    def count_nonzero(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> TyArrayInt64:
        non_zeros = self.astype(bool_).astype(int64)
        return non_zeros.sum(axis=axis, keepdims=keepdims, dtype=int64)

    def nonzero(self) -> tuple[TyArrayInt64, ...]:
        if self.ndim == 0:
            raise ValueError("'nonzero' is not defined for scalar arrays")

        res = TyArrayInt64(op.non_zero(self._var))
        out = []
        for i in range(self.ndim):
            out.append(res[i, :])
        return tuple(out)

    def permute_dims(self, axes: tuple[int, ...]) -> Self:
        var = op.transpose(self._var, perm=axes)
        return type(self)(var)

    def reshape(self, shape: tuple[int, ...] | TyArrayInt64) -> Self:
        if isinstance(shape, tuple | list):
            var = op.reshape(self._var, op.const(shape, np.int64), allowzero=True)
        else:
            var = op.reshape(self._var, shape._var, allowzero=True)
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
            res = op.squeeze(self._var, op.const(axis, np.int64))
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
        var = op.gather(self._var, indices._var, axis=axis)
        return type(self)(var)

    def take_along_axis(self, indices: TyArrayInt64, /, *, axis: int = -1) -> Self:
        var = op.gather_elements(self._var, indices._var, axis=axis)
        return type(self)(var)

    def tile(self, repetitions: tuple[int, ...], /) -> Self:
        n = self.ndim
        m = len(repetitions)
        x = self
        if n > m:
            repetitions = tuple([1] * (n - m) + list(repetitions))
        elif n < m:
            x = self[tuple([None] * (m - n) + [...])]

        var = op.tile(x._var, op.const(repetitions, np.int64))
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
        var = op.trilu(self._var, k=op.const(k, dtype=np.int64), upper=False)
        return type(self)(var)

    def triu(self, /, *, k: int = 0) -> Self:
        var = op.trilu(self._var, k=op.const(k, dtype=np.int64), upper=True)
        return type(self)(var)

    def unique_all(self) -> tuple[Self, TyArrayInt64, TyArrayInt64, TyArrayInt64]:
        flattened = self.reshape((-1,))
        res = op.unique(flattened._var, sorted=True)
        values, indices, inverse_indices, counts = (
            type(self)(res[0]),
            TyArrayInt64(res[1]),
            TyArrayInt64(res[2]),
            TyArrayInt64(res[3]),
        )

        inverse_indices = inverse_indices.reshape(self.dynamic_shape)
        counts = counts.reshape(values.dynamic_shape)

        return (values, indices, inverse_indices, counts)

    def __ndx_cast_to__(self, dtype: DType[TY_ARRAY_BASE_co]) -> TY_ARRAY_BASE_co:
        # We pretend that we don't know about any other data type. We
        # delegate all work to ``DType.__ndx_cast_from__``
        return NotImplemented

    @overload  # type: ignore
    def __eq__(self, other: TyArray | PyScalar) -> TyArrayBool: ...

    @overload
    def __eq__(self, other: TyArrayBase | PyScalar) -> TyArrayBool: ...

    def __eq__(self, other: TyArrayBase | PyScalar) -> TyArrayBool:
        if not isinstance(other, TyArray | bool | int | float | str):
            return NotImplemented
        return safe_cast(TyArrayBool, super().__eq__(other))

    def __ndx_equal__(self, other) -> TyArrayBool:
        if isinstance(other, TyArray | bool | int | float | str):
            a, b = promote(self, other)
            var = op.equal(a._var, b._var)
            return TyArrayBool(var)
        return NotImplemented

    def isin(self, items: Sequence[VALUE]) -> TyArrayBool:
        # Filter out nan values since we never want to compare equal to them (NumPy semantics)
        items = [el for el in items if not isinstance(el, float) or not np.isnan(el)]

        # Optimizations:
        if len(items) == 0:
            return const(False, dtype=bool_).broadcast_to(self.dynamic_shape)
        if len(items) == 1:
            return safe_cast(TyArrayBool, self == const(items[0]))

        # label_encoder based implementation
        mapping = dict(zip(items, (True,) * len(items)))
        return safe_cast(TyArrayBool, self.apply_mapping(mapping, False))

    def apply_mapping(self, mapping: Mapping[KEY, VALUE], default: VALUE) -> TyArray:
        if not mapping:
            return safe_cast(TyArray, const(default).broadcast_to(self.dynamic_shape))
        np_arr_dtype = self.dtype.unwrap_numpy()
        np_keys = np.array(list(mapping.keys()))

        result_np_dtype_key = np.result_type(np_arr_dtype, np_keys)
        result_dtype_key: _OnnxDType = from_numpy_dtype(result_np_dtype_key)

        if result_dtype_key != self.dtype:
            return self.astype(result_dtype_key).apply_mapping(mapping, default)

        values = np.array(list(mapping.values()))
        # Compat for Windows on numpy 1.x
        if values.dtype == np.int32:
            values = values.astype(np.int64)
        if values.dtype.kind in ("O", "U"):
            values = values.astype(str)
            # Don't use values.dtype to cast the default since it may
            # have a length associated with it!
            default_ = np.array([default], dtype=str)
        else:
            default_ = np.array([default], dtype=values.dtype)

        var = op.label_encoder(
            self._var,
            keys_tensor=np_keys.astype(result_np_dtype_key),
            values_tensor=values,
            default_tensor=default_,
        )

        return _var_to_tyarray(var)

    @overload
    def __ndx_where__(
        self, cond: TyArrayBool, y: TyArray, /
    ) -> TyArray | NotImplementedType: ...

    @overload
    def __ndx_where__(
        self, cond: TyArrayBool, y: TyArrayBase | PyScalar, /
    ) -> TyArrayBase | NotImplementedType: ...

    def __ndx_where__(
        self, cond: TyArrayBool, y: TyArrayBase | PyScalar, /
    ) -> TyArrayBase | NotImplementedType:
        if isinstance(y, TyArray | PyScalar):
            x, y = promote(self, y)
            var = op.where(cond._var, x._var, y._var)
            return type(x)(var)

        return NotImplemented


class TyArrayUtf8(TyArray):
    @property
    def dtype(self) -> Utf8:
        return utf8

    def __add__(self, other: TyArrayBase | PyScalar) -> TyArrayUtf8:
        if isinstance(other, TyArrayUtf8 | str):
            a, b = promote(self, other)
            var = op.string_concat(a._var, b._var)
            return safe_cast(TyArrayUtf8, _var_to_tyarray(var))

        return NotImplemented

    def __radd__(self, other: TyArrayBase | PyScalar) -> TyArrayUtf8:
        if isinstance(other, str):
            b, a = promote(self, other)
            var = op.string_concat(a._var, b._var)
            return safe_cast(TyArrayUtf8, _var_to_tyarray(var))

        return NotImplemented

    def isnan(self) -> TyArrayBool:
        return const(False, dtype=bool_).broadcast_to(self.dynamic_shape)

    def __ndx_cast_to__(self, dtype: DType[TY_ARRAY_BASE_co]) -> TY_ARRAY_BASE_co:
        if dtype == bool_:
            # Only the empty string is false-y
            return (self != const("")).astype(dtype)
        return super().__ndx_cast_to__(dtype=dtype)


class TyArrayNumber(TyArray):
    def _arg_minmax(
        self, spox_op, /, *, axis: int | None = None, keepdims: bool = False
    ) -> TyArrayInt64:
        x = self
        axis_ = axis
        if axis_ is None:
            x = x.reshape((-1,))
            axis_ = 0
        res = TyArrayInt64(spox_op(x._var, axis=axis_, keepdims=keepdims))
        if axis is None and keepdims:
            res = res.reshape(tuple(1 for _ in range(self.ndim)))
        return res

    def argmax(
        self, /, *, axis: int | None = None, keepdims: bool = False
    ) -> TyArrayInt64:
        return self._arg_minmax(op.arg_max, axis=axis, keepdims=keepdims)

    def argmin(
        self, /, *, axis: int | None = None, keepdims: bool = False
    ) -> TyArrayInt64:
        return self._arg_minmax(op.arg_min, axis=axis, keepdims=keepdims)

    def argsort(
        self, /, *, axis: int = -1, descending: bool = False, stable: bool = True
    ) -> TyArrayInt64:
        if axis < 0:
            axis += self.ndim
        k = self.dynamic_shape[axis : axis + 1]._var  # k must be 1D, thus the slice
        _values, indices = op.top_k(
            self._var, k, axis=axis, largest=descending, sorted=stable
        )
        return TyArrayInt64(indices)

    def clip(
        self, min: TyArrayBase | None = None, max: TyArrayBase | None = None
    ) -> Self:
        if min is None and max is None:
            return self.copy()

        # The standard says that min/max must not change the output type.
        min_ = None if min is None else min.astype(self.dtype)._var
        max_ = None if max is None else max.astype(self.dtype)._var

        var = op.clip(self._var, min_, max_)

        return type(self)(var)

    def cumulative_prod(
        self,
        /,
        *,
        axis: int | None = None,
        dtype: DType | None = None,
        include_initial: bool = False,
    ) -> TyArrayBase:
        # See https://github.com/onnx/onnx/issues/6590 for the
        # discussion on adding an efficient implementation to the
        # standard
        return (
            self.log()
            .cumulative_sum(axis=axis, dtype=dtype, include_initial=include_initial)
            .exp()
        )

    def cumulative_sum(
        self,
        /,
        *,
        axis: int | None = None,
        dtype: DType | None = None,
        include_initial: bool = False,
    ) -> TyArrayBase:
        dtype_ = self._accumulation_dtype(dtype)
        if self.dtype != dtype_:
            return self.astype(dtype_).cumulative_sum(
                axis=axis, include_initial=include_initial, dtype=dtype_
            )

        if axis is None:
            if self.ndim > 1:
                raise ValueError(
                    "'axis' parameter must be provided for arrays with more than one dimension"
                )
            axis_ = 0
        else:
            axis_ = axis

        out = safe_cast(
            type(self),
            _var_to_tyarray(
                op.cumsum(
                    self._var,
                    op.const(axis_, np.int64),
                    exclusive=False,
                )
            ),
        )
        if include_initial:
            out_shape = out.dynamic_shape
            out_shape[axis_] = const(1, dtype=int64)
            zeros = safe_cast(type(self), const(0, out.dtype).broadcast_to(out_shape))
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
            self._var,
            axis_,
            keepdims=keepdims,
            noop_with_empty_axes=False,
        )
        return _var_to_tyarray(var)

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
            op.reduce_max(
                self._var, axes=axes, keepdims=keepdims, noop_with_empty_axes=False
            )
        )

    def min(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Self:
        if axis == ():
            return self.copy()

        axes = _axis_var(axis, self.ndim)
        return type(self)(
            op.reduce_min(
                self._var, axes=axes, keepdims=keepdims, noop_with_empty_axes=False
            )
        )

    @overload
    def prod(
        self,
        /,
        *,
        dtype: DType[TY_ARRAY_BASE_co],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> TY_ARRAY_BASE_co: ...

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
            "prod", op.reduce_prod, axis=axis, dtype=dtype, keepdims=keepdims
        )

    def sort(
        self, /, *, axis: int = -1, descending: bool = False, stable: bool = True
    ) -> Self:
        if axis < 0:
            axis += self.ndim
        k = self.dynamic_shape[axis : axis + 1]._var  # k must be 1D, thus the slice
        values, _indices = op.top_k(
            self._var, k, axis=axis, largest=descending, sorted=stable
        )
        return type(self)(values)

    def diff(
        self,
        /,
        *,
        axis: int = -1,
        n: int = 1,
        prepend: Self | None = None,
        append: Self | None = None,
    ) -> Self:
        # TODO: Add NOTICE; taken from numpy.diff
        a = self
        if n == 0:
            return a
        if n < 0:
            raise ValueError("order must be non-negative but got " + repr(n))

        ndim = a.ndim
        if ndim == 0:
            raise ValueError("diff requires input that is at least one dimensional")
        axis = axis if axis >= 0 else ndim + axis

        combined = []
        if prepend is not None:
            if prepend.ndim == 0:
                shape = a.dynamic_shape.copy()
                shape[axis] = const(1, int64)
                prepend = prepend.broadcast_to(shape)
            combined.append(prepend)

        combined.append(a)

        if append is not None:
            if append.ndim == 0:
                shape = a.dynamic_shape.copy()
                shape[axis] = const(1, int64)
                append = append.broadcast_to(shape)
            combined.append(append)

        if len(combined) > 1:
            a = combined[0].concat(combined[1:], axis=axis)

        slice1 = [slice(None)] * ndim
        slice2 = [slice(None)] * ndim
        slice1[axis] = slice(1, None)
        slice2[axis] = slice(None, -1)

        for _ in range(n):
            a = a[tuple(slice1)] - a[tuple(slice2)]

        return a

    @overload
    def sum(
        self,
        /,
        *,
        dtype: DType[TY_ARRAY_BASE_co],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> TY_ARRAY_BASE_co: ...

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
            "sum", op.reduce_sum, axis=axis, dtype=dtype, keepdims=keepdims
        )

    def __abs__(self) -> Self:
        return type(self)(op.abs(self._var))

    def __neg__(self) -> TyArray:
        return type(self)(op.neg(self._var))

    def __pos__(self) -> Self:
        return self.copy()

    def _apply(
        self,
        other: TyArrayBase | PyScalar,
        op: Callable[[Var, Var], Var],
        forward: bool,
        result_type: type[TY_ARRAY],
    ) -> TY_ARRAY | NotImplementedType:
        if isinstance(other, TyArrayNumber | int | float):
            if forward:
                a, b = promote(self, other)
            else:
                b, a = promote(self, other)
            var = op(a._var, b._var)
            return safe_cast(result_type, _var_to_tyarray(var))
        return NotImplemented

    @overload
    def __add__(self: Self, other: Self | int) -> Self: ...

    @overload
    def __add__(self, other: TyArrayNumber | int | float) -> TyArrayNumber: ...

    @overload
    def __add__(self, other: TyArrayBase | PyScalar) -> TyArrayBase: ...

    def __add__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return self._apply(other, op.add, forward=True, result_type=TyArrayNumber)

    def __radd__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return self._apply(other, op.add, forward=False, result_type=TyArrayNumber)

    @overload
    def __ge__(self, other: TyArrayNumber | int | float) -> TyArrayBool: ...

    @overload
    def __ge__(self, other: TyArrayBase | PyScalar) -> TyArrayBase: ...

    def __ge__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return self._apply(
            other, op.greater_or_equal, forward=True, result_type=TyArrayBool
        )

    @overload
    def __gt__(self, other: TyArrayNumber | int | float) -> TyArrayBool: ...

    @overload
    def __gt__(self, other: TyArrayBase | PyScalar) -> TyArrayBase: ...

    def __gt__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return self._apply(other, op.greater, forward=True, result_type=TyArrayBool)

    @overload
    def __le__(self, other: TyArrayNumber | int | float) -> TyArrayBool: ...

    @overload
    def __le__(self, other: TyArrayBase | PyScalar) -> TyArrayBase: ...

    def __le__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return self._apply(
            other, op.less_or_equal, forward=True, result_type=TyArrayBool
        )

    @overload
    def __lt__(self, other: TyArrayNumber | int | float) -> TyArrayBool: ...

    @overload
    def __lt__(self, other: TyArrayBase | PyScalar) -> TyArrayBase: ...

    def __lt__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return self._apply(other, op.less, forward=True, result_type=TyArrayBool)

    def __matmul__(self, other) -> TyArrayBase:
        return self._apply(other, op.matmul, forward=True, result_type=TyArrayNumber)

    def __mul__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return self._apply(other, op.mul, forward=True, result_type=TyArrayNumber)

    def __rmul__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return self._apply(other, op.mul, forward=False, result_type=TyArrayNumber)

    def __pow__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return self._apply(other, op.pow, forward=True, result_type=TyArrayNumber)

    def __rpow__(self, other) -> TyArrayBase:
        return self._apply(other, op.pow, forward=False, result_type=TyArrayNumber)

    @overload
    def __sub__(self: Self, other: Self | int, /) -> Self: ...

    @overload
    def __sub__(self, other: TyArrayNumber | int | float) -> TyArrayNumber: ...

    @overload
    def __sub__(self, other: TyArrayBase | PyScalar) -> TyArrayBase: ...

    def __sub__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return self._apply(other, op.sub, forward=True, result_type=TyArrayNumber)

    def __rsub__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return self._apply(other, op.sub, forward=False, result_type=TyArrayNumber)

    def __truediv__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return self._apply(other, op.div, forward=True, result_type=TyArrayNumber)

    def __rtruediv__(self, other) -> TyArrayBase:
        return self._apply(other, op.div, forward=False, result_type=TyArrayNumber)

    def __floordiv__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        if isinstance(other, TyArrayNumber | int | float):
            promo_result, _ = promote(self, other)
            return (self / other).floor().astype(promo_result.dtype)
        return NotImplemented

    def __rfloordiv__(self, other) -> TyArrayBase:
        if isinstance(other, int | float):
            promo_result, _ = promote(self, other)
            return (other / self).floor().astype(promo_result.dtype)
        return NotImplemented

    def sign(self) -> Self:
        return type(self)(op.sign(self._var))

    def sqrt(self) -> Self:
        return type(self)(op.sqrt(self._var))

    def square(self) -> TyArray:
        return safe_cast(TyArray, self**2)

    @overload
    def __ndx_maximum__(self, rhs: Self, /) -> Self: ...

    @overload
    def __ndx_maximum__(self, rhs: TyArrayNumber | int | float, /) -> TyArrayNumber: ...

    @overload
    def __ndx_maximum__(self, rhs: TyArrayBase | PyScalar, /) -> TyArrayBase: ...

    def __ndx_maximum__(self, rhs: TyArrayBase | PyScalar, /) -> TyArrayBase:
        return self._apply(
            rhs,
            lambda a, b: op.max([a, b]),
            forward=True,
            result_type=TyArrayNumber,
        )

    @overload
    def __ndx_minimum__(self, rhs: Self, /) -> Self: ...

    @overload
    def __ndx_minimum__(self, rhs: TyArrayNumber | int | float, /) -> TyArrayNumber: ...

    @overload
    def __ndx_minimum__(self, rhs: TyArrayBase | PyScalar, /) -> TyArrayBase: ...

    def __ndx_minimum__(self, rhs: TyArrayBase | PyScalar, /) -> TyArrayBase:
        return self._apply(
            rhs,
            lambda a, b: op.min([a, b]),
            forward=True,
            result_type=TyArrayNumber,
        )

    def __ndx_tensordot__(
        self,
        other: TyArrayBase,
        /,
        *,
        axes: int | tuple[Sequence[int], Sequence[int]] = 2,
    ) -> TyArrayNumber:
        if not isinstance(other, TyArray):
            return NotImplemented

        if self.dtype != other.dtype:
            x1, x2 = promote(self, other)
            res = x1.__ndx_tensordot__(x2, axes=axes)
            return safe_cast(TyArrayNumber, res)

        def letter():
            for i in range(ord("a"), ord("z") + 1):
                yield chr(i)
            raise ValueError(
                "exceeded available letters for einsum equation in the implementation of 'tensordot': this means that the number of dimensions of 'x1' and 'x2' are too large"
            )

        letter_gen = letter()

        if self.ndim == 0 or other.ndim == 0:
            return safe_cast(TyArrayNumber, self * other)

        if isinstance(axes, int):
            axes = (
                [-axes + i for i in range(axes)],
                [i for i in range(axes)],
            )

        axes_a, axes_b = axes

        axes_a = [(ax + self.ndim) if ax < 0 else ax for ax in axes_a]
        axes_b = [(bx + other.ndim) if bx < 0 else bx for bx in axes_b]

        a_letters = [next(letter_gen) for _ in range(self.ndim)]

        b_letters = [
            a_letters[axes_a[axes_b.index(bx)]] if bx in axes_b else next(letter_gen)
            for bx in range(other.ndim)
        ]

        joint_letters = [
            let for idx, let in enumerate(a_letters) if idx not in axes_a
        ] + [let for idx, let in enumerate(b_letters) if idx not in axes_b]

        equation = (
            f"{''.join(a_letters)},{''.join(b_letters)}->{''.join(joint_letters)}"
        )

        var = op.einsum([self._var, other._var], equation=equation)

        return self.dtype._build(var)


class TyArrayInteger(TyArrayNumber):
    def _apply_int_only(
        self,
        other: TyArrayBase | PyScalar,
        op: Callable[[Var, Var], Var],
        forward: bool,
    ) -> TyArrayInteger | NotImplementedType:
        if isinstance(other, TyArrayInteger | TyArrayBool | int | bool):
            if forward:
                a, b = promote(self, other)
            else:
                b, a = promote(self, other)
            var = op(a._var, b._var)
            return safe_cast(TyArrayInteger, _var_to_tyarray(var))
        return NotImplemented

    def __truediv__(self, rhs: TyArrayBase | PyScalar) -> TyArrayBase:
        # Casting rules are implementation defined. We default to float64 like NumPy
        if isinstance(rhs, TyArrayNumber):
            return self.astype(float64) / rhs.astype(float64)
        if isinstance(rhs, int):
            rhs = float(rhs)
        return super().__truediv__(rhs)

    def __rtruediv__(self, lhs: TyArrayBase | PyScalar) -> TyArrayBase:
        # Casting rules are implementation defined. We default to float64 like NumPy
        if isinstance(lhs, TyArrayNumber):
            return lhs.astype(float64) / self.astype(float64)
        if isinstance(lhs, int):
            lhs = float(lhs)
        return super().__rtruediv__(lhs)

    def __and__(self, other) -> TyArrayBase:
        return self._apply_int_only(other, op.bitwise_and, forward=True)

    def __rand__(self, other) -> TyArrayBase:
        return self._apply_int_only(other, op.bitwise_and, forward=False)

    def __or__(self, other) -> TyArrayBase:
        return self._apply_int_only(other, op.bitwise_or, forward=True)

    def __ror__(self, other) -> TyArrayBase:
        return self._apply_int_only(other, op.bitwise_or, forward=False)

    def __mod__(self, other) -> TyArrayInteger:
        return self._apply_int_only(
            other, lambda a, b: op.mod(a, b, fmod=0), forward=True
        )

    def __rmod__(self, other) -> TyArrayInteger:
        return self._apply_int_only(
            other, lambda a, b: op.mod(a, b, fmod=0), forward=False
        )

    def __xor__(self, other) -> TyArrayBase:
        return self._apply_int_only(other, op.bitwise_xor, forward=True)

    def __rxor__(self, other) -> TyArrayBase:
        return self._apply_int_only(other, op.bitwise_xor, forward=False)

    def __lshift__(self, other) -> TyArrayBase:
        return self._apply_int_only(
            other,
            lambda a, b: op.bit_shift(a, b, direction="LEFT"),
            forward=True,
        )

    def __rlshift__(self, other) -> TyArrayBase:
        return self._apply_int_only(
            other,
            lambda a, b: op.bit_shift(a, b, direction="LEFT"),
            forward=False,
        )

    def __invert__(self) -> Self:
        res = op.bitwise_not(self._var)
        return type(self)(res)

    def ceil(self) -> Self:
        return self.copy()

    def cumulative_prod(
        self,
        /,
        *,
        axis: int | None = None,
        dtype: DType | None = None,
        include_initial: bool = False,
    ) -> TyArrayBase:
        # Requires special handling since the current implementation
        # is using log/exp which does not exist for integer data types
        if isinstance(dtype, Integer | None):
            res = self.astype(float64).cumulative_prod(
                axis=axis, include_initial=include_initial
            )
            if dtype is None:
                if isinstance(self, TyArrayUnsignedInteger):
                    return res.astype(uint64)
                return res.astype(int64)
            return res.astype(dtype)
        return super().cumulative_prod(
            axis=axis, dtype=dtype, include_initial=include_initial
        )

    def floor(self) -> Self:
        return self.copy()

    def round(self) -> Self:
        return self.copy()

    def isfinite(self) -> TyArrayBool:
        return TyArrayBool(op.const(True)).broadcast_to(self.dynamic_shape)

    def isnan(self) -> TyArrayBool:
        return TyArrayBool(op.const(False)).broadcast_to(self.dynamic_shape)

    def isinf(self) -> TyArrayBool:
        return TyArrayBool(op.const(False)).broadcast_to(self.dynamic_shape)

    def trunc(self) -> Self:
        return self.copy()


class TyArraySignedInteger(TyArrayInteger):
    # The array-api standard defines the right shift as arithmetic
    # (i.e. sign-propagating). The ONNX standard is logical.

    def __rshift__(self, other) -> TyArray:
        if isinstance(other, TyArrayInteger | int):
            return safe_cast(TyArray, self // (2**other))
        return NotImplemented

    def __rrshift__(self, other) -> TyArray:
        if isinstance(other, int):
            return safe_cast(TyArray, other // (2**self))
        return NotImplemented


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

    def __rshift__(self, other) -> TyArrayBase:
        # The array-api standard defines the right shift as arithmetic
        # (i.e. sign-propagating). The ONNX standard is logical. We
        # can use a more efficient implementation if we know that
        # there is no sign.
        if isinstance(other, TyArrayInteger | int):
            a, b = promote(self, other)
            var = op.bit_shift(a._var, b._var, direction="RIGHT")
            return safe_cast(TyArrayInteger, _var_to_tyarray(var))
        return super().__rshift__(other)

    def __rrshift__(self, other) -> TyArrayBase:
        if isinstance(other, TyArrayUnsignedInteger | int):
            b, a = promote(self, other)
            var = op.bit_shift(a._var, b._var, direction="RIGHT")
            return safe_cast(TyArrayUnsignedInteger, _var_to_tyarray(var))
        return NotImplemented


class TyArrayFloating(TyArrayNumber):
    def __mod__(self, other) -> TyArrayBase:
        if isinstance(other, TyArrayFloating | float):
            # This function is complicated for two reasons:
            # 1. The ONNX standard is undefined if dividend is 0, but the array-api is not.
            # 2. The array-api follows the Python semantics, which are rather odd.
            a, b = promote(self, other)
            var = op.mod(a._var, b._var, fmod=1)
            mod = safe_cast(TyArrayFloating, _var_to_tyarray(var))
            # NOTE: onnxruntime appears to have a bug where the sign
            # of zeros is only preserved if they are on the
            # false-branch!
            # TODO: File a bug!
            fixed_mod = where((b < 0) == (mod < 0), mod, mod + b)
            fixed_zeros = where(b > 0, const(0.0, mod.dtype), const(-0.0, mod.dtype))
            return where(
                safe_cast(TyArrayBool, ~((mod == 0.0) & (b != 0.0))),
                fixed_mod,
                fixed_zeros,
            )

        return super().__mod__(other)

    def __rmod__(self, other) -> TyArrayBase:
        if isinstance(other, TyArrayFloating | float):
            b, a = promote(self, other)
            var = op.mod(a._var, b._var, fmod=1)
            return safe_cast(TyArrayFloating, _var_to_tyarray(var))
        return super().__mod__(other)

    def __ndx_logaddexp__(self, x2: TyArrayBase | int | float, /) -> TyArrayFloating:
        if isinstance(x2, TyArrayNumber | int | float):
            x1, x2 = promote(self, x2)
            return safe_cast(TyArrayFloating, (x1.exp() + x2.exp()).log())
        return NotImplemented

    def __ndx_rlogaddexp__(self, x1: TyArrayBase | int | float, /) -> TyArrayFloating:
        if isinstance(x1, TyArrayNumber | int | float):
            x2, x1 = promote(self, x1)
            return safe_cast(TyArrayFloating, (x1.exp() + x2.exp()).log())
        return NotImplemented

    def ceil(self) -> Self:
        return type(self)(op.ceil(self._var))

    def floor(self) -> Self:
        return type(self)(op.floor(self._var))

    def reciprocal(self) -> Self:
        return type(self)(op.reciprocal(self._var))

    def round(self) -> Self:
        return type(self)(op.round(self._var))

    def max(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Self:
        # ONNX standard does not define nan handling so we have to do
        # some special handling for floating points
        res = super().max(axis=axis, keepdims=keepdims)

        is_nan = self.isnan().any(axis=axis, keepdims=keepdims)
        return safe_cast(
            type(self), where(is_nan, const(float("nan"), dtype=res.dtype), res)
        )

    def min(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Self:
        # ONNX standard does not define nan handling so we have to do
        # some special handling for floating points

        res = super().min(axis=axis, keepdims=keepdims)

        is_nan = self.isnan().any(axis=axis, keepdims=keepdims)
        return safe_cast(
            type(self), where(is_nan, const(float("nan"), dtype=res.dtype), res)
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
            op.reduce_sum(
                self._var, axes=axes, keepdims=keepdims, noop_with_empty_axes=False
            )
        )
        in_shape = self.dynamic_shape
        axis_ = _axis_array(axis, self.ndim)
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
            size = safe_cast(TyArrayInt64, size - const(correction, dtype=int64))
        res = (nom / size).astype(self.dtype)

        # There is an uncanny corner case here: If self is 0-sized,
        # means is NaN, however, the sum over a zero-sized array is
        # `0`! I.e. we lose the NaN information. We recover it with
        # the `where` call later.
        nan = const(np.nan, dtype=self.dtype)
        is_zero_sized = safe_cast(
            TyArrayBool,
            self.dynamic_shape.prod(dtype=int64) == const(0, dtype=int64),
        )
        return safe_cast(type(self), where(is_zero_sized, nan, res))

    def isfinite(self) -> TyArrayBool:
        return safe_cast(TyArrayBool, ~(self.isinf() | self.isnan()))

    def isinf(self) -> TyArrayBool:
        return TyArrayBool(op.isinf(self._var))

    def isnan(self) -> TyArrayBool:
        return TyArrayBool(op.isnan(self._var))

    # Element-wise for floating point
    def acos(self) -> Self:
        return type(self)(op.acos(self._var))

    def acosh(self) -> Self:
        return type(self)(op.acosh(self._var))

    def asin(self) -> Self:
        return type(self)(op.asin(self._var))

    def asinh(self) -> Self:
        return type(self)(op.asinh(self._var))

    def atan(self) -> Self:
        return type(self)(op.atan(self._var))

    def atanh(self) -> Self:
        return type(self)(op.atanh(self._var))

    def cos(self) -> Self:
        return type(self)(op.cos(self._var))

    def cosh(self) -> Self:
        return type(self)(op.cosh(self._var))

    def exp(self) -> Self:
        return type(self)(op.exp(self._var))

    def log(self) -> Self:
        return type(self)(op.log(self._var))

    def log2(self) -> Self:
        res = self.log() / float(np.log(2))
        return safe_cast(type(self), res)

    def log10(self) -> Self:
        res = self.log() / float(np.log(10))
        return safe_cast(type(self), res)

    def sin(self) -> Self:
        return type(self)(op.sin(self._var))

    def sinh(self) -> Self:
        return type(self)(op.sinh(self._var))

    def tan(self) -> Self:
        return type(self)(op.tan(self._var))

    def tanh(self) -> Self:
        return type(self)(op.tanh(self._var))

    def trunc(self) -> Self:
        return safe_cast(
            type(self),
            where(safe_cast(TyArrayBool, self < 0), self.ceil(), self.floor()),
        )


class TyArrayBool(TyArray):
    @property
    def dtype(self) -> Bool:
        return bool_

    def _apply(
        self,
        other: TyArrayBase | PyScalar,
        op: Callable[[Var, Var], Var],
        forward: bool,
    ) -> TyArrayBool | NotImplementedType:
        if isinstance(other, TyArrayBool | bool):
            if forward:
                a, b = promote(self, other)
            else:
                b, a = promote(self, other)
            var = op(a._var, b._var)
            return TyArrayBool(var)
        return NotImplemented

    def _apply_arithmetic(
        self,
        other: TyArrayBase | PyScalar,
        op: Callable[[Var, Var], Var],
        forward: bool,
    ) -> TyArrayNumber | NotImplementedType:
        if isinstance(other, TyArrayNumber | TyArrayBool | int | float):
            if forward:
                a, b = promote(self, other)
            else:
                b, a = promote(self, other)
            var = op(a._var, b._var)
            return safe_cast(TyArrayNumber, _var_to_tyarray(var))
        return NotImplemented

    # We would get the following arithmetic functions for free if
    # booleans were a subtype of `_Numbers`, but that may come with
    # other undesirable properties. Thus, we do a bit more typing here
    # in order to properly keep booleans separate.
    def __add__(self, other: TyArrayBase | PyScalar) -> TyArrayNumber:
        return self._apply_arithmetic(other, op.add, forward=True)

    def __radd__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return self._apply_arithmetic(other, op.add, forward=False)

    def __sub__(self, other: TyArrayBase | PyScalar) -> TyArrayNumber:
        return self._apply_arithmetic(other, op.sub, forward=True)

    def __rsub__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return self._apply_arithmetic(other, op.sub, forward=False)

    def __mul__(self, other: TyArrayBase | PyScalar) -> TyArrayNumber:
        return self._apply_arithmetic(other, op.mul, forward=True)

    def __rmul__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return self._apply_arithmetic(other, op.mul, forward=False)

    @overload
    def __or__(self, other: TyArrayBool) -> TyArrayBool: ...

    @overload
    def __or__(self, other) -> TyArrayBase: ...
    def __or__(self, other) -> TyArrayBase:
        return self._apply(other, op.or_, forward=True)

    def __ror__(self, other) -> TyArrayBase:
        return self._apply(other, op.or_, forward=False)

    def __xor__(self, other) -> TyArrayBase:
        return self._apply(other, op.xor, forward=True)

    def __rxor__(self, other) -> TyArrayBase:
        return self._apply(other, op.xor, forward=False)

    @overload
    def __and__(self, other: TyArrayBool) -> TyArrayBool: ...

    @overload
    def __and__(self, other) -> TyArrayBase: ...
    def __and__(self, other) -> TyArrayBase:
        return self._apply(other, op.and_, forward=True)

    def __rand__(self, other) -> TyArrayBase:
        return self._apply(other, op.and_, forward=False)

    def __invert__(self) -> Self:
        return type(self)(op.not_(self._var))

    def __ndx_logical_and__(self, other: TyArrayBase | PyScalar, /) -> Self:
        return type(self)(self._apply(other, op.and_, forward=True)._var)

    def __ndx_rlogical_and__(self, other: TyArrayBase | PyScalar, /) -> Self:
        return type(self)(self._apply(other, op.and_, forward=False)._var)

    def __ndx_logical_xor__(self, other: TyArrayBase | PyScalar, /) -> Self:
        return type(self)(self._apply(other, op.xor, forward=True)._var)

    def __ndx_rlogical_xor__(self, other: TyArrayBase | PyScalar, /) -> Self:
        return type(self)(self._apply(other, op.xor, forward=False)._var)

    def __ndx_logical_or__(self, other: TyArrayBase | PyScalar, /) -> Self:
        return type(self)(self._apply(other, op.or_, forward=True)._var)

    def __ndx_rlogical_or__(self, other: TyArrayBase | PyScalar, /) -> Self:
        return type(self)(self._apply(other, op.or_, forward=False)._var)

    def nonzero(self) -> tuple[TyArrayInt64, ...]:
        # Use numeric implementation
        return self.astype(uint8).nonzero()

    def isnan(self) -> TyArrayBool:
        return TyArrayBool(op.const(False)).broadcast_to(self.dynamic_shape)

    def logical_not(self) -> Self:
        return ~self

    def static_map(self, mapping: Mapping[KEY, VALUE], default: VALUE) -> TyArray:
        mapping_ = {bool(k): v for k, v in mapping.items()}

        true_val = mapping_.get(True, default)
        false_val = mapping_.get(False, default)

        return safe_cast(TyArray, where(self, const(true_val), const(false_val)))


class TyArrayInt8(TyArraySignedInteger):
    @property
    def dtype(self) -> Int8:
        return int8


class TyArrayInt16(TyArraySignedInteger):
    @property
    def dtype(self) -> Int16:
        return int16


class TyArrayInt32(TyArraySignedInteger):
    @property
    def dtype(self) -> Int32:
        return int32


class TyArrayInt64(TyArraySignedInteger):
    @property
    def dtype(self) -> Int64:
        return int64


class TyArrayUInt8(TyArrayUnsignedInteger):
    @property
    def dtype(self) -> UInt8:
        return uint8


class TyArrayUInt16(TyArrayUnsignedInteger):
    @property
    def dtype(self) -> UInt16:
        return uint16


class TyArrayUInt32(TyArrayUnsignedInteger):
    @property
    def dtype(self) -> UInt32:
        return uint32


class TyArrayUInt64(TyArrayUnsignedInteger):
    @property
    def dtype(self) -> UInt64:
        return uint64


class TyArrayFloat16(TyArrayFloating):
    @property
    def dtype(self) -> Float16:
        return float16


class TyArrayFloat32(TyArrayFloating):
    @property
    def dtype(self) -> Float32:
        return float32


class TyArrayFloat64(TyArrayFloating):
    @property
    def dtype(self) -> Float64:
        return float64


# TODO: Make private
def _var_to_tyarray(var: Var) -> TyArray:
    dtype = from_numpy_dtype(var.unwrap_tensor().dtype)

    return dtype._build(var)


@overload
def const(
    obj: bool | int | float | str | np.ndarray, dtype: _OnnxDType[TY_ARRAY_co]
) -> TY_ARRAY_co: ...


@overload
def const(obj: int, dtype: None = None) -> TyArrayInt64: ...


@overload
def const(
    obj: bool | int | float | str | np.ndarray, dtype: _OnnxDType | None = None
) -> TyArray: ...


def const(
    obj: bool | int | float | str | np.ndarray, dtype: _OnnxDType | None = None
) -> TyArray:
    """Create a constant from the given value.

    The `dtype` argument may be used in favor of a subsequent `astype` call to create a cleaner ONNX graph.
    """
    # don't blindly fall back to NumPy to maintain better np1x
    # compatibility on Windows which defaults to int32
    if isinstance(obj, bool):
        obj = np.asarray(obj, dtype=bool)
    if isinstance(obj, int):
        obj = np.asarray(obj, dtype=np.int64)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        if not all(isinstance(el, str) for el in obj.flatten()):
            raise ValueError(
                "found array with object dtype but it contains non-string elements"
            )
        obj = obj.astype(str)
    else:
        obj = np.asarray(obj)
    if dtype is not None:
        obj = np.asarray(obj, dtype=dtype.unwrap_numpy())
    return _var_to_tyarray(op.const(obj))


def _contains_no_ellipsis(
    seq: tuple[SetitemItem, ...],
) -> TypeGuard[tuple[int | slice, ...]]:
    return all(not isinstance(el, EllipsisType) for el in seq)


def _is_integer_array_indexing(
    key: GetitemIndex,
) -> TypeGuard[tuple[int | TyArrayInt64, ...]]:
    if not isinstance(key, tuple):
        return False
    for el in key:
        if isinstance(el, int):
            continue
        if isinstance(el, TyArrayInt64):
            continue
        return False
    return True


def _is_single_axis_indexing(
    key: GetitemIndex,
) -> TypeGuard[int | _ScalarInt | slice | EllipsisType | None]:
    if isinstance(key, TyArrayInteger) and key.ndim == 0:
        return True
    return isinstance(key, int | slice | EllipsisType | None)


def _is_multi_axis_indexing(
    key: GetitemIndex,
) -> TypeGuard[tuple[int | _ScalarInt | slice | EllipsisType | None, ...]]:
    if not isinstance(key, tuple):
        return False
    for el in key:
        if isinstance(el, TyArrayInteger) and el.ndim == 0:
            continue
        elif isinstance(el, int | slice | EllipsisType | None):
            continue
        return False
    return True


#####################
# Conversion tables #
#####################

# Promotion tables taken from:
# https://data-apis.org/array-api/draft/API_specification/type_promotion.html#type-promotion
# and
# https://numpy.org/neps/nep-0050-scalar-promotion.html#motivation-and-scope
_signed_signed: dict[tuple[_Number, _Number], NumericDTypes] = {
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
_unsigned_unsigned: dict[tuple[_Number, _Number], NumericDTypes] = {
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
_mixed_integers: dict[tuple[_Number, _Number], NumericDTypes] = {
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

_floating_floating: dict[tuple[_Number, _Number], NumericDTypes] = {
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
_singed_int_with_uint64: dict[tuple[_Number, _Number], NumericDTypes] = {
    (int8, uint64): float64,
    (int16, uint64): float64,
    (int32, uint64): float64,
    (int64, uint64): float64,
}
_singed_int_with_uint64 |= {(k[1], k[0]): v for k, v in _singed_int_with_uint64.items()}

_bool_with_number: dict[tuple[_Number | Bool, _Number | Bool], NumericDTypes] = {
    (bool_, float32): float32,
    (bool_, float64): float64,
    (bool_, int8): int8,
    (bool_, int16): int16,
    (bool_, int32): int32,
    (bool_, int64): int64,
    (bool_, uint8): uint8,
    (bool_, uint16): uint16,
    (bool_, uint32): uint32,
    (bool_, uint64): uint64,
}
_bool_with_number |= {(k[1], k[0]): v for k, v in _bool_with_number.items()}

# Mixed integers and floating point numbers are not
# defined by the standard.
_int_to_floating: dict[_Number | Bool, NumericDTypes] = {
    bool_: float32,
    int8: float32,
    uint8: float32,
    int16: float32,
    uint16: float32,
    int32: float64,
    uint32: float64,
    int64: float64,
    uint64: float64,
}


def result_type(
    a: _OnnxDType, b: _OnnxDType | int | float | str
) -> _OnnxDType | NotImplementedType:
    """Entry point for promotion logic between ONNX data types."""
    if a == b:
        return a
    if isinstance(b, str):
        if isinstance(a, Utf8):
            return a
        return NotImplemented

    if isinstance(b, bool):
        if isinstance(a, Bool):
            return a
        b = bool_

    if isinstance(b, int | float):
        if isinstance(a, Bool) and isinstance(b, int):
            return int64
        if isinstance(a, Bool) and isinstance(b, float):
            return float64
        if isinstance(a, _Number):
            if isinstance(b, int):
                return a
            if isinstance(a, Integer) and isinstance(b, float):
                return float64
            if isinstance(a, Floating):
                return a
        return NotImplemented

    if isinstance(a, Integer) and isinstance(b, Floating):
        a = _int_to_floating[a]
    if isinstance(a, Floating) and isinstance(b, Integer):
        b = _int_to_floating[b]
    if isinstance(a, Bool | _Number) and isinstance(b, Bool | _Number):
        if ret := _bool_with_number.get((a, b)):
            return ret

    if isinstance(a, _Number) and isinstance(b, _Number):
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
        if ret := _singed_int_with_uint64.get((a, b)):
            return ret
    return NotImplemented


@overload
def promote(
    lhs: TyArrayFloating, *others: TyArrayFloating | int | float
) -> tuple[TyArrayFloating, ...]: ...


@overload
def promote(
    lhs: TyArray, *others: TyArray | bool | int | float | str
) -> tuple[TyArray, ...]: ...


def promote(
    lhs: TyArray, *others: TyArray | bool | int | float | str
) -> tuple[TyArray, ...]:
    others_ = [el.dtype if isinstance(el, TyArray) else el for el in others]
    dtype = lhs.dtype
    for el in others_:
        dtype_ = result_type(dtype, el)
        if dtype_ is NotImplemented:
            raise TypeError(
                f"failed to promote `{dtype}` and `{el}` into common data type"
            )
        dtype = dtype_

    return tuple(
        el.astype(dtype) if isinstance(el, TyArray) else const(el, dtype)
        for el in [lhs] + list(others)
    )


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
    return res._var


def _to_slice_kwargs(slices: list[FancySlice]) -> dict[str, Var]:
    if all(el.is_noop() for el in slices):
        return {}

    transposed = zip(
        *[(el.start, el.stop, el.step) for el in slices if not el.is_noop()]
    )

    def concat(lst: tuple[TyArrayInt64, ...]) -> TyArrayInt64:
        # This concatenation is a bit of a hot spot in the test suite.
        # Thus, we use some hacky constant folding here to skip some
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
    kw = {
        k: concat(lst)._var for k, lst in zip(["starts", "ends", "steps"], transposed)
    }

    kw["axes"] = TyArrayInt64(
        op.const([i for i, el in enumerate(slices) if not el.is_noop()], np.int64)
    )._var
    return kw


def _move_ellipsis_back(
    ndim: int,
    key: tuple[SetitemItem, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[SetitemItem, ...]]:
    """Permute axes such that the ellipsis-axes are at the end."""

    if ... not in key:
        raise ValueError("No ellipsis found in 'key'")
    ellipsis_pos = key.index(...)
    current_dims = list(range(ndim))
    ellipsis_len = ndim - (len(key) - 1)
    new_perm = tuple(
        current_dims[:ellipsis_pos]
        + current_dims[ellipsis_pos + ellipsis_len :]
        + current_dims[ellipsis_pos:][:ellipsis_len]
    )
    inverse_map = tuple(int(el) for el in np.argsort(new_perm))

    keys_no_ellipsis = tuple(el for el in key if el != ...)
    return new_perm, inverse_map, keys_no_ellipsis


T = TypeVar("T")


def from_numpy_dtype(np_dtype: np.dtype) -> DTypes:
    # Ensure that this also works with np.generic such as np.int64
    np_dtype = np.dtype(np_dtype)

    if np_dtype == np.int8:
        return int8
    if np_dtype == np.int16:
        return int16
    if np_dtype == np.int32:
        return int32
    if np_dtype == np.int64:
        return int64

    if np_dtype == np.uint8:
        return uint8
    if np_dtype == np.uint16:
        return uint16
    if np_dtype == np.uint32:
        return uint32
    if np_dtype == np.uint64:
        return uint64

    if np_dtype == np.float16:
        return float16
    if np_dtype == np.float32:
        return float32
    if np_dtype == np.float64:
        return float64

    if np_dtype == np.bool_:
        return bool_

    # We don't support "T" ("Text" is the kind used for `StringDType` in numpy >= 2), yet.
    # See https://numpy.org/neps/nep-0055-string_dtype.html#python-api-for-stringdtype
    if np_dtype.kind == "U":
        return utf8

    raise ValueError(f"`{np_dtype}` does not have a corresponding ndonnx data type")


def _key_to_indices(key: tuple[slice | int, ...], shape: TyArrayInt64) -> TyArrayInt64:
    """Compute expanded indices for ``key`` for an array of shape ``shape``."""
    if len(key) == 0:
        raise NotImplementedError("indexing with an empty key is not yet supported")
    # TODO: Allow dynamic inputs to DType.__ndx_arange__?
    indices_per_dim = []
    for s, length in zip(key, shape):
        if isinstance(s, int):
            stop = None if s == -1 else s + 1
            s = slice(s, stop, 1)
        indices_per_dim.append(
            _var_to_tyarray(op.range(*[el._var for el in _get_indices(s, length)]))
        )

    grid = meshgrid(*indices_per_dim, indexing="ij")
    first, *others = [el.reshape((-1, 1)) for el in grid]
    res = first.concat(others, axis=-1)
    return safe_cast(TyArrayInt64, res)


def _get_indices(
    s: slice, length: int | TyArrayInt64
) -> tuple[TyArrayInt64, TyArrayInt64, TyArrayInt64]:
    """Array-compatible implementation of ``slice.indices``.

    Slice members must be of type ``int`` or ``None`` while `length` may be an array.
    """
    length_ = length if isinstance(length, TyArrayInt64) else const(length, int64)

    step = 1 if s.step is None else s.step
    if not isinstance(step, int):
        raise IndexError(f"'step' must be of type 'int' found `{type(step)}`")

    if step == 0:
        raise IndexError("'step' must not be zero")

    step_is_negative = step < 0

    if step_is_negative:
        lower = const(-1, int64)
        upper = length_ + lower
    else:
        lower = const(0, int64)
        upper = length_

    if s.start is None:
        start_arr = upper if step_is_negative else lower
    elif isinstance(s.start, int):
        start_is_neg = s.start < 0
        slice_start = const(s.start, dtype=int64)
        if start_is_neg:
            start_arr = maximum(slice_start + length_, lower)
        else:
            start_arr = minimum(slice_start, upper)
    else:
        raise IndexError(f"'start' must be 'int' or 'None', found `{s.start}`")

    if s.stop is None:
        stop = lower if step_is_negative else upper
    elif isinstance(s.stop, int):
        stop = const(s.stop, dtype=int64)
        stop_is_neg = stop < 0
        if stop_is_neg:
            stop = maximum(stop + length_, lower)
        else:
            stop = minimum(stop, upper)
    else:
        raise IndexError(f"'stop' must be 'int' or 'None', found `{s.stop}`")

    return start_arr, stop, const(step, int64)


def meshgrid(*arrays: TyArrayBase, indexing: str = "xy") -> list[TyArrayBase]:
    ndim = len(arrays)

    if indexing not in ("xy", "ij"):
        raise ValueError(
            f"'indexing' argument must be one 'xy' or 'ij', found `{indexing}`"
        )

    base_shape = (1,) * ndim
    output = [
        x.reshape(base_shape[:i] + (-1,) + base_shape[i + 1 :])
        for i, x in enumerate(arrays)
    ]

    if indexing == "xy" and ndim > 1:
        # switch first and second axis
        output[0] = output[0].reshape((1, -1) + base_shape[2:])
        output[1] = output[1].reshape((-1, 1) + base_shape[2:])

    return list(broadcast_arrays(*output))


def broadcast_arrays(*arrays: TY_ARRAY_BASE) -> tuple[TY_ARRAY_BASE, ...]:
    if len(arrays) < 2:
        return tuple(a.copy() for a in arrays)

    def numeric_like(x: TyArrayBase) -> TyArrayInt32:
        # Using int32 rather than something smaller because we don't
        # want ort_compat to kick in here.
        return const(0, dtype=int32).broadcast_to(x.dynamic_shape)

    it = iter(arrays)
    ret = numeric_like(next(it))
    while (x := next(it, None)) is not None:
        ret = ret + numeric_like(x)

    target_shape = ret.dynamic_shape

    return tuple(a.broadcast_to(target_shape) for a in arrays)
