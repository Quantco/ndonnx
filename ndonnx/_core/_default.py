# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np

import ndonnx as ndx
import ndonnx._data_types as dtypes
import ndonnx._opset_extensions as opx

from ._utils import (
    from_corearray,
)


class OperationsBlock:
    """Interface for data types to implement top-level functions exported by ndonnx."""

    # elementwise.py

    def abs(self, x) -> ndx.Array:
        return NotImplemented

    def acos(self, x) -> ndx.Array:
        return NotImplemented

    def acosh(self, x) -> ndx.Array:
        return NotImplemented

    def add(self, x, y) -> ndx.Array:
        return NotImplemented

    def asin(self, x) -> ndx.Array:
        return NotImplemented

    def asinh(self, x) -> ndx.Array:
        return NotImplemented

    def atan(self, x) -> ndx.Array:
        return NotImplemented

    def atan2(self, y, x) -> ndx.Array:
        return ndx.atan(ndx.divide(y, x))

    def atanh(self, x) -> ndx.Array:
        return NotImplemented

    def bitwise_and(self, x, y) -> ndx.Array:
        return NotImplemented

    def bitwise_left_shift(self, x, y) -> ndx.Array:
        return NotImplemented

    def bitwise_invert(self, x) -> ndx.Array:
        return NotImplemented

    def bitwise_or(self, x, y) -> ndx.Array:
        return NotImplemented

    def bitwise_right_shift(self, x, y) -> ndx.Array:
        return NotImplemented

    def bitwise_xor(self, x, y) -> ndx.Array:
        return NotImplemented

    def ceil(self, x) -> ndx.Array:
        return NotImplemented

    def conj(*args):
        return NotImplemented

    def cos(self, x) -> ndx.Array:
        return NotImplemented

    def cosh(self, x) -> ndx.Array:
        return NotImplemented

    def divide(self, x, y) -> ndx.Array:
        return NotImplemented

    def equal(self, x, y) -> ndx.Array:
        return NotImplemented

    def exp(self, x) -> ndx.Array:
        return NotImplemented

    def expm1(self, x) -> ndx.Array:
        return ndx.subtract(ndx.exp(x), 1)

    def floor(self, x) -> ndx.Array:
        return NotImplemented

    def floor_divide(self, x, y) -> ndx.Array:
        return ndx.floor(ndx.divide(x, y))

    def greater(self, x, y) -> ndx.Array:
        return NotImplemented

    def greater_equal(self, x, y) -> ndx.Array:
        return NotImplemented

    def imag(*args):
        return NotImplemented

    def isfinite(self, x) -> ndx.Array:
        return ndx.logical_not(ndx.isinf(x))

    def isinf(self, x) -> ndx.Array:
        return NotImplemented

    def isnan(self, x) -> ndx.Array:
        return NotImplemented

    def less(self, x, y) -> ndx.Array:
        return NotImplemented

    def less_equal(self, x, y) -> ndx.Array:
        return NotImplemented

    def log(self, x) -> ndx.Array:
        return NotImplemented

    def log1p(self, x) -> ndx.Array:
        return ndx.add(ndx.log(x), ndx.asarray(1, x.dtype))

    def log2(self, x) -> ndx.Array:
        return ndx.log(x) / np.log(2)

    def log10(self, x) -> ndx.Array:
        return ndx.log(x) / np.log(10)

    def logaddexp(self, x, y) -> ndx.Array:
        return ndx.log(ndx.exp(x) + ndx.exp(y))

    def logical_and(self, x, y) -> ndx.Array:
        return NotImplemented

    def logical_not(self, x) -> ndx.Array:
        return NotImplemented

    def logical_or(self, x, y) -> ndx.Array:
        return NotImplemented

    def logical_xor(self, x, y) -> ndx.Array:
        return NotImplemented

    def multiply(self, x, y) -> ndx.Array:
        return NotImplemented

    def negative(self, x) -> ndx.Array:
        return NotImplemented

    def not_equal(self, x, y) -> ndx.Array:
        return ndx.logical_not(ndx.equal(x, y))

    def positive(self, x) -> ndx.Array:
        return NotImplemented

    def pow(self, x, y) -> ndx.Array:
        return NotImplemented

    def real(*args):
        return NotImplemented

    def remainder(self, x, y) -> ndx.Array:
        return NotImplemented

    def round(self, x) -> ndx.Array:
        return NotImplemented

    def sign(self, x) -> ndx.Array:
        return NotImplemented

    def sin(self, x) -> ndx.Array:
        return NotImplemented

    def sinh(self, x) -> ndx.Array:
        return NotImplemented

    def square(self, x) -> ndx.Array:
        return ndx.multiply(x, x)

    def sqrt(self, x) -> ndx.Array:
        return NotImplemented

    def subtract(self, x, y) -> ndx.Array:
        return NotImplemented

    def tan(self, x) -> ndx.Array:
        return NotImplemented

    def tanh(self, x) -> ndx.Array:
        return NotImplemented

    def trunc(self, x) -> ndx.Array:
        return NotImplemented

    # linalg.py

    def matmul(self, x, y) -> ndx.Array:
        return NotImplemented

    def matrix_transpose(self, x) -> ndx.Array:
        return ndx.permute_dims(x, list(range(x.ndim - 2)) + [x.ndim - 1, x.ndim - 2])

    # searching.py

    def argmax(self, x, axis=None, keepdims=False) -> ndx.Array:
        return NotImplemented

    def argmin(self, x, axis=None, keepdims=False) -> ndx.Array:
        return NotImplemented

    def nonzero(self, x) -> tuple[ndx.Array, ...]:
        return NotImplemented

    def searchsorted(
        self,
        x1,
        x2,
        *,
        side: Literal["left", "right"] = "left",
        sorter: ndx.Array | None = None,
    ) -> ndx.Array:
        return NotImplemented

    def where(self, condition, x, y):
        if x.dtype != y.dtype:
            target_dtype = ndx.result_type(x, y)
            x = ndx.astype(x, target_dtype)
            y = ndx.astype(y, target_dtype)
        if isinstance(condition.dtype, dtypes.Nullable) and not isinstance(
            x.dtype, (dtypes.Nullable, dtypes.CoreType)
        ):
            raise TypeError("where condition is nullable, but both outputs are not")
        if condition.to_numpy() is not None and condition.dtype == dtypes.bool:
            if (
                condition.to_numpy().size == 1
                and condition.to_numpy().ndim <= x.ndim
                and condition.to_numpy().item()
            ):
                return x.copy()
            elif (
                condition.to_numpy().size == 1
                and condition.to_numpy().ndim <= y.ndim
                and not condition.to_numpy().item()
            ):
                return y.copy()
        if x.dtype == y.dtype and x.to_numpy() is not None and y.to_numpy() is not None:
            if isinstance(x.to_numpy(), np.ma.MaskedArray):
                if np.ma.allequal(x.to_numpy(), y.to_numpy(), fill_value=False):
                    return ndx.asarray(
                        np.broadcast_arrays(x.to_numpy(), y.to_numpy())[0]
                    )
            else:
                cond = np.equal(x.to_numpy(), y.to_numpy())
                if isinstance(x.dtype, ndx.Floating):
                    cond |= np.isnan(x.to_numpy()) & np.isnan(y.to_numpy())
                if cond.all():
                    return ndx.asarray(
                        np.broadcast_arrays(x.to_numpy(), y.to_numpy())[0]
                    )
        condition_values = (
            condition.values if condition.dtype == ndx.nbool else condition
        )

        # generic layout operation, work over both arrays with same data type to apply the condition
        def where_dtype_agnostic(a: ndx.Array, b: ndx.Array) -> ndx.Array:
            if a.dtype != b.dtype:
                raise ValueError("The dtype of both arrays must be the same")
            if a.dtype == dtypes.bool:
                return ndx.logical_xor(
                    ndx.logical_and(condition_values, a),
                    ndx.logical_and(~condition_values, b),
                )
            elif a.dtype in (dtypes.int8, dtypes.int16):
                return from_corearray(
                    opx.where(
                        condition_values._core(),
                        a.astype(dtypes.int32)._core(),
                        b.astype(dtypes.int32)._core(),
                    )
                ).astype(a.dtype)
            elif a.dtype in (dtypes.uint16, dtypes.uint32, dtypes.uint64):
                return from_corearray(
                    opx.where(
                        condition_values._core(),
                        a.astype(dtypes.int64)._core(),
                        b.astype(dtypes.int64)._core(),
                    )
                ).astype(a.dtype)
            elif isinstance(a.dtype, dtypes.CoreType):
                return from_corearray(
                    opx.where(
                        condition_values._core(),
                        a._core(),
                        b._core(),
                    )
                )
            else:
                fields = {}
                for name in a.dtype._fields():
                    fields[name] = where_dtype_agnostic(
                        getattr(a, name), getattr(b, name)
                    )
                return ndx.Array._from_fields(a.dtype, **fields)

        output = where_dtype_agnostic(x, y)
        if condition.dtype == ndx.nbool:
            # propagate null if present
            if not isinstance(output.dtype, dtypes.Nullable):
                output = ndx.additional.make_nullable(output, condition.null)
            else:
                output.null = output.null | condition.null
        return output

    # set.py
    def unique_all(self, x):
        return NotImplemented

    def unique_counts(self, x):
        return NotImplemented

    def unique_inverse(self, x):
        return NotImplemented

    def unique_values(self, x):
        return NotImplemented

    # sorting.py

    def argsort(self, x, *, axis=-1, descending=False, stable=True) -> ndx.Array:
        return NotImplemented

    def sort(self, x, *, axis=-1, descending=False, stable=True) -> ndx.Array:
        return NotImplemented

    # statistical.py

    def cumulative_sum(
        self,
        x,
        *,
        axis: int | None = None,
        dtype: ndx.CoreType | dtypes.StructType | None = None,
        include_initial: bool = False,
    ):
        return NotImplemented

    def max(self, x, *, axis=None, keepdims: bool = False) -> ndx.Array:
        return NotImplemented

    def mean(self, x, *, axis=None, keepdims: bool = False) -> ndx.Array:
        return NotImplemented

    def min(self, x, *, axis=None, keepdims: bool = False) -> ndx.Array:
        return NotImplemented

    def prod(
        self,
        x,
        *,
        axis=None,
        dtype: dtypes.CoreType | dtypes.StructType | None = None,
        keepdims: bool = False,
    ) -> ndx.Array:
        return NotImplemented

    def clip(self, x, *, min=None, max=None) -> ndx.Array:
        return NotImplemented

    def std(
        self,
        x,
        axis=None,
        keepdims: bool = False,
        correction=0.0,
        dtype: dtypes.StructType | dtypes.CoreType | None = None,
    ) -> ndx.Array:
        return NotImplemented

    def sum(
        self,
        x,
        *,
        axis=None,
        dtype: dtypes.CoreType | dtypes.StructType | None = None,
        keepdims: bool = False,
    ) -> ndx.Array:
        return NotImplemented

    def var(
        self,
        x,
        *,
        axis=None,
        keepdims: bool = False,
        correction=0.0,
        dtype: dtypes.CoreType | dtypes.StructType | None = None,
    ) -> ndx.Array:
        return NotImplemented

    # utility.py

    def all(self, x, *, axis=None, keepdims: bool = False) -> ndx.Array:
        return NotImplemented

    def any(self, x, *, axis=None, keepdims: bool = False) -> ndx.Array:
        return NotImplemented

    # indexing.py
    def take(self, x, indices, axis=None) -> ndx.Array:
        if axis is None:
            axis = 0
        if isinstance(indices, ndx.Array):
            indices = indices._core()
        else:
            indices = opx.const(indices, dtype=dtypes.int64)
        return x._transmute(lambda corearray: opx.gather(corearray, indices, axis=axis))

    def broadcast_to(self, x, shape) -> ndx.Array:
        shape = ndx.asarray(shape, dtype=dtypes.int64)._core()
        return x._transmute(lambda corearray: opx.expand(corearray, shape))

    def expand_dims(self, x, axis) -> ndx.Array:
        return x._transmute(
            lambda corearray: opx.unsqueeze(
                corearray, axes=opx.const([axis], dtype=dtypes.int64)
            )
        )

    def flip(self, x, axis) -> ndx.Array:
        if x.ndim == 0:
            return x.copy()

        if axis is None:
            axis = range(x.ndim)
        elif not isinstance(axis, Iterable):
            axis = [axis]

        axis = [x.ndim + ax if ax < 0 else ax for ax in axis]

        index = tuple(
            slice(None, None, None) if i not in axis else (slice(None, None, -1))
            for i in range(0, x.ndim)
        )

        return x[index]

    def permute_dims(self, x, axes) -> ndx.Array:
        return x._transmute(lambda corearray: opx.transpose(corearray, perm=axes))

    def reshape(self, x, shape, *, copy=None) -> ndx.Array:
        if (
            isinstance(shape, (list, tuple))
            and len(shape) == x.ndim == 1
            and shape[0] == -1
        ):
            return x.copy()
        else:
            x = ndx.asarray(x)
            shape = ndx.asarray(shape, dtype=dtypes.int64)._core()
            out = x._transmute(lambda corearray: opx.reshape(corearray, shape))
            if copy is False:
                return x._set(out)
            return out

    def roll(self, x, shift, axis) -> ndx.Array:
        if not isinstance(shift, Sequence):
            shift = [shift]

        old_shape = x.shape

        if axis is None:
            x = ndx.reshape(x, [-1])
            axis = 0

        if not isinstance(axis, Sequence):
            axis = [axis]

        if len(shift) != len(axis):
            raise ValueError("shift and axis must have the same length")

        for sh, ax in zip(shift, axis):
            len_single = opx.gather(
                ndx.asarray(x.shape, dtype=dtypes.int64)._core(), opx.const(ax)
            )
            shift_single = opx.add(opx.const(-sh, dtype=dtypes.int64), len_single)
            # Find the needed element index and then gather from it
            range = opx.cast(
                opx.range(
                    opx.const(0, dtype=len_single.dtype),
                    len_single,
                    opx.const(1, dtype=len_single.dtype),
                ),
                to=dtypes.int64,
            )
            new_indices = opx.mod(opx.add(range, shift_single), len_single)
            x = ndx.take(x, from_corearray(new_indices), axis=ax)

        return ndx.reshape(x, old_shape)

    def squeeze(self, x, axis) -> ndx.Array:
        axis = [axis] if not isinstance(axis, Sequence) else axis
        if len(axis) == 0:
            return x.copy()
        axes = opx.const(axis, dtype=dtypes.int64)
        return x._transmute(lambda corearray: opx.squeeze(corearray, axes=axes))

    # creation.py
    def full(self, shape, fill_value, dtype=None, device=None) -> ndx.Array:
        fill_value = ndx.asarray(fill_value)
        if fill_value.ndim != 0:
            raise ValueError("fill_value must be a scalar")

        dtype = fill_value.dtype if dtype is None else dtype
        shape = ndx.asarray(shape, dtype=dtypes.int64)
        if shape.ndim == 0:
            shape = ndx.reshape(shape, [1])
        return fill_value._transmute(
            lambda corearray: opx.expand(corearray, shape._core())
        ).astype(dtype)

    def full_like(self, x, fill_value, dtype=None, device=None) -> ndx.Array:
        fill_value = ndx.asarray(fill_value, dtype=dtype or x.dtype)
        return ndx.broadcast_to(fill_value, ndx.asarray(x).shape)

    def ones(
        self,
        shape,
        dtype: dtypes.CoreType | dtypes.StructType | None = None,
        device=None,
    ):
        return NotImplemented

    def ones_like(
        self, x, dtype: dtypes.StructType | dtypes.CoreType | None = None, device=None
    ):
        return ndx.ones(x.shape, dtype=dtype or x.dtype, device=device)

    def zeros(
        self,
        shape,
        dtype: dtypes.CoreType | dtypes.StructType | None = None,
        device=None,
    ):
        return NotImplemented

    def zeros_like(
        self, x, dtype: dtypes.CoreType | dtypes.StructType | None = None, device=None
    ):
        return ndx.zeros(x.shape, dtype=dtype or x.dtype, device=device)

    def empty(self, shape, dtype=None, device=None) -> ndx.Array:
        return NotImplemented

    def empty_like(self, x, dtype=None, device=None) -> ndx.Array:
        return NotImplemented

    def arange(self, start, stop=None, step=None, dtype=None, device=None) -> ndx.Array:
        return NotImplemented

    def eye(self, n_rows, n_cols=None, k=0, dtype=None, device=None) -> ndx.Array:
        return NotImplemented

    def tril(self, x, k=0) -> ndx.Array:
        return NotImplemented

    def triu(self, x, k=0) -> ndx.Array:
        return NotImplemented

    def linspace(
        self, start, stop, num, *, dtype=None, device=None, endpoint=True
    ) -> ndx.Array:
        return NotImplemented

    # additional.py
    def fill_null(self, x, value):
        value = ndx.asarray(value)

        if not isinstance(x.dtype, dtypes._NullableCore):
            raise TypeError("fill_null accepts only nullable arrays")

        if value.dtype != x.values.dtype:
            value = value.astype(x.values.dtype)
        return ndx.where(x.null, value, x.values)

    def shape(self, x):
        current = x
        while isinstance(current, ndx.Array):
            current = next(iter(current._fields.values()))
        return from_corearray(opx.shape(current))

    def make_nullable(self, x, null):
        if null.dtype != dtypes.bool or not isinstance(x.dtype, dtypes.CoreType):
            return NotImplemented
        return ndx.Array._from_fields(
            dtypes.into_nullable(x.dtype),
            values=x.copy(),
            null=ndx.reshape(null, x.shape),
        )

    def can_cast(self, from_, to) -> bool:
        return NotImplemented
