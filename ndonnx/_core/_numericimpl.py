# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
import functools
import operator
import warnings
from collections import namedtuple
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Literal

import numpy as np

import ndonnx as ndx
import ndonnx._data_types as dtypes
import ndonnx._opset_extensions as opx
from ndonnx._utility import promote

from ._default import OperationsBlock
from ._utils import (
    binary_op,
    from_corearray,
    split_nulls_and_values,
    unary_op,
    variadic_op,
    via_upcast,
)

if TYPE_CHECKING:
    from ndonnx import Array
    from ndonnx._corearray import _CoreArray


class NumericOperationsImpl(OperationsBlock):
    # elementwise.py

    def abs(self, x):
        return unary_op(x, opx.abs)

    def acos(self, x):
        return unary_op(x, opx.acos, dtypes.float32)

    def acosh(self, x):
        return unary_op(x, opx.acosh, dtypes.float32)

    def add(self, x, y) -> ndx.Array:
        return _via_i64_f64(opx.add, [x, y])

    def asin(self, x):
        return unary_op(x, opx.asin, dtypes.float32)

    def asinh(self, x):
        return unary_op(x, opx.asinh, dtypes.float32)

    def atan(self, x):
        return unary_op(x, opx.atan, dtypes.float32)

    def atan2(self, y, x):
        return self.atan(self.divide(y, x))

    def atanh(self, x):
        return unary_op(x, opx.atanh, dtypes.float32)

    def bitwise_and(self, x, y):
        return binary_op(x, y, opx.bitwise_and)

    # TODO: ONNX standard -> not cyclic
    def bitwise_left_shift(self, x, y):
        return binary_op(
            x, y, lambda a, b: opx.bit_shift(a, b, direction="LEFT"), dtypes.uint64
        )

    def bitwise_invert(self, x):
        return unary_op(x, opx.bitwise_not)

    def bitwise_or(self, x, y):
        return binary_op(x, y, opx.bitwise_or)

    # TODO: ONNX standard -> not cyclic
    def bitwise_right_shift(self, x, y):
        return binary_op(
            x,
            y,
            lambda x, y: opx.bit_shift(x, y, direction="RIGHT"),
            dtypes.uint64,
        )

    def bitwise_xor(self, x, y):
        return binary_op(x, y, opx.bitwise_xor)

    def ceil(self, x):
        if isinstance(x.dtype, (dtypes.Floating, dtypes.NullableFloating)):
            return unary_op(x, opx.ceil, dtypes.float64)
        return ndx.asarray(x, copy=False)

    def cos(self, x):
        return unary_op(x, opx.cos, dtypes.float32)

    def cosh(self, x):
        return unary_op(x, opx.cosh, dtypes.float32)

    def divide(self, x, y):
        x, y = promote(x, y)
        if not isinstance(x.dtype, (dtypes.Numerical, dtypes.NullableNumerical)):
            raise TypeError(f"Unsupported dtype for divide: {x.dtype}")
        bits = (
            ndx.iinfo(x.dtype).bits
            if isinstance(x.dtype, (dtypes.Integral, dtypes.NullableIntegral))
            else ndx.finfo(x.dtype).bits
        )
        via_dtype = (
            dtypes.float64
            if bits > 32 or x.dtype in (dtypes.nuint32, dtypes.uint32)
            else dtypes.float32
        )
        return variadic_op([x, y], opx.div, via_dtype=via_dtype, cast_return=False)

    def equal(self, x, y) -> Array:
        x, y = promote(x, y)
        if isinstance(x.dtype, (dtypes.Integral, dtypes.NullableIntegral)):
            return variadic_op([x, y], opx.equal, dtypes.int64, cast_return=False)
        else:
            return binary_op(x, y, opx.equal)

    def exp(self, x):
        return unary_op(x, opx.exp, dtypes.float32)

    def expm1(self, x):
        return self.subtract(self.exp(x), 1)

    def floor(self, x):
        x = ndx.asarray(x)
        if isinstance(x.dtype, (dtypes.Floating, dtypes.NullableFloating)):
            return unary_op(x, opx.floor, dtypes.float64)
        return x

    def floor_divide(self, x, y):
        x, y = promote(x, y)
        dtype = x.dtype
        out = self.floor(self.divide(x, y))
        if isinstance(
            dtype,
            (
                dtypes.Integral,
                dtypes.NullableIntegral,
                dtypes.Unsigned,
                dtypes.NullableUnsigned,
            ),
        ):
            out = out.astype(dtype)
        return out

    def greater(self, x, y):
        return _via_i64_f64(opx.greater, [x, y], cast_return=False)

    def greater_equal(self, x, y):
        return _via_i64_f64(opx.greater_or_equal, [x, y], cast_return=False)

    def isfinite(self, x):
        return self.logical_not(self.isinf(x))

    def isinf(self, x):
        if isinstance(x.dtype, (dtypes.Floating, dtypes.NullableFloating)):
            return unary_op(x, opx.isinf)
        return ndx.full(x.shape, fill_value=False)

    def isnan(self, x):
        if not isinstance(x.dtype, (dtypes.Floating, dtypes.NullableFloating)):
            return ndx.full_like(x, False, dtype=dtypes.bool)
        else:
            return unary_op(x, opx.isnan)

    def less(self, x, y):
        return _via_i64_f64(opx.less, [x, y], cast_return=False)

    def less_equal(self, x, y):
        return _via_i64_f64(opx.less_or_equal, [x, y], cast_return=False)

    def log(self, x):
        return unary_op(x, opx.log, dtypes.float64)

    def log1p(self, x):
        return self.add(self.log(x), ndx.asarray(1, x.dtype))

    def log2(self, x):
        return self.log(x) / np.log(2)

    def log10(self, x):
        return self.log(x) / np.log(10)

    def logaddexp(self, x, y):
        return self.log(self.exp(x) + self.exp(y))

    def multiply(self, x, y):
        x, y = promote(x, y)
        dtype = x.dtype
        via_dtype: dtypes.CoreType
        if isinstance(dtype, (dtypes.Integral, dtypes.NullableIntegral)) or dtype in (
            dtypes.nbool,
            dtypes.bool,
        ):
            via_dtype = dtypes.int64
        elif isinstance(dtype, (dtypes.Floating, dtypes.NullableFloating)):
            via_dtype = dtypes.float64
        else:
            raise TypeError(f"Unsupported dtype for multiply: {dtype}")
        return binary_op(x, y, opx.mul, via_dtype)

    def negative(self, x):
        if isinstance(
            x.dtype, (dtypes.Unsigned, dtypes.NullableUnsigned)
        ) or x.dtype in (ndx.int16, ndx.nint16):
            return unary_op(x, opx.neg, dtypes.int64)
        return unary_op(x, opx.neg)

    def positive(self, x):
        if not isinstance(x.dtype, (dtypes.Numerical, dtypes.NullableNumerical)):
            raise TypeError(f"Unsupported dtype for positive: {x.dtype}")
        return x.copy()

    def pow(self, x, y):
        x, y = ndx.asarray(x), ndx.asarray(y)
        dtype = ndx.result_type(x, y)
        if isinstance(dtype, (dtypes.Unsigned, dtypes.NullableUnsigned)):
            return binary_op(x, y, opx.pow, dtypes.int64)
        else:
            return binary_op(x, y, opx.pow)

    def remainder(self, x, y):
        return binary_op(x, y, lambda x, y: opx.mod(x, y, fmod=1))

    def round(self, x):
        x = ndx.asarray(x)
        if isinstance(x.dtype, (dtypes.Floating, dtypes.NullableFloating)):
            return unary_op(x, opx.round)
        else:
            return x

    def sign(self, x):
        (x_values,), (x_null,) = split_nulls_and_values(x)
        out_values = ndx.where(x_values > 0, 1, ndx.where(x_values < 0, -1, 0))
        if x_null is None:
            return out_values
        else:
            return ndx.additional.make_nullable(out_values, x_null)

    def sin(self, x):
        return unary_op(x, opx.sin, dtypes.float64)

    def sinh(self, x):
        return unary_op(x, opx.sinh, dtypes.float64)

    def square(self, x):
        return self.multiply(x, x)

    def sqrt(self, x):
        return unary_op(x, opx.sqrt, dtypes.float32)

    def subtract(self, x, y):
        x, y = promote(x, y)
        if isinstance(
            x.dtype, (dtypes.Unsigned, dtypes.NullableUnsigned)
        ) or x.dtype in (dtypes.int16, dtypes.int8, dtypes.nint16, dtypes.nint8):
            via_dtype = dtypes.int64
        else:
            via_dtype = None
        return binary_op(x, y, opx.sub, via_dtype=via_dtype)

    def tan(self, x):
        return unary_op(x, opx.tan, dtypes.float32)

    def tanh(self, x):
        return unary_op(x, opx.tanh, dtypes.float32)

    def trunc(self, x):
        x = ndx.asarray(x)
        if isinstance(x.dtype, (dtypes.Floating, dtypes.NullableFloating)):
            return ndx.where(x < 0, self.ceil(x), self.floor(x))
        return x

    # linalg.py

    def matmul(self, x, y):
        return _via_i64_f64(opx.matmul, [x, y])

    # searching.py

    def argmax(self, x, axis=None, keepdims=False):
        if axis is None:
            reshaped_x = ndx.reshape(x, [-1])._core()
            if keepdims:
                return from_corearray(
                    opx.reshape(
                        opx.arg_max(reshaped_x, axis=0, keepdims=False),
                        opx.const([1 for x in range(x.ndim)], dtype=dtypes.int64),
                    )
                )
            else:
                return from_corearray(
                    opx.reshape(
                        opx.arg_max(reshaped_x, axis=0, keepdims=False),
                        opx.const([], dtype=dtypes.int64),
                    )
                )
        return _via_i64_f64(lambda x: opx.arg_max(x, axis=axis, keepdims=keepdims), [x])

    def argmin(self, x, axis=None, keepdims=False):
        if axis is None:
            reshaped_x = ndx.reshape(x, [-1])._core()
            if keepdims:
                return from_corearray(
                    opx.reshape(
                        opx.arg_min(reshaped_x, axis=0, keepdims=False),
                        opx.const([1 for x in range(x.ndim)], dtype=dtypes.int64),
                    )
                )
            else:
                return from_corearray(
                    opx.reshape(
                        opx.arg_min(reshaped_x, axis=0, keepdims=False),
                        opx.const([], dtype=dtypes.int64),
                    )
                )
        return _via_i64_f64(lambda x: opx.arg_min(x, axis=axis, keepdims=keepdims), [x])

    def nonzero(self, x) -> tuple[Array, ...]:
        if not isinstance(x.dtype, dtypes.CoreType):
            return NotImplemented

        if x.ndim == 0:
            return (ndx.arange(0, x != 0, dtype=dtypes.int64),)

        ret_full_flattened = ndx.reshape(
            from_corearray(opx.ndindex(opx.shape(x._core())))[x != 0],
            [-1],
        )

        return tuple(
            ndx.reshape(
                from_corearray(
                    opx.gather_elements(
                        ret_full_flattened._core(),
                        ndx.arange(
                            i,
                            ret_full_flattened.shape[0],
                            x.ndim,
                            dtype=dtypes.int64,
                        )._core(),
                    )
                ),
                [-1],
            )
            for i in range(x.ndim)
        )

    def searchsorted(
        self,
        x1,
        x2,
        *,
        side: Literal["left", "right"] = "left",
        sorter: ndx.Array | None = None,
    ):
        if x1.ndim != 1:
            raise TypeError("x1 must be a 1-dimensional array")
        if sorter is not None:
            if not isinstance(sorter.dtype, ndx.Integral):
                raise TypeError("sorter must be an array of integers")
            x1 = ndx.take(x1, sorter)

        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")

        nan_mask = ndx.isnan(x2)
        combined = ndx.concat([x1, x2])
        positions = ndx.unique_all(combined).inverse_indices + 1

        unique_x1 = ndx.unique_all(x1)
        counts = unique_x1.counts[unique_x1.inverse_indices]

        indices_x1, indices_x2 = map(
            from_corearray, opx.get_indices(x1._core(), x2._core(), positions._core())
        )

        how_many = ndx.zeros(ndx.asarray(combined.shape) + 1, dtype=dtypes.int64)
        how_many[
            ndx.where(indices_x1 + 1 <= combined.shape[0], indices_x1 + 1, indices_x1)
        ] = counts
        how_many = ndx.cumulative_sum(how_many, include_initial=True)

        ret = ndx.zeros(x2.shape, dtype=dtypes.int64)

        if side == "left":
            ret = how_many[indices_x2]
            ret[nan_mask] = ndx.asarray(x1.shape, dtype=dtypes.int64) - 1
        else:
            ret = how_many[indices_x2 + 1]
            ret[nan_mask] = ndx.asarray(x1.shape, dtype=dtypes.int64)

        return ret

    # set.py
    def unique_all(self, x):
        new_dtype = x.dtype
        if isinstance(x.dtype, dtypes.Integral) or x.dtype in (
            dtypes.bool,
            dtypes.nbool,
        ):
            new_dtype = dtypes.int64
        elif isinstance(x.dtype, dtypes.Floating):
            warnings.warn(
                "Float unique_all is incomplete as nans are not handled correctly in onnxruntime"
            )
            new_dtype = dtypes.float64

        flattened = ndx.reshape(x, [-1])
        transformed = flattened.astype(new_dtype)._core()

        ret_opd = opx.unique(transformed, sorted=True)

        # FIXME: I think we can simply use arange/ones+cumsum or something for the indices
        # maybe: indices = opx.cumsum(ones_like(flattened, dtype=dtypes.i64), axis=ndx.asarray(0))
        indices = opx.squeeze(
            opx.ndindex(opx.shape(flattened._core())),
            opx.const([1], dtype=dtypes.int64),
        )

        ret = namedtuple("ret", ["values", "indices", "inverse_indices", "counts"])

        values = from_corearray(ret_opd[0])
        indices = from_corearray(indices[ret_opd[1]])
        inverse_indices = ndx.reshape(from_corearray(ret_opd[2]), x.shape)
        counts = from_corearray(ret_opd[3])

        return ret(
            values=values,
            indices=indices,
            inverse_indices=inverse_indices,
            counts=counts,
        )

    def unique_counts(self, x):
        ret = namedtuple("ret", ["values", "counts"])
        ret_all = self.unique_all(x)
        return ret(values=ret_all.values, counts=ret_all.counts)

    def unique_inverse(self, x):
        ret = namedtuple("ret", ["values", "inverse_indices"])
        ret_all = self.unique_all(x)
        return ret(values=ret_all.values, inverse_indices=ret_all.inverse_indices)

    def unique_values(self, x):
        return self.unique_all(x).values

    # sorting.py

    def argsort(self, x, *, axis=-1, descending=False, stable=True):
        if axis < 0:
            axis += x.ndim
        _len = opx.shape(x._core(), start=axis, end=axis + 1)
        return _via_i64_f64(
            lambda x: opx.top_k(x, _len, largest=descending, axis=axis)[1], [x]
        )

    def sort(self, x, *, axis=-1, descending=False, stable=True):
        if axis < 0:
            axis += x.ndim
        _len = opx.shape(x._core(), start=axis, end=axis + 1)
        return _via_i64_f64(
            lambda x: opx.top_k(x, _len, largest=descending, axis=axis)[0], [x]
        )

    # statistical.py
    def cumulative_sum(
        self,
        x,
        *,
        axis: int | None = None,
        dtype: dtypes.CoreType | dtypes.StructType | None = None,
        include_initial: bool = False,
    ):
        if not isinstance(x.dtype, ndx.CoreType):
            return NotImplemented
        if axis is None:
            if x.ndim <= 1:
                axis = 0
            else:
                raise ValueError("axis must be specified for multi-dimensional arrays")
        out = from_corearray(
            opx.cumsum(
                x._core(), axis=opx.const(axis), exclusive=int(not include_initial)
            )
        )
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def max(self, x, *, axis=None, keepdims: bool = False):
        if axis is None:
            axes = []
        elif not isinstance(axis, Iterable):
            axes = [axis]
        else:
            axes = axis  # type: ignore

        if isinstance(x.dtype, dtypes.NullableFloating):
            fill_value = dtypes.get_finfo(x.dtype.values).min
            x = ndx.where(x.null, fill_value, x.values)
        elif isinstance(x.dtype, dtypes.NullableIntegral):
            fill_value = dtypes.get_iinfo(x.dtype.values).min
            x = ndx.where(x.null, fill_value, x.values)
        if not isinstance(x.dtype, dtypes.Numerical):
            raise TypeError("min is not supported for non-numeric types")

        return _via_i64_f64(
            lambda x: opx.reduce_max(
                x,
                opx.const(axes, dtype=dtypes.int64),
                keepdims=keepdims,
                noop_with_empty_axes=axis is not None,
            ),
            [x],
        )

    def mean(self, x, *, axis=None, keepdims: bool = False):
        return self.sum(x, axis=axis, keepdims=keepdims) / self.sum(
            ndx.full_like(x, 1), axis=axis, keepdims=keepdims
        )

    def min(self, x, *, axis=None, keepdims: bool = False):
        if axis is None:
            axes = []
        elif not isinstance(axis, Iterable):
            axes = [axis]
        else:
            axes = axis  # type: ignore

        if isinstance(x.dtype, dtypes.NullableFloating):
            fill_value = dtypes.get_finfo(x.dtype.values).max
            x = ndx.where(x.null, fill_value, x.values)
        elif isinstance(x.dtype, dtypes.NullableIntegral):
            fill_value = dtypes.get_iinfo(x.dtype.values).max
            x = ndx.where(x.null, fill_value, x.values)
        if not isinstance(x.dtype, dtypes.Numerical):
            raise TypeError("min is not supported for non-numeric types")

        return _via_i64_f64(
            lambda x: opx.reduce_min(
                x,
                opx.const(axes, dtype=dtypes.int64),
                keepdims=keepdims,
                noop_with_empty_axes=axis is not None,
            ),
            [x],
        )

    def prod(
        self,
        x,
        *,
        axis=None,
        dtype: dtypes.CoreType | dtypes.StructType | None = None,
        keepdims: bool = False,
    ):
        if axis is None:
            axes = []
        elif not isinstance(axis, Iterable):
            axes = [axis]
        else:
            axes = axis  # type: ignore

        if isinstance(x.dtype, dtypes.NullableNumerical):
            fill_value = ndx.asarray(1, dtype=x.dtype.values)
            x = ndx.where(x.null, fill_value, x.values)
        if not isinstance(x.dtype, dtypes.Numerical):
            raise TypeError("prod is not supported for non-numeric types")

        if dtype is not None:
            x = x.astype(dtype)

        out = via_upcast(
            lambda x: opx.reduce_prod(
                x,
                opx.const(axes, dtype=dtypes.int64),
                keepdims=keepdims,
                noop_with_empty_axes=axis is not None,
            ),
            [x],
            int_dtype=dtypes.int64,
            float_dtype=dtypes.float32,
        )

        if isinstance(x.dtype, dtypes.Unsigned):
            out = out.astype(dtypes.uint64)
        elif isinstance(x.dtype, dtypes.NullableUnsigned):
            out = out.astype(dtypes.nuint64)

        return out

    def clip(
        self,
        x,
        *,
        min=None,
        max=None,
    ):
        input = x.copy()
        if min is not None:
            min = ndx.asarray(min)
        if max is not None:
            max = ndx.asarray(max)
        if (
            min is not None
            and max is not None
            and min.ndim == 0
            and max.ndim == 0
            and isinstance(x.dtype, dtypes.Numerical)
        ):
            x, min, max = promote(x, min, max)
            if isinstance(x.dtype, dtypes._NullableCore):
                out_null = x.null
                x_values = x.values._core()
                clipped = from_corearray(opx.clip(x_values, min._core(), max._core()))
                clipped = Array._from_fields(x.dtype, values=clipped, null=out_null)
            else:
                clipped = from_corearray(opx.clip(x._core(), min._core(), max._core()))
            if isinstance(x.dtype, (dtypes.Floating, dtypes.NullableFloating)):
                return ndx.where(ndx.isnan(input), np.nan, clipped)
            else:
                return clipped
        else:
            if max is not None:
                x = ndx.where(x <= max, x, max)
            if min is not None:
                x = ndx.where(x >= min, x, min)
            if isinstance(x.dtype, (dtypes.Floating, dtypes.NullableFloating)):
                return ndx.where(ndx.isnan(input), np.nan, x)
            else:
                return x

    def std(
        self,
        x,
        axis=None,
        keepdims: bool = False,
        correction=0.0,
        dtype: dtypes.StructType | dtypes.CoreType | None = None,
    ):
        dtype = x.dtype if dtype is None else dtype
        if not isinstance(dtype, dtypes.CoreType):
            raise TypeError("std is not supported for non-core types")
        return ndx.sqrt(
            ndx.var(x, axis=axis, keepdims=keepdims, correction=correction, dtype=dtype)
        )

    def sum(
        self,
        x,
        *,
        axis=None,
        dtype: dtypes.StructType | dtypes.CoreType | None = None,
        keepdims: bool = False,
    ):
        if axis is None:
            axes = []
        elif not isinstance(axis, Iterable):
            axes = [axis]
        else:
            axes = axis  # type: ignore

        # Fill any nulls with 0
        if isinstance(x.dtype, dtypes.NullableNumerical):
            x = ndx.where(x.null, 0, x.values)

        if not isinstance(x.dtype, dtypes.Numerical):
            raise TypeError("sum is not supported for non-core types")

        if dtype is not None:
            x = x.astype(dtype)

        out = _via_i64_f64(
            lambda corearray: opx.reduce_sum(
                corearray,
                opx.const(axes, dtype=dtypes.int64),
                keepdims=keepdims,
                noop_with_empty_axes=axis is not None,
            ),
            [x],
        )

        if isinstance(x.dtype, dtypes.Unsigned):
            out = out.astype(dtypes.uint64)
        elif isinstance(x.dtype, dtypes.NullableUnsigned):
            out = out.astype(dtypes.nuint64)

        return out

    def var(
        self,
        x,
        *,
        axis=None,
        keepdims: bool = False,
        correction=0.0,
        dtype: dtypes.CoreType | dtypes.StructType | None = None,
    ):
        dtype = x.dtype if dtype is None else dtype
        if not isinstance(dtype, dtypes.CoreType):
            raise TypeError("var is not supported for non-core types")
        # We keepdims, so it is still broadcastable to x
        mean_ = self.mean(x, axis=axis, keepdims=True).astype(dtype)

        return self.sum(self.square(x - mean_), axis=axis, keepdims=keepdims).astype(
            dtype
        ) / (
            self.sum(ndx.full_like(x, 1), axis=axis, keepdims=keepdims).astype(dtype)
            - correction
        )

    # additional.py
    def fill_null(self, x, value):
        value = ndx.asarray(value)

        if not isinstance(x.dtype, dtypes._NullableCore):
            raise TypeError("fill_null accepts only nullable arrays")

        if value.dtype != x.values.dtype:
            value = value.astype(x.values.dtype)
        return ndx.where(x.null, value, x.values)

    def make_nullable(self, x, null):
        if null.dtype != dtypes.bool:
            raise TypeError("null must be a boolean array")
        if not isinstance(x.dtype, dtypes.CoreType):
            raise TypeError("'make_nullable' does not accept nullable arrays")
        return ndx.Array._from_fields(
            dtypes.into_nullable(x.dtype),
            values=x.copy(),
            null=ndx.reshape(null, x.shape),
        )

    def shape(self, x):
        current = x
        while isinstance(current, ndx.Array):
            current = next(iter(current._fields.values()))
        return from_corearray(opx.shape(current))

    def can_cast(self, from_, to) -> bool:
        if isinstance(from_, dtypes.CoreType) and isinstance(to, ndx.CoreType):
            return np.can_cast(from_.to_numpy_dtype(), to.to_numpy_dtype())
        return NotImplemented

    def all(self, x, *, axis=None, keepdims: bool = False):
        if isinstance(x.dtype, dtypes._NullableCore):
            x = ndx.where(x.null, True, x.values)
        if functools.reduce(operator.mul, x._static_shape, 1) == 0:
            return ndx.asarray(True, dtype=ndx.bool)
        return ndx.min(x.astype(ndx.int8), axis=axis, keepdims=keepdims).astype(
            ndx.bool
        )

    def any(self, x, *, axis=None, keepdims: bool = False):
        if isinstance(x.dtype, dtypes._NullableCore):
            x = ndx.where(x.null, False, x.values)
        if functools.reduce(operator.mul, x._static_shape, 1) == 0:
            return ndx.asarray(False, dtype=ndx.bool)
        return ndx.max(x.astype(ndx.int8), axis=axis, keepdims=keepdims).astype(
            ndx.bool
        )

    def arange(self, start, stop=None, step=None, dtype=None, device=None) -> ndx.Array:
        step = ndx.asarray(step)
        if stop is None:
            stop = start
            start = ndx.asarray(0)
        elif start is None:
            start = ndx.asarray(0)

        start, stop, step = promote(start, stop, step)

        if not builtins.all(
            isinstance(x.dtype, ndx.CoreType) for x in (start, stop, step)
        ):
            raise ndx.UnsupportedOperationError(
                "Cannot perform arange with nullable `start`, `stop` or `step` parameters."
            )
        return ndx.astype(
            from_corearray(opx.range(start._core(), stop._core(), step._core())), dtype
        )

    def tril(self, x, k=0) -> ndx.Array:
        if isinstance(x.dtype, dtypes._NullableCore):
            # NumPy appears to just ignore the mask so we do the same
            x = x.values
        return x._transmute(
            lambda core: opx.trilu(
                core, k=ndx.asarray(k, dtype=dtypes.int64)._core(), upper=0
            )
        )

    def triu(self, x, k=0) -> ndx.Array:
        if isinstance(x.dtype, dtypes._NullableCore):
            # NumPy appears to just ignore the mask so we do the same
            x = x.values
        return x._transmute(
            lambda core: opx.trilu(
                core, k=ndx.asarray(k, dtype=dtypes.int64)._core(), upper=1
            )
        )

    def eye(self, n_rows, n_cols=None, k=0, dtype=None, device=None) -> ndx.Array:
        if n_cols is None:
            n_cols = n_rows
        n_rows = ndx.asarray(n_rows, dtype=dtypes.int64)
        n_cols = ndx.asarray(n_cols, dtype=dtypes.int64)
        n_rows = ndx.reshape(n_rows, [-1])
        n_cols = ndx.reshape(n_cols, [-1])
        shape = ndx.concat([n_rows, n_cols])
        dummy = ndx.zeros(shape, dtype=dtype)
        return from_corearray(opx.eye_like(dummy._core(), k=k))

    def linspace(
        self, start, stop, num, *, dtype=None, device=None, endpoint=True
    ) -> ndx.Array:
        return ndx.asarray(
            np.linspace(start, stop, num=num, endpoint=endpoint), dtype=dtype
        )

    def ones(
        self,
        shape,
        dtype: dtypes.CoreType | dtypes.StructType | None = None,
        device=None,
    ):
        dtype = dtypes.float64 if dtype is None else dtype
        return ndx.full(shape, 1, dtype=dtype)

    def ones_like(
        self, x, dtype: dtypes.StructType | dtypes.CoreType | None = None, device=None
    ):
        return ndx.full_like(x, 1, dtype=dtype)

    def zeros(
        self,
        shape,
        dtype: dtypes.CoreType | dtypes.StructType | None = None,
        device=None,
    ):
        dtype = dtypes.float64 if dtype is None else dtype
        return ndx.full(shape, 0, dtype=dtype)

    def zeros_like(
        self, x, dtype: dtypes.CoreType | dtypes.StructType | None = None, device=None
    ):
        return ndx.full_like(x, 0, dtype=dtype)

    def empty(self, shape, dtype=None, device=None) -> ndx.Array:
        shape = ndx.asarray(shape, dtype=dtypes.int64)
        return ndx.full(shape, 0, dtype=dtype)

    def empty_like(self, x, dtype=None, device=None) -> ndx.Array:
        return ndx.full_like(x, 0, dtype=dtype)


def _via_i64_f64(
    fn: Callable[..., _CoreArray], arrays: list[Array], *, cast_return=True
) -> ndx.Array:
    """Like ``_via_dtype`` but uses ``i64`` for integer or boolean types and ``float64``
    for floating point types.

    Raises TypeError if the provided arrays are neither.
    """
    return via_upcast(
        fn,
        arrays,
        int_dtype=dtypes.int64,
        float_dtype=dtypes.float64,
        use_unsafe_uint_cast=True,  # TODO this can cause overflow, we should set it to false and fix all uses
        cast_return=cast_return,
    )
