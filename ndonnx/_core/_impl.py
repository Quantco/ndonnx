# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import warnings
from collections import namedtuple
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Literal

import numpy as np

import ndonnx as ndx
import ndonnx._data_types as dtypes
import ndonnx._opset_extensions as opx
from ndonnx._utility import promote
from ndonnx.additional import make_nullable

from ._interface import OperationsBlock

if TYPE_CHECKING:
    from ndonnx._array import Array
    from ndonnx._corearray import _CoreArray


class CoreOperationsImpl(OperationsBlock):
    # elementwise.py

    def abs(self, x):
        return _unary_op(x, opx.abs)

    def acos(self, x):
        return _unary_op(x, opx.acos, dtypes.float32)

    def acosh(self, x):
        return _unary_op(x, opx.acosh, dtypes.float32)

    def add(self, x, y) -> ndx.Array:
        x, y = promote(x, y)
        out_dtype = x.dtype
        if out_dtype in (ndx.nutf8, ndx.utf8):
            return _binary_op(x, y, opx.string_concat)
        return _via_i64_f64(opx.add, [x, y])

    def asin(self, x):
        return _unary_op(x, opx.asin, dtypes.float32)

    def asinh(self, x):
        return _unary_op(x, opx.asinh, dtypes.float32)

    def atan(self, x):
        return _unary_op(x, opx.atan, dtypes.float32)

    def atan2(self, y, x):
        return self.atan(self.divide(y, x))

    def atanh(self, x):
        return _unary_op(x, opx.atanh, dtypes.float32)

    def bitwise_and(self, x, y):
        # FIXME: Bitwise operations like this one should raise a type
        # error when encountering booleans!
        x, y = promote(x, y)
        if x.dtype in (dtypes.bool, dtypes.nbool):
            return self.logical_and(x, y)

        return _binary_op(x, y, opx.bitwise_and)

    # TODO: ONNX standard -> not cyclic
    def bitwise_left_shift(self, x, y):
        return _binary_op(
            x, y, lambda a, b: opx.bit_shift(a, b, direction="LEFT"), dtypes.uint64
        )

    def bitwise_invert(self, x):
        x = ndx.asarray(x)
        if x.dtype in (dtypes.bool, dtypes.nbool):
            return self.logical_not(x)
        return _unary_op(x, opx.bitwise_not)

    def bitwise_or(self, x, y):
        x, y = promote(x, y)
        if x.dtype in (dtypes.bool, dtypes.nbool):
            return self.logical_or(x, y)
        return _binary_op(x, y, opx.bitwise_or)

    # TODO: ONNX standard -> not cyclic
    def bitwise_right_shift(self, x, y):
        x, y = promote(x, y)
        return _binary_op(
            x,
            y,
            lambda x, y: opx.bit_shift(x, y, direction="RIGHT"),
            dtypes.uint64,
        )

    def bitwise_xor(self, x, y):
        if x.dtype in (dtypes.bool, dtypes.nbool):
            return self.logical_xor(x, y)
        return _binary_op(x, y, opx.bitwise_xor)

    def ceil(self, x):
        if isinstance(x.dtype, (dtypes.Floating, dtypes.NullableFloating)):
            return _unary_op(x, opx.ceil, dtypes.float64)
        return ndx.asarray(x, copy=False)

    def cos(self, x):
        return _unary_op(x, opx.cos, dtypes.float32)

    def cosh(self, x):
        return _unary_op(x, opx.cosh, dtypes.float32)

    def divide(self, x, y):
        return _variadic_op([x, y], opx.div, via_dtype=dtypes.float64)

    def equal(self, x, y) -> Array:
        x, y = promote(x, y)
        if isinstance(x.dtype, (dtypes.Integral, dtypes.NullableIntegral)):
            return _variadic_op([x, y], opx.equal, dtypes.int64, cast_return=False)
        else:
            return _binary_op(x, y, opx.equal)

    def exp(self, x):
        return _unary_op(x, opx.exp, dtypes.float32)

    def expm1(self, x):
        return self.subtract(self.exp(x), 1)

    def floor(self, x):
        x = ndx.asarray(x)
        if isinstance(x.dtype, (dtypes.Floating, dtypes.NullableFloating)):
            return _unary_op(x, opx.floor, dtypes.float64)
        return x

    def floor_divide(self, x, y):
        return self.floor(self.divide(x, y))

    def greater(self, x, y):
        return _via_i64_f64(opx.greater, [x, y], cast_return=False)

    def greater_equal(self, x, y):
        return _via_i64_f64(opx.greater_or_equal, [x, y], cast_return=False)

    def isfinite(self, x):
        return self.logical_not(self.isinf(x))

    def isinf(self, x):
        x = ndx.asarray(x)
        if isinstance(x.dtype, (dtypes.Floating, dtypes.NullableFloating)):
            return _unary_op(x, opx.isinf)
        return ndx.full(x.shape, fill_value=False)

    def isnan(self, x):
        if not isinstance(x.dtype, (dtypes.Floating, dtypes.NullableFloating)):
            return ndx.full_like(x, False, dtype=dtypes.bool)
        else:
            return _unary_op(x, opx.isnan)

    def less(self, x, y):
        return _via_i64_f64(opx.less, [x, y], cast_return=False)

    def less_equal(self, x, y):
        x, y = ndx.asarray(x), ndx.asarray(y)
        if ndx.result_type(x, y) in (dtypes.utf8, dtypes.nutf8):
            return self.less(x, y) | self.equal(x, y)
        else:
            return _via_i64_f64(opx.less_or_equal, [x, y], cast_return=False)

    def log(self, x):
        return _unary_op(x, opx.log, dtypes.float64)

    def log1p(self, x):
        return self.add(self.log(x), ndx.asarray(1, x.dtype))

    def log2(self, x):
        return self.log(x) / np.log(2)

    def log10(self, x):
        return self.log(x) / np.log(10)

    def logaddexp(self, x, y):
        return self.log(self.exp(x) + self.exp(y))

    def logical_and(self, x, y):
        x, y = promote(x, y)
        if x.dtype not in (dtypes.bool, dtypes.nbool):
            raise TypeError(f"Unsupported dtype for logical_and: {x.dtype}")

        # If one of the operands is True and the broadcasted shape can be guaranteed to be the other array's shape,
        # we can return this other array directly.
        if (
            x.to_numpy() is not None
            and x.to_numpy().size == 1
            and x.ndim <= y.ndim
            and x.to_numpy().item()
        ):
            return y.copy()
        elif (
            y.to_numpy() is not None
            and y.to_numpy().size == 1
            and y.ndim <= x.ndim
            and y.to_numpy().item()
        ):
            return x.copy()
        return _binary_op(x, y, opx.and_)

    def logical_not(self, x):
        return _unary_op(x, opx.not_)

    def logical_or(self, x, y):
        x, y = ndx.asarray(x), ndx.asarray(y)
        dtype = ndx.result_type(x.dtype, y.dtype)
        if dtype not in (dtypes.bool, dtypes.nbool):
            raise TypeError(f"Unsupported dtype for logical_or: {dtype}")
        # If one of the operands is False and the broadcasted shape can be guaranteed to be the other array's shape,
        # we can return this other array directly.
        if (
            x.to_numpy() is not None
            and x.to_numpy().size == 1
            and x.ndim <= y.ndim
            and not x.to_numpy().item()
        ):
            return y.copy()
        elif (
            y.to_numpy() is not None
            and y.to_numpy().size == 1
            and y.ndim <= x.ndim
            and not y.to_numpy().item()
        ):
            return x.copy()
        return _binary_op(x, y, opx.or_)

    def logical_xor(self, x, y):
        return _binary_op(x, y, opx.xor)

    def multiply(self, x, y):
        x, y = promote(x, y)
        dtype = ndx.result_type(x, y)
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
        return _binary_op(x, y, opx.mul, via_dtype)

    def negative(self, x):
        if isinstance(
            x.dtype, (dtypes.Unsigned, dtypes.NullableUnsigned)
        ) or x.dtype in (ndx.int16, ndx.nint16):
            return _unary_op(x, opx.neg, dtypes.int64)
        return _unary_op(x, opx.neg)

    def positive(self, x):
        if not isinstance(x.dtype, (dtypes.Numerical, dtypes.NullableNumerical)):
            raise TypeError(f"Unsupported dtype for positive: {x.dtype}")
        return x.copy()

    def pow(self, x, y):
        x, y = ndx.asarray(x), ndx.asarray(y)
        dtype = ndx.result_type(x, y)
        if isinstance(dtype, (dtypes.Unsigned, dtypes.NullableUnsigned)):
            return _binary_op(x, y, opx.pow, dtypes.int64)
        else:
            return _binary_op(x, y, opx.pow)

    def remainder(self, x, y):
        return _binary_op(x, y, lambda x, y: opx.mod(x, y, fmod=1))

    def round(self, x):
        x = ndx.asarray(x)
        if isinstance(x.dtype, (dtypes.Floating, dtypes.NullableFloating)):
            return _unary_op(x, opx.round)
        else:
            return x

    def sign(self, x):
        (x_values,), (x_null,) = _split_nulls_and_values(x)
        out_values = self.where(x_values > 0, 1, self.where(x_values < 0, -1, 0))
        if x_null is None:
            return out_values
        else:
            return make_nullable(out_values, x_null)

    def sin(self, x):
        return _unary_op(x, opx.sin, dtypes.float64)

    def sinh(self, x):
        return _unary_op(x, opx.sinh, dtypes.float64)

    def square(self, x):
        return self.multiply(x, x)

    def sqrt(self, x):
        return _unary_op(x, opx.sqrt, dtypes.float32)

    def subtract(self, x, y):
        x, y = promote(x, y)
        if isinstance(
            x.dtype, (dtypes.Unsigned, dtypes.NullableUnsigned)
        ) or x.dtype in (dtypes.int16, dtypes.int8, dtypes.nint16, dtypes.nint8):
            via_dtype = dtypes.int64
        else:
            via_dtype = None
        return _binary_op(x, y, opx.sub, via_dtype=via_dtype)

    def tan(self, x):
        return _unary_op(x, opx.tan, dtypes.float32)

    def tanh(self, x):
        return _unary_op(x, opx.tanh, dtypes.float32)

    def trunc(self, x):
        x = ndx.asarray(x)
        if isinstance(x.dtype, (dtypes.Floating, dtypes.NullableFloating)):
            return self.where(x < 0, self.ceil(x), self.floor(x))
        return x

    # linalg.py

    def matmul(self, x, y):
        return _via_i64_f64(opx.matmul, [x, y])

    # searching.py

    def argmax(self, x, axis=None, keepdims=False):
        if axis is None:
            reshaped_x = ndx.reshape(x, [-1])._core()
            if keepdims:
                return _from_corearray(
                    opx.reshape(
                        opx.arg_max(reshaped_x, axis=0, keepdims=False),
                        opx.const([1 for x in range(x.ndim)], dtype=dtypes.int64),
                    )
                )
            else:
                return _from_corearray(
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
                return _from_corearray(
                    opx.reshape(
                        opx.arg_min(reshaped_x, axis=0, keepdims=False),
                        opx.const([1 for x in range(x.ndim)], dtype=dtypes.int64),
                    )
                )
            else:
                return _from_corearray(
                    opx.reshape(
                        opx.arg_min(reshaped_x, axis=0, keepdims=False),
                        opx.const([], dtype=dtypes.int64),
                    )
                )
        return _via_i64_f64(lambda x: opx.arg_min(x, axis=axis, keepdims=keepdims), [x])

    def nonzero(self, x):
        if x.ndim == 0:
            return [ndx.arange(0, x != 0, dtype=dtypes.int64)]

        ret_full_flattened = ndx.reshape(
            _from_corearray(opx.ndindex(opx.shape(x)))[x != 0],
            [-1],
        )

        return [
            ndx.reshape(
                _from_corearray(
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
        ]

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
            _from_corearray, opx.get_indices(x1._core(), x2._core(), positions._core())
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

    def where(self, condition, x, y):
        condition, x, y = ndx.asarray(condition), ndx.asarray(x), ndx.asarray(y)
        if x.dtype != y.dtype:
            x, y = promote(x, y)
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
                return _from_corearray(
                    opx.where(
                        condition_values._core(),
                        a.astype(dtypes.int32)._core(),
                        b.astype(dtypes.int32)._core(),
                    )
                )
            elif isinstance(a.dtype, dtypes.CoreType):
                return _from_corearray(
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
                output = make_nullable(output, condition.null)
            else:
                output.null = output.null | condition.null
        return output

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

        values = _from_corearray(ret_opd[0])
        indices = _from_corearray(indices[ret_opd[1]])
        inverse_indices = ndx.reshape(_from_corearray(ret_opd[2]), x.shape)
        counts = _from_corearray(ret_opd[3])

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
        out = _from_corearray(
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
            x = self.where(x.null, fill_value, x.values)
        elif isinstance(x.dtype, dtypes.NullableIntegral):
            fill_value = dtypes.get_iinfo(x.dtype.values).min
            x = self.where(x.null, fill_value, x.values)
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
            x = self.where(x.null, fill_value, x.values)
        elif isinstance(x.dtype, dtypes.NullableIntegral):
            fill_value = dtypes.get_iinfo(x.dtype.values).max
            x = self.where(x.null, fill_value, x.values)
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
            fill_value = 1
            x = self.where(x.null, fill_value, x.values)
        if not isinstance(x.dtype, dtypes.Numerical):
            raise TypeError("prod is not supported for non-numeric types")

        if dtype is not None:
            x = x.astype(dtype)

        out = _via_i64_f64(
            lambda x: opx.reduce_prod(
                x,
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
                clipped = _from_corearray(opx.clip(x_values, min._core(), max._core()))
                clipped = Array._from_fields(x.dtype, values=clipped, null=out_null)
            else:
                clipped = _from_corearray(opx.clip(x._core(), min._core(), max._core()))
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
        return self.sqrt(
            self.var(
                x, axis=axis, keepdims=keepdims, correction=correction, dtype=dtype
            )
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
            x = self.where(x.null, 0, x.values)

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
            raise TypeError("make_nullable does not accept nullable arrays")
        return ndx.Array._from_fields(
            dtypes.promote_nullable(x.dtype),
            values=x.copy(),
            null=ndx.reshape(null, x.shape),
        )

    def shape(self, x):
        current = x
        while isinstance(current, ndx.Array):
            current = next(iter(current._fields.values()))
        return _from_corearray(opx.shape(current))


def _binary_op(
    x: ndx.Array,
    y: ndx.Array,
    op: Callable[[_CoreArray, _CoreArray], _CoreArray],
    via_dtype: dtypes.CoreType | None = None,
):
    return _variadic_op([x, y], op, via_dtype)


def _unary_op(
    x: ndx.Array,
    op: Callable[[_CoreArray], _CoreArray],
    via_dtype: dtypes.CoreType | None = None,
) -> ndx.Array:
    return _variadic_op([x], op, via_dtype)


def _variadic_op(
    args: list[Array],
    op: Callable[..., _CoreArray],
    via_dtype: dtypes.CoreType | None = None,
    cast_return: bool = True,
) -> ndx.Array:
    args = promote(*args)
    out_dtype = args[0].dtype
    if not isinstance(out_dtype, (dtypes.CoreType, dtypes._NullableCore)):
        raise TypeError(
            f"Expected ndx.Array with CoreType or NullableCoreType, got {args[0].dtype}"
        )
    data, nulls = _split_nulls_and_values(*args)
    if via_dtype is None:
        values = _from_corearray(op(*(x._core() for x in data)))
    else:
        values = _via_dtype(op, via_dtype, data, cast_return=cast_return)

    if (out_null := functools.reduce(_or_nulls, nulls)) is not None:
        dtype = dtypes.promote_nullable(values.dtype)
        return ndx.Array._from_fields(dtype, values=values, null=out_null)
    else:
        return values


def _split_nulls_and_values(*xs: ndx.Array) -> tuple[list[Array], list[Array | None]]:
    """Helper function that splits a series of ndx.Arrays into their constituent value
    and null mask components.

    Raises if an unexpected typed array is provided.
    """
    data: list[Array] = []
    nulls: list[Array | None] = []
    for x in xs:
        if isinstance(x.dtype, dtypes.Nullable):
            nulls.append(x.null)
            data.append(x.values)
        elif isinstance(x, ndx.Array):
            nulls.append(None)
            data.append(x)
        else:
            raise TypeError(f"Expected ndx.Array got {x}")
    return data, nulls


def _or_nulls(x: ndx.Array | None, y: ndx.Array | None) -> ndx.Array | None:
    if x is None:
        return y
    if y is None:
        return x
    return ndx.logical_or(x, y)


def _via_dtype(
    fn: Callable[..., _CoreArray],
    dtype: dtypes.CoreType | dtypes.StructType,
    arrays: list[Array],
    *,
    cast_return=True,
) -> ndx.Array:
    """Call ``fn`` after casting to ``dtype`` and cast result back to the input dtype.

    The point of this function is to work around quirks / limited type support in the
    onnxruntime.

    ndx.Arrays are promoted to a common type prior to their first use.
    """
    promoted = promote(*arrays)
    out_dtype = promoted[0].dtype

    if isinstance(out_dtype, dtypes._NullableCore) and out_dtype.values == dtype:
        dtype = out_dtype

    values, nulls = _split_nulls_and_values(
        *[ndx.astype(arr, dtype) for arr in promoted]
    )

    out_value = _from_corearray(fn(*[value._core() for value in values]))
    out_null = functools.reduce(_or_nulls, nulls)

    if out_null is not None:
        out_value = ndx.Array._from_fields(
            dtypes.promote_nullable(out_value.dtype), values=out_value, null=out_null
        )
    else:
        out_value = ndx.Array._from_fields(out_value.dtype, data=out_value._core())

    if cast_return:
        return ndx.astype(out_value, dtype=out_dtype)
    else:
        return out_value


def _via_i64_f64(
    fn: Callable[..., _CoreArray], arrays: list[Array], *, cast_return=True
) -> ndx.Array:
    """Like ``_via_dtype`` but uses ``i64`` for integer or boolean types and ``float64``
    for floating point types.

    Raises TypeError if the provided arrays are neither.
    """
    promoted_values = promote(*arrays)

    dtype = promoted_values[0].dtype

    via_dtype: dtypes.CoreType
    if isinstance(dtype, (dtypes.Integral, dtypes.NullableIntegral)) or dtype in (
        dtypes.nbool,
        dtypes.bool,
    ):
        via_dtype = dtypes.int64
    elif isinstance(dtype, (dtypes.Floating, dtypes.NullableFloating)):
        via_dtype = dtypes.float64
    else:
        raise TypeError(f"Expected numerical data type, found {dtype}")

    return _variadic_op(arrays, fn, via_dtype, cast_return)


def _from_corearray(
    corearray: _CoreArray,
) -> ndx.Array:
    return ndx.Array._from_fields(corearray.dtype, data=corearray)
