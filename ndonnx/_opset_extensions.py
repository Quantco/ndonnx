# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
import typing
from collections.abc import Mapping
from typing import TypeVar

import numpy as np
import spox.opset.ai.onnx.ml.v4 as ml
import spox.opset.ai.onnx.v20 as op
from spox import Var
from spox._internal_op import unsafe_reshape

import ndonnx._data_types as dtypes

from ._corearray import _CoreArray
from ._index import ScalarIndexType
from ._propagation import eager_propagate
from ._utility import get_dtype, get_rank, get_shape

if typing.TYPE_CHECKING:
    from ndonnx._data_types.coretype import CoreType

ellipsis = TypeVar("ellipsis")


KeyType = TypeVar("KeyType", int, str, float)
ValueType = TypeVar("ValueType", int, str, float)


@eager_propagate
def add(x: _CoreArray, y: _CoreArray) -> _CoreArray:
    return _CoreArray(
        op.add(x.var, y.var),
    )


@eager_propagate
def string_concat(x: _CoreArray, y: _CoreArray) -> _CoreArray:
    return _CoreArray(
        op.string_concat(x.var, y.var),
    )


@eager_propagate
def expand(input: _CoreArray, shape: _CoreArray) -> _CoreArray:
    return _CoreArray(
        op.expand(input.var, shape.var),
    )


@eager_propagate
def asin(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.asin(x.var))


@eager_propagate
def range(start: _CoreArray, limit: _CoreArray, delta: _CoreArray) -> _CoreArray:
    return _CoreArray(op.range(start.var, limit.var, delta.var))


@eager_propagate
def asinh(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.asinh(x.var))


@eager_propagate
def atan(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.atan(x.var))


@eager_propagate
def atanh(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.atanh(x.var))


@eager_propagate
def or_(x: _CoreArray, y: _CoreArray) -> _CoreArray:
    return _CoreArray(op.or_(x.var, y.var))


@eager_propagate
def mul(x: _CoreArray, y: _CoreArray) -> _CoreArray:
    return _CoreArray(op.mul(x.var, y.var))


@eager_propagate
def cast(x: _CoreArray, to: CoreType) -> _CoreArray:
    return _CoreArray(
        op.cast(x.var, to=to.to_numpy_dtype()),
    )


@eager_propagate
def shape(x: _CoreArray, *, end: int | None = None, start: int = 0) -> _CoreArray:
    return _CoreArray(op.shape(x.var, end=end, start=start))


@eager_propagate
def top_k(
    x: _CoreArray, k: _CoreArray, *, axis: int = -1, largest: int = 1, sorted: int = 1
) -> tuple[_CoreArray, _CoreArray]:
    values, indices = op.top_k(x.var, k.var, axis=axis, largest=largest, sorted=sorted)
    return _CoreArray(values), _CoreArray(indices)


@eager_propagate
def where(cond: _CoreArray, x: _CoreArray, y: _CoreArray) -> _CoreArray:
    return _CoreArray(op.where(cond.var, x.var, y.var))


@eager_propagate
def not_(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.not_(x.var))


@eager_propagate
def eye_like(
    input: _CoreArray, *, dtype: CoreType | None = None, k: int = 0
) -> _CoreArray:
    return _CoreArray(
        op.eye_like(
            input.var, dtype=dtype if dtype is None else dtype.to_numpy_dtype(), k=k
        )
    )


@eager_propagate
def gather_elements(
    data: _CoreArray, indices: _CoreArray, *, axis: int = 0
) -> _CoreArray:
    return _CoreArray(op.gather_elements(data.var, indices.var, axis=axis))


@eager_propagate
def equal(x: _CoreArray, y: _CoreArray) -> _CoreArray:
    return _CoreArray(op.equal(x.var, y.var))


@eager_propagate
def reshape(x: _CoreArray, shape: _CoreArray) -> _CoreArray:
    return _CoreArray(op.reshape(x.var, shape.var))


@eager_propagate
def mod(x: _CoreArray, y: _CoreArray, fmod: int = 0) -> _CoreArray:
    return _CoreArray(op.mod(x.var, y.var, fmod=fmod))


@eager_propagate
def pow(x: _CoreArray, y: _CoreArray) -> _CoreArray:
    return _CoreArray(op.pow(x.var, y.var))


@eager_propagate
def div(x: _CoreArray, y: _CoreArray) -> _CoreArray:
    return _CoreArray(op.div(x.var, y.var))


@eager_propagate
def sub(x: _CoreArray, y: _CoreArray) -> _CoreArray:
    return _CoreArray(op.sub(x.var, y.var))


@eager_propagate
def ceil(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.ceil(x.var))


@eager_propagate
def transpose(x: _CoreArray, perm: list[int]) -> _CoreArray:
    return _CoreArray(op.transpose(x.var, perm=perm))


@eager_propagate
def const(value, dtype: CoreType | None = None) -> _CoreArray:
    np_dtype = dtype.to_numpy_dtype() if dtype is not None else None
    return _CoreArray(
        np.array(value, dtype=np_dtype),
    )


@eager_propagate
def squeeze(data: _CoreArray, axes: _CoreArray | None = None) -> _CoreArray:
    if axes is not None and axes.dtype != dtypes.int64:
        raise ValueError(f"Expected axes to be of type int64, got {axes.dtype}")
    return _CoreArray(op.squeeze(data.var, axes=axes.var if axes is not None else None))


@eager_propagate
def abs(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.abs(x.var))


@eager_propagate
def sin(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.sin(x.var))


@eager_propagate
def cos(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.cos(x.var))


@eager_propagate
def log(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.log(x.var))


@eager_propagate
def exp(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.exp(x.var))


@eager_propagate
def tan(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.tan(x.var))


@eager_propagate
def acos(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.acos(x.var))


@eager_propagate
def neg(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.neg(x.var))


@eager_propagate
def sqrt(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.sqrt(x.var))


@eager_propagate
def floor(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.floor(x.var))


@eager_propagate
def tanh(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.tanh(x.var))


@eager_propagate
def sinh(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.sinh(x.var))


@eager_propagate
def cosh(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.cosh(x.var))


@eager_propagate
def round(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.round(x.var))


@eager_propagate
def less(x: _CoreArray, y: _CoreArray) -> _CoreArray:
    return _CoreArray(op.less(x.var, y.var))


@eager_propagate
def less_or_equal(x: _CoreArray, y: _CoreArray) -> _CoreArray:
    return _CoreArray(op.less_or_equal(x.var, y.var))


@eager_propagate
def greater(x: _CoreArray, y: _CoreArray) -> _CoreArray:
    return _CoreArray(op.greater(x.var, y.var))


@eager_propagate
def greater_or_equal(x: _CoreArray, y: _CoreArray) -> _CoreArray:
    return _CoreArray(op.greater_or_equal(x.var, y.var))


@eager_propagate
def isnan(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.isnan(x.var))


@eager_propagate
def isinf(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.isinf(x.var))


@eager_propagate
def reduce_sum(
    data: _CoreArray, axes: _CoreArray, keepdims: int = 1, noop_with_empty_axes: int = 0
) -> _CoreArray:
    return _CoreArray(
        op.reduce_sum(
            data.var,
            axes.var,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )
    )


@eager_propagate
def reduce_min(
    data: _CoreArray, axes: _CoreArray, keepdims: int = 1, noop_with_empty_axes: int = 0
) -> _CoreArray:
    return _CoreArray(
        op.reduce_min(
            data.var,
            axes.var,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )
    )


@eager_propagate
def reduce_max(
    data: _CoreArray, axes: _CoreArray, keepdims: int = 1, noop_with_empty_axes: int = 0
) -> _CoreArray:
    return _CoreArray(
        op.reduce_max(
            data.var,
            axes.var,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )
    )


@eager_propagate
def reduce_mean(
    data: _CoreArray, axes: _CoreArray, keepdims: int = 1, noop_with_empty_axes: int = 0
) -> _CoreArray:
    return _CoreArray(
        op.reduce_mean(
            data.var,
            axes.var,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )
    )


@eager_propagate
def reduce_prod(
    data: _CoreArray, axes: _CoreArray, keepdims: int = 1, noop_with_empty_axes: int = 0
) -> _CoreArray:
    return _CoreArray(
        op.reduce_prod(
            data.var,
            axes.var,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )
    )


@eager_propagate
def concat(inputs: list[_CoreArray], axis: int) -> _CoreArray:
    return _CoreArray(op.concat([x.var for x in inputs], axis=axis))


@eager_propagate
def unsqueeze(data: _CoreArray, axes: _CoreArray) -> _CoreArray:
    if axes.dtype != dtypes.int64:
        raise TypeError(f"axes must be int64, got {axes.dtype}")
    return _CoreArray(op.unsqueeze(data.var, axes.var))


@eager_propagate
def trilu(x: _CoreArray, *, k: _CoreArray | None = None, upper: int = 0) -> _CoreArray:
    return _CoreArray(op.trilu(x.var, k.var if k is not None else None, upper=upper))


@eager_propagate
def unique(
    x: _CoreArray, axis: int | None = None, sorted: int = 1
) -> tuple[_CoreArray, _CoreArray, _CoreArray, _CoreArray]:
    y, indices, inverse_indices, counts = op.unique(x.var, axis=axis, sorted=sorted)
    return (
        _CoreArray(y),
        _CoreArray(indices),
        _CoreArray(inverse_indices),
        _CoreArray(counts),
    )


@eager_propagate
def and_(x: _CoreArray, y: _CoreArray) -> _CoreArray:
    return _CoreArray(op.and_(x.var, y.var))


@eager_propagate
def gather(data: _CoreArray, indices: _CoreArray, axis: int = 0) -> _CoreArray:
    return _CoreArray(op.gather(data.var, indices.var, axis=axis))


@eager_propagate
def cumsum(
    x: _CoreArray, axis: _CoreArray, exclusive: int = 0, reverse: int = 0
) -> _CoreArray:
    return _CoreArray(
        op.cumsum(x.var, axis=axis.var, exclusive=exclusive, reverse=reverse)
    )


@eager_propagate
def split(
    input: _CoreArray,
    split: _CoreArray | None = None,
    *,
    num_outputs: int,
    axis: int = 0,
) -> tuple[_CoreArray, ...]:
    return tuple(
        map(
            _CoreArray,
            op.split(
                input.var,
                split.var if split is not None else None,
                num_outputs=num_outputs,
                axis=axis,
            ),
        )
    )


@eager_propagate
def clip(input: _CoreArray, min: _CoreArray, max: _CoreArray) -> _CoreArray:
    return _CoreArray(op.clip(input.var, min.var, max.var))


@eager_propagate
def matmul(a: _CoreArray, b: _CoreArray) -> _CoreArray:
    return _CoreArray(op.matmul(a.var, b.var))


@eager_propagate
def arg_max(
    data: _CoreArray, *, axis: int = 0, keepdims: int = 1, select_last_index: int = 0
) -> _CoreArray:
    return _CoreArray(
        op.arg_max(
            data.var, axis=axis, keepdims=keepdims, select_last_index=select_last_index
        )
    )


@eager_propagate
def arg_min(
    data: _CoreArray, *, axis: int = 0, keepdims: int = 1, select_last_index: int = 0
) -> _CoreArray:
    return _CoreArray(
        op.arg_min(
            data.var, axis=axis, keepdims=keepdims, select_last_index=select_last_index
        )
    )


@eager_propagate
def bitwise_and(x: _CoreArray, y: _CoreArray) -> _CoreArray:
    return _CoreArray(op.bitwise_and(x.var, y.var))


@eager_propagate
def bitwise_or(x: _CoreArray, y: _CoreArray) -> _CoreArray:
    return _CoreArray(op.bitwise_or(x.var, y.var))


@eager_propagate
def bitwise_xor(x: _CoreArray, y: _CoreArray) -> _CoreArray:
    return _CoreArray(op.bitwise_xor(x.var, y.var))


@eager_propagate
def bitwise_not(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.bitwise_not(x.var))


@eager_propagate
def bit_shift(x: _CoreArray, y: _CoreArray, *, direction: str) -> _CoreArray:
    return _CoreArray(op.bit_shift(x.var, y.var, direction=direction))


@eager_propagate
def xor(x: _CoreArray, y: _CoreArray) -> _CoreArray:
    return _CoreArray(op.xor(x.var, y.var))


@eager_propagate
def acosh(x: _CoreArray) -> _CoreArray:
    return _CoreArray(op.acosh(x.var))


@eager_propagate
def getitem_null(corearray: _CoreArray, index: _CoreArray) -> _CoreArray:
    # Compress doesn't compress in case of scalars
    # A solution that works is to transform
    # var -> [var]
    # index -> [index]

    var = corearray.var
    if get_rank(index) == 0:

        def extend_shape(x: Var) -> Var:
            return op.concat([op.const([1], dtype=np.int64), op.shape(x)], axis=0)

        var = op.reshape(var, extend_shape(var), allowzero=True)
        index_var = op.reshape(index.var, extend_shape(index.var), allowzero=True)
        return _CoreArray(op.compress(var, index_var, axis=0))

    else:
        if get_rank(index) == 1:
            reshaped_var, reshaped_index = var, index.var
        else:
            ret_shape = op.concat(
                [op.const([-1], dtype=np.int64), op.shape(var, start=get_rank(index))],
                axis=0,
            )
            reshaped_var = op.reshape(var, ret_shape, allowzero=True)
            reshaped_index = op.reshape(index.var, op.const([-1]), allowzero=True)

        return _CoreArray(
            op.compress(
                unsafe_reshape(
                    reshaped_var,
                    tuple(
                        None
                        for _ in builtins.range(get_rank(var) - get_rank(index) + 1)
                    ),
                ),
                reshaped_index,
                axis=0,
            )
        )


@eager_propagate
def getitem(
    corearray: _CoreArray,
    index: tuple[ScalarIndexType, ...] | _CoreArray,
) -> _CoreArray:
    if isinstance(index, _CoreArray):
        if get_dtype(index) == np.bool_:
            if get_rank(corearray) < get_rank(index):
                raise IndexError("Indexing with boolean array cannot happen")
            return getitem_null(corearray, index)
        else:
            return _CoreArray(op.gather(corearray.var, index.var, axis=0))
    elif len(index) == 0:
        return corearray.copy()
    elif all(isinstance(i, bool) for i in index):
        index = typing.cast(tuple[bool, ...], index)
        return getitem(
            corearray,
            _CoreArray(op.const(index) if len(index) > 1 else op.const(index[0])),
        )
    # index is a tuple of scalars
    index_some = [x for x in index if x is not None]

    axis_slices = [
        (x.start, x.stop, x.step, ind)
        for ind, x in enumerate(index_some)
        if isinstance(x, slice) and x.step is not None
    ]
    var = corearray.var
    if axis_slices:
        ndim = get_rank(var)
        var = op.slice(
            var,
            op.const(np.array([x[0] for x in axis_slices], np.int64)),
            op.const(np.array([x[1] for x in axis_slices], np.int64)),
            op.const(np.array([x[3] for x in axis_slices], np.int64)),
            op.const(np.array([x[2] for x in axis_slices], np.int64)),
        )

        # TODO: onnx should be fixing this
        var = unsafe_reshape(var, tuple(None for i in builtins.range(ndim)))

    axis_indices = [(ind, x) for ind, x in enumerate(index_some) if isinstance(x, int)]
    for axis, axis_index in reversed(axis_indices):
        var = op.gather(var, op.const(axis_index), axis=axis)

    index_filtered = [x for x in index if isinstance(x, (type(None), slice))]
    axis_new_axes = [ind for ind, x in enumerate(index_filtered) if x is None]
    if len(axis_new_axes) != 0:
        var = op.unsqueeze(var, axes=op.const(axis_new_axes, dtype=np.int64))

    return _CoreArray(var)


@eager_propagate
def setitem(
    corearray: _CoreArray, index: tuple | _CoreArray, updates: _CoreArray
) -> _CoreArray:
    if get_rank(corearray) == 0 and isinstance(index, tuple):
        return updates
    elif get_rank(corearray) == 0 and isinstance(index, Var):
        return _CoreArray(
            op.if_(
                index,
                then_branch=lambda: [updates.var],
                else_branch=lambda: [corearray.var],
            )[0]
        )

    var = corearray.var
    indices = getitem(ndindex(_CoreArray(op.shape(var))), index)

    # broadcast updates as appropriate
    index_path_shape = op.slice(
        op.shape(indices.var),
        op.const([0], dtype=np.int64),
        op.const([-1], dtype=np.int64),
    )
    return _CoreArray(
        op.scatter_nd(var, indices.var, op.expand(updates.var, index_path_shape))
    )


@eager_propagate
def ndindex(shape: _CoreArray, to_reverse=None, axes_permutation=None) -> _CoreArray:
    """
    Returns a tensor of shape `shape`, where each element is its index in the tensor
    Parameters:
        shape: the shape of the tensor
        to_reverse: a list of axes to reverse
        axes_permutation: a permutation of the axes
    Returns:
        A tensor of shape `shape`, where each element is its index in the tensor
    """
    if to_reverse is None:
        to_reverse = []
    (rank,) = get_shape(shape.var)

    if not isinstance(rank, int):
        raise ValueError("rank must be a constant integer")

    if rank == 0:
        return const([], dtype=dtypes.int64)

    if axes_permutation is None:
        axes_indices = list(builtins.range(rank))
    else:
        axes_indices = [axes_permutation.index(i) for i in builtins.range(rank)]

    shape_var = shape.var
    dtype = shape_var.unwrap_tensor().dtype
    ranges = [
        (
            op.range(
                op.const(0, dtype=dtype),
                op.gather(shape_var, op.const(i)),
                op.const(1, dtype=dtype),
            )
            if i not in to_reverse
            else op.range(
                op.sub(op.gather(shape_var, op.const(i)), op.const(1, dtype=dtype)),
                op.const(-1, dtype=dtype),
                op.const(-1, dtype=dtype),
            )
        )
        for i in builtins.range(rank)
    ]

    fit_ranges = [
        op.unsqueeze(
            r,
            op.const(
                [j for j in builtins.range(rank) if axes_indices[i] != j],
                dtype=np.int64,
            ),
        )
        for i, r in enumerate(ranges)
    ]

    if axes_permutation is not None:
        shape_var = op.gather(shape_var, op.const(axes_permutation))

    expanded_ranges = [op.expand(r, shape_var) for r in fit_ranges]

    ret = op.concat(
        [op.unsqueeze(r, op.const([-1], dtype=np.int64)) for r in expanded_ranges],
        axis=-1,
    )

    return _CoreArray(ret)


@eager_propagate
def reshape_like(x: _CoreArray, y: _CoreArray) -> _CoreArray:
    x_var, y_var = x.var, y.var
    return _CoreArray(
        unsafe_reshape(
            op.reshape(x_var, op.shape(y_var)),
            [x.to_simple() for x in y_var.unwrap_tensor()._shape],  # type: ignore
        )
    )


@eager_propagate
def static_map(
    input: _CoreArray, mapping: Mapping[KeyType, ValueType], default: ValueType | None
) -> _CoreArray:
    keys = np.array(tuple(mapping.keys()))
    if keys.dtype == np.int32:
        keys = keys.astype(np.int64)
    values = np.array(tuple(mapping.values()))
    value_dtype = values.dtype
    if default is None:
        if value_dtype == object:
            default_tensor = np.array(["MISSING"])
        else:
            default_tensor = np.array([0], dtype=value_dtype)
    else:
        default_tensor = np.array([default], dtype=value_dtype)

    if keys.dtype == np.float64 and isinstance(input.dtype, dtypes.Integral):
        input = cast(input, dtypes.from_numpy_dtype(keys.dtype))
    elif keys.dtype == np.int64 and isinstance(input.dtype, dtypes.Floating):
        keys = keys.astype(input.dtype.to_numpy_dtype())
    return _CoreArray(
        ml.label_encoder(
            input.var,
            keys_tensor=keys,
            values_tensor=values,
            default_tensor=default_tensor,
        )
    )


@eager_propagate
def get_indices(
    x: _CoreArray, y: _CoreArray, positions: _CoreArray
) -> tuple[_CoreArray, _CoreArray]:
    """Given a position array representing the positions of x concatenated to those of
    y, this function extracts the original positions back out as two tensors."""
    split_pos = shape(x)
    split_pos_var = split_pos.var
    positions_var = positions.var
    indices_x = op.reshape(
        op.slice(positions_var, op.const([0], dtype=np.int64), split_pos_var),
        op.const([-1], dtype=np.int64),
    )
    indices_y = op.reshape(
        op.slice(positions_var, split_pos_var, op.const([np.iinfo(np.int64).max])),
        op.const([-1], dtype=np.int64),
    )
    return _CoreArray(indices_x), _CoreArray(indices_y)
