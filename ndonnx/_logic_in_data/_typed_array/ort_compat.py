# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
"""Compatibility layer to work around missing kernels in onnxruntime.

Currently targets 1.19.2:
https://github.com/microsoft/onnxruntime/blob/v1.19.2/docs/OperatorKernels.md
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from warnings import warn

import numpy as np
import spox.opset.ai.onnx.v21 as op
from spox import Var


class Warn:
    def __init__(self, ty: type[np.generic]):
        self.ty = ty


_MappingDictType = Mapping[tuple[type[np.generic], ...], Warn | type[np.generic]]


def _detour_type(
    dtype: np.dtype, mapping: _MappingDictType
) -> type[np.generic] | Warn | None:
    for unsupported_types, via_dtype in mapping.items():
        if dtype in unsupported_types:
            return via_dtype
    return None


def _wrap_unary(
    fun: Callable[[Var], Var],
    mapping: _MappingDictType,
    cast_output=True,
    fun_name: str | None = None,
) -> Callable[[Var], Var]:
    def wrapped(a: Var) -> Var:
        dtype_in = a.unwrap_tensor().dtype

        if via_dtype := _detour_type(dtype_in, mapping):
            if isinstance(via_dtype, Warn):
                f_name = fun_name or fun.__name__
                _warn_lossy(f_name, dtype_in, via_dtype.ty)
                via_dtype = via_dtype.ty
            a = op.cast(a, to=via_dtype)
            res = fun(a)
            if cast_output:
                return op.cast(res, to=dtype_in)
            return res
        return fun(a)

    return wrapped


def _wrap_binary(
    fun: Callable[[Var, Var], Var], mapping: _MappingDictType, cast_output=True
) -> Callable[[Var, Var], Var]:
    def wrapped(a: Var, b: Var) -> Var:
        dtype_in = a.unwrap_tensor().dtype
        if via_dtype := _detour_type(dtype_in, mapping):
            if isinstance(via_dtype, Warn):
                _warn_lossy(fun.__name__, dtype_in, via_dtype.ty)
                via_dtype = via_dtype.ty

            a = op.cast(a, to=via_dtype)
            b = op.cast(b, to=via_dtype)
            res = fun(a, b)
            if cast_output:
                return op.cast(res, to=dtype_in)
            return res
        return fun(a, b)

    return wrapped


def _mitigate_segfault_from_zero_dims(
    fun: Callable[[Var, Var], Var],
) -> Callable[[Var, Var], Var]:
    """ORT crashes for mut/mat/mul if one of the terms is a rank-1 with zero
    elements."""

    def wrapped(a: Var, b: Var) -> Var:
        a_shape = a.unwrap_tensor().shape or ()
        b_shape = b.unwrap_tensor().shape or ()
        if len(a_shape) == 1 or len(b_shape):
            a = op.unsqueeze(a, op.const([-1]))
            b = op.unsqueeze(b, op.const([-1]))
            res = fun(a, b)
            return op.squeeze(res, op.const([-1]))
        return fun(a, b)

    return wrapped


def _warn_lossy(fun_name: str, unsupported: np.dtype, via: type[np.generic]):
    warn(
        f"'{fun_name}' is not implemented for '{unsupported}' in onnxruntime. A lossy cast to '{via}' is used instead"
    )


# The following is a commonly used set of input types in numeric binary functions
# T: tensor(double), tensor(float), tensor(int32), tensor(int64)
_common_mapping: _MappingDictType = {
    (np.uint8, np.int8, np.int16, np.uint16): np.int32,
    (np.uint32,): np.int64,
    (np.uint64,): Warn(np.int64),
}
add = _mitigate_segfault_from_zero_dims(_wrap_binary(op.add, _common_mapping))
equal = _wrap_binary(op.equal, _common_mapping, cast_output=False)
greater = _wrap_binary(op.greater, _common_mapping, cast_output=False)
greater_or_equal = _wrap_binary(op.greater_or_equal, _common_mapping, cast_output=False)
less = _wrap_binary(op.less, _common_mapping, cast_output=False)
less_or_equal = _wrap_binary(op.less_or_equal, _common_mapping, cast_output=False)
mul = _mitigate_segfault_from_zero_dims(_wrap_binary(op.mul, _common_mapping))
sub = _mitigate_segfault_from_zero_dims(_wrap_binary(op.sub, _common_mapping))

div = _wrap_binary(
    op.div,
    {
        (np.uint8, np.int8, np.int16, np.uint16): np.int32,
    },
)

_mapping_float_only: _MappingDictType = {(np.float64,): Warn(np.float32)}
abs = op.abs
acos = _wrap_unary(op.acos, _mapping_float_only)
acosh = _wrap_unary(op.acosh, _mapping_float_only)
asin = _wrap_unary(op.asin, _mapping_float_only)
asinh = _wrap_unary(op.asinh, _mapping_float_only)
atan = _wrap_unary(op.atan, _mapping_float_only)
atanh = _wrap_unary(op.atanh, _mapping_float_only)
cos = _wrap_unary(op.cos, _mapping_float_only)
cosh = _wrap_unary(op.cosh, _mapping_float_only)

_mapping_float_double: _MappingDictType = {
    (np.uint8, np.int8, np.int16, np.uint16): np.float32,
    (np.uint32, np.int32): np.float64,
    (np.uint64, np.int64): Warn(np.float64),
}
ceil = _wrap_unary(op.ceil, _mapping_float_double)
floor = _wrap_unary(op.floor, _mapping_float_double)
# T: tensor(double), tensor(float), tensor(int32), tensor(int64), tensor(int8)
neg = _wrap_unary(
    op.neg,
    {
        (np.uint8, np.int16, np.uint16): np.int32,
        (np.uint32,): np.int64,
        (np.uint64,): Warn(np.int64),
    },
)
round = _wrap_unary(op.floor, _mapping_float_double)
# T tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
sign = op.sign
sin = _wrap_unary(op.sin, _mapping_float_double)
sinh = _wrap_unary(op.sinh, _mapping_float_only)
sqrt = _wrap_unary(op.sqrt, _mapping_float_double)
tan = _wrap_unary(op.tan, _mapping_float_only)
tanh = _wrap_unary(op.tanh, _mapping_float_double)


def reduce_op(
    data: Var,
    axes: Var | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
    spox_op: Callable,
    mapping: _MappingDictType,
) -> Var:
    fun = _wrap_unary(
        partial(
            spox_op,
            axes=axes,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        ),
        mapping,
        fun_name=spox_op.__name__,
    )
    return fun(data)


# tensor(float), tensor(int32), tensor(int64)
_mapping_reduce_prod: _MappingDictType = {
    (np.int8, np.int16, np.uint8, np.uint16): np.int32,
    (np.uint64,): Warn(np.float32),
}
reduce_prod = partial(reduce_op, spox_op=op.reduce_prod, mapping=_mapping_reduce_prod)

# tensor(bfloat16), tensor(double), tensor(float), tensor(float16),
# tensor(int32), tensor(int64)
_mapping_reduce_sum: _MappingDictType = {
    (np.int8, np.int16, np.uint8, np.uint16): np.int32,
    (np.uint64,): Warn(np.float64),
}
reduce_sum = partial(reduce_op, spox_op=op.reduce_sum, mapping=_mapping_reduce_sum)

# tensor(double), tensor(float), tensor(float16), tensor(int32),
# tensor(int64)
_mapping_reduce_max: _MappingDictType = {
    (np.int8, np.int16, np.uint8, np.uint16): np.int32,
    (np.uint64,): Warn(np.float64),
}
reduce_max = partial(reduce_op, spox_op=op.reduce_max, mapping=_mapping_reduce_max)

# tensor(double), tensor(float), tensor(float16), tensor(int32),
# tensor(int64), tensor(int8), tensor(uint8)
_mapping_reduce_min: _MappingDictType = {
    (np.int16, np.uint16): np.int32,
    (np.uint32, np.uint64): Warn(np.float64),
}
reduce_min = partial(reduce_op, spox_op=op.reduce_min, mapping=_mapping_reduce_min)

# tensor(double), tensor(float), tensor(float16), tensor(int32)
_mapping_reduce_mean: _MappingDictType = {
    (np.int8, np.int16, np.uint8, np.uint16): np.int32,
    (np.uint64, np.int64): Warn(np.float64),
}
reduce_mean = partial(reduce_op, spox_op=op.reduce_mean, mapping=_mapping_reduce_mean)


def cumsum(
    x: Var,
    axis: Var,
    *,
    exclusive: int = 0,
    reverse: int = 0,
) -> Var:
    # It seems that ORT 1.19.2 already supports float64 in practice,
    # but this is not reflected in the OperatorKernels.md file which reads for 1.19.2:
    # tensor(float), tensor(float16), tensor(int32), tensor(int64),
    # tensor(uint32), tensor(uint64)

    # ORT always returns 64bit integer types. Spox detects the
    # standard violation and aborts the value propagation. Thus, we
    # always cast to 64bit types.
    mapping: _MappingDictType = {
        (np.uint8, np.int8, np.int16, np.uint16): np.int64,
        (np.uint32,): np.int64,
        (np.uint64,): Warn(np.float64),
    }
    fun = _wrap_unary(
        partial(
            op.cum_sum,
            axis=axis,
            exclusive=exclusive,
            reverse=reverse,
        ),
        mapping,
        fun_name=op.cum_sum.__name__,
    )
    return fun(x)


def bit_shift(
    X: Var,
    Y: Var,
    *,
    direction: str,
) -> Var:
    # tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
    mapping: _MappingDictType = {
        (np.int8, np.int16, np.int32, np.int64): Warn(np.uint64),
    }
    dtype_in = X.unwrap_tensor().dtype
    detour_type = _detour_type(dtype_in, mapping)

    if detour_type is None:
        return op.bit_shift(X, Y, direction=direction)
    if isinstance(detour_type, Warn):
        detour_type = detour_type.ty
    res = op.bit_shift(
        op.cast(X, to=detour_type), op.cast(Y, to=detour_type), direction=direction
    )
    return op.cast(res, to=dtype_in)


def pow(a: Var, b: Var, /) -> Var:
    compat_args = []
    did_warn = False

    # 1.19.2 support:
    # T = tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64)
    # T1 = tensor(double), tensor(float), tensor(float16), tensor(int32), tensor(int64)
    mapping: _MappingDictType = {
        (np.uint8, np.int8, np.int16, np.uint16): np.int32,
        (np.uint32,): np.int64,
        (np.uint64,): Warn(np.int64),
    }
    for arg in [a, b]:
        dtype_in = arg.unwrap_tensor().dtype
        for unsupported_types, via_dtype in mapping.items():
            if dtype_in in unsupported_types:
                if isinstance(via_dtype, Warn):
                    if not did_warn:
                        _warn_lossy("pow", dtype_in, via_dtype.ty)
                        did_warn = True
                    via_dtype = via_dtype.ty
                compat_args.append(op.cast(arg, to=via_dtype))
                break
        else:
            compat_args.append(arg)

    res = op.pow(*compat_args)
    a_ty = a.unwrap_tensor().dtype
    if res.unwrap_tensor().dtype != a_ty:
        return op.cast(res, to=a_ty)
    return res


_min_max_mapping: _MappingDictType = {
    (np.int8, np.int16, np.uint8, np.uint16): np.int32,
}


# tensor(double), tensor(float), tensor(float16), tensor(int32),
# tensor(int64), tensor(uint32), tensor(uint64)
def max(data_0: Sequence[Var], /) -> Var:
    xs = list(data_0)
    dtype_in = xs[0].unwrap_tensor().dtype
    if via_dtype := _detour_type(dtype_in, _min_max_mapping):
        if isinstance(via_dtype, Warn):
            _warn_lossy("max", dtype_in, via_dtype.ty)
            via_dtype = via_dtype.ty
        xs = [op.cast(x, to=via_dtype) for x in xs]
        out = op.max(xs)
        return op.cast(out, to=dtype_in)
    return op.max(xs)


def min(data_0: Sequence[Var], /) -> Var:
    xs = list(data_0)
    dtype_in = xs[0].unwrap_tensor().dtype
    if via_dtype := _detour_type(dtype_in, _min_max_mapping):
        if isinstance(via_dtype, Warn):
            _warn_lossy("max", dtype_in, via_dtype.ty)
            via_dtype = via_dtype.ty
        xs = [op.cast(x, to=via_dtype) for x in xs]
        out = op.min(xs)
        return op.cast(out, to=dtype_in)
    return op.min(xs)


def clip(input_: Var, min_: Var | None = None, max_: Var | None = None, /) -> Var:
    # Implement clip as a combination of minimum and maximum since the
    # broadcasting semantics of clip are a bit unclear in the standard/ORT
    res = input_
    if min_ is not None:
        res = max([res, min_])
    if max_ is not None:
        res = min([res, max_])
    return res


def top_k(
    X: Var,
    K: Var,
    *,
    axis: int = -1,
    largest: int = 1,
    sorted: int = 1,
) -> tuple[Var, Var]:
    # tensor(double), tensor(float), tensor(int32), tensor(int64)
    mapping = _common_mapping
    # ORT only implements sorted=1
    sorted = 1
    dtype_in = X.unwrap_tensor().dtype

    kwargs = {"axis": axis, "largest": largest, "sorted": sorted}

    if via_dtype := _detour_type(dtype_in, mapping):
        if isinstance(via_dtype, Warn):
            _warn_lossy("top_k", dtype_in, via_dtype.ty)
            via_dtype = via_dtype.ty
        values, indices = op.top_k(op.cast(X, to=via_dtype), K, **kwargs)
        return op.cast(values, to=dtype_in), indices

    return op.top_k(X, K, **kwargs)
