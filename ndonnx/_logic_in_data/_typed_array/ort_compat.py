# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
"""Compatibility layer to work around missing kernels in onnxruntime.

Currently targets 1.19.2:
https://github.com/microsoft/onnxruntime/blob/v1.19.2/docs/OperatorKernels.md
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from warnings import warn

import numpy as np
import spox.opset.ai.onnx.v21 as op
from spox import Var


def reduce_prod(
    data: Var,
    axes: Var | None = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> Var:
    dtype = data.unwrap_tensor().dtype
    if dtype in (np.int8, np.int16, np.uint8, np.uint16):
        data = op.cast(data, to=np.int32)
    elif dtype == np.uint32:
        data = op.cast(data, to=np.int64)
    elif dtype == np.uint64:
        warn(
            "'reduce_prod' is not implemented for 'uint64' in onnxruntime. A lossy cast to uint32 is used instead"
        )
        data = op.cast(data, to=np.int64)

    res = op.reduce_prod(
        data, axes=axes, keepdims=keepdims, noop_with_empty_axes=noop_with_empty_axes
    )
    if res.unwrap_tensor().dtype != dtype:
        return op.cast(res, to=dtype)
    return res


def cumsum(
    x: Var,
    axis: Var,
    *,
    exclusive: int = 0,
    reverse: int = 0,
) -> Var:
    dtype = x.unwrap_tensor().dtype
    # Unsigned are supported in 1.19.2
    if dtype in (np.uint8, np.int8, np.int16, np.uint16):
        x = op.cast(x, to=np.int32)
    elif dtype == np.uint32:
        x = op.cast(x, to=np.int64)
    elif dtype == np.uint64:
        warn(
            "'cumsum' is not implemented for 'uint64' in onnxruntime. A lossy cast to int64 is used instead"
        )
        x = op.cast(x, to=np.int64)

    res = op.cumsum(x, axis, exclusive=exclusive, reverse=reverse)
    if res.unwrap_tensor().dtype != dtype:
        return op.cast(res, to=dtype)
    return res


class Warn:
    def __init__(self, ty: type[np.generic]):
        self.ty = ty


_MappingDictType = Mapping[tuple[type[np.generic], ...], Warn | type[np.generic]]


def _wrap_element_wise(
    fun: Callable[[Var], Var], mapping: _MappingDictType, cast_output=True
) -> Callable[[Var], Var]:
    def wrapped(a: Var) -> Var:
        dtype_in = a.unwrap_tensor().dtype

        for unsupported_types, via_dtype in mapping.items():
            if dtype_in in unsupported_types:
                if isinstance(via_dtype, Warn):
                    _warn_lossy(fun.__name__, dtype_in, via_dtype.ty)
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

        for unsupported_types, via_dtype in mapping.items():
            if dtype_in in unsupported_types:
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
add = _wrap_binary(op.add, _common_mapping)
equal = _wrap_binary(op.equal, _common_mapping, cast_output=False)
greater = _wrap_binary(op.greater, _common_mapping, cast_output=False)
greater_or_equal = _wrap_binary(op.greater_or_equal, _common_mapping, cast_output=False)
less = _wrap_binary(op.less, _common_mapping, cast_output=False)
less_or_equal = _wrap_binary(op.less_or_equal, _common_mapping, cast_output=False)
mul = _wrap_binary(op.mul, _common_mapping)
sub = _wrap_binary(op.sub, _common_mapping)

div = _wrap_binary(
    op.div,
    {
        (np.uint8, np.int8, np.int16, np.uint16): np.int32,
    },
)

_mapping_float_only: _MappingDictType = {(np.float64,): Warn(np.float32)}
abs = op.abs
acos = _wrap_element_wise(op.acos, _mapping_float_only)
acosh = _wrap_element_wise(op.acosh, _mapping_float_only)
asin = _wrap_element_wise(op.asin, _mapping_float_only)
asinh = _wrap_element_wise(op.asinh, _mapping_float_only)
atan = _wrap_element_wise(op.atan, _mapping_float_only)
atanh = _wrap_element_wise(op.atanh, _mapping_float_only)
cos = _wrap_element_wise(op.cos, _mapping_float_only)
cosh = _wrap_element_wise(op.cosh, _mapping_float_only)

_mapping_float_double: _MappingDictType = {
    (np.uint8, np.int8, np.int16, np.uint16): np.float32,
    (np.uint32, np.int32): np.float64,
    (np.uint64, np.int64): Warn(np.float64),
}
ceil = _wrap_element_wise(op.ceil, _mapping_float_double)
floor = _wrap_element_wise(op.floor, _mapping_float_double)
# T: tensor(double), tensor(float), tensor(int32), tensor(int64), tensor(int8)
neg = _wrap_element_wise(
    op.neg,
    {
        (np.uint8, np.int16, np.uint16): np.int32,
        (np.uint32,): np.int64,
        (np.uint64,): Warn(np.int64),
    },
)
round = _wrap_element_wise(op.floor, _mapping_float_double)
# T tensor(bfloat16), tensor(double), tensor(float), tensor(float16), tensor(int16), tensor(int32), tensor(int64), tensor(int8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(uint8)
sign = op.sign
sin = _wrap_element_wise(op.sin, _mapping_float_double)
sinh = _wrap_element_wise(op.sinh, _mapping_float_only)
tan = _wrap_element_wise(op.tan, _mapping_float_only)
tanh = _wrap_element_wise(op.tanh, _mapping_float_double)


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
