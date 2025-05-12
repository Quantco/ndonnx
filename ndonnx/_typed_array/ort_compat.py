# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
"""Compatibility layer to work around missing kernels in onnxruntime.

Currently targets 1.20.1:
https://github.com/microsoft/onnxruntime/blob/v1.20.1/docs/OperatorKernels.md

Updates to this file may be informed by inspecting the diff for ``OperatorKernels.md`` between two tags (e.g. https://github.com/microsoft/onnxruntime/compare/v1.19.0..v1.20.1/).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from warnings import warn

import numpy as np
import spox.opset.ai.onnx.ml.v4 as ml
import spox.opset.ai.onnx.v21 as op
from spox import Var
from spox.opset.ai.onnx.v21 import abs as abs
from spox.opset.ai.onnx.v21 import and_ as and_
from spox.opset.ai.onnx.v21 import bitwise_and as bitwise_and
from spox.opset.ai.onnx.v21 import bitwise_not as bitwise_not
from spox.opset.ai.onnx.v21 import bitwise_or as bitwise_or
from spox.opset.ai.onnx.v21 import bitwise_xor as bitwise_xor
from spox.opset.ai.onnx.v21 import cast as cast
from spox.opset.ai.onnx.v21 import ceil as ceil
from spox.opset.ai.onnx.v21 import compress as compress
from spox.opset.ai.onnx.v21 import concat as concat
from spox.opset.ai.onnx.v21 import const as const
from spox.opset.ai.onnx.v21 import exp as exp  # Only for floats in both standards
from spox.opset.ai.onnx.v21 import expand as expand
from spox.opset.ai.onnx.v21 import floor as floor
from spox.opset.ai.onnx.v21 import gather as gather
from spox.opset.ai.onnx.v21 import gather_elements as gather_elements
from spox.opset.ai.onnx.v21 import gather_nd as gather_nd
from spox.opset.ai.onnx.v21 import isinf as isinf
from spox.opset.ai.onnx.v21 import isnan as isnan
from spox.opset.ai.onnx.v21 import log as log
from spox.opset.ai.onnx.v21 import mod as mod
from spox.opset.ai.onnx.v21 import not_ as not_
from spox.opset.ai.onnx.v21 import or_ as or_
from spox.opset.ai.onnx.v21 import reciprocal as reciprocal
from spox.opset.ai.onnx.v21 import reshape as reshape
from spox.opset.ai.onnx.v21 import round as round
from spox.opset.ai.onnx.v21 import scatter_nd as scatter_nd
from spox.opset.ai.onnx.v21 import shape as shape
from spox.opset.ai.onnx.v21 import size as size
from spox.opset.ai.onnx.v21 import slice as slice
from spox.opset.ai.onnx.v21 import squeeze as squeeze
from spox.opset.ai.onnx.v21 import string_concat as string_concat
from spox.opset.ai.onnx.v21 import tile as tile
from spox.opset.ai.onnx.v21 import transpose as transpose
from spox.opset.ai.onnx.v21 import unsqueeze as unsqueeze
from spox.opset.ai.onnx.v21 import xor as xor


class Warn:
    """Wrapper to keep track if a cast is lossy so that we may issue a warning."""

    def __init__(self, ty: type[np.generic]):
        self.ty = ty


_MappingDictType = Mapping[tuple[type[np.generic], ...], Warn | type[np.generic]]


def _detour_type(
    dtype: np.dtype, mapping: _MappingDictType
) -> type[np.generic] | Warn | None:
    """Return the "detour" type appropriate for ``dtype`` or None.

    Parameters
    ----------
    dtype
        Data type which is to be used as an input to an operator.
    mapping
        Mapping from data types unsupported by the operator to the
        designated detour type.
    """
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
    """Wrap ``fun`` with logic that conditionally applies "detour dtypes" as defined in
    ``mapping``.

    Parameters
    ----------
    fun
        Function to wrap.
    mapping
        Mapping from data types unsupported by the operator to the
        designated detour type.
    cast_output
        Cast output back into input data type. This is wanted if
        ``fun`` is a function ``T -> T``. It should be ``False`` if
        ``fun`` is ``T -> bool`` for instance.
    fun_name
        Stringy identifier of ``fun`` used in warnings.
    """

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
    fun: Callable[[Var, Var], Var],
    mapping: _MappingDictType,
    cast_output=True,
    fun_name: str | None = None,
) -> Callable[[Var, Var], Var]:
    """Wrap ``fun`` with logic that conditionally applies "detour dtypes" as defined in
    ``mapping``.

    Parameters
    ----------
    fun
        Function to wrap.
    mapping
        Mapping from data types unsupported by the operator to the
        designated detour type.
    cast_output
        Cast output back into input data type. This is wanted if
        ``fun`` is a function ``(T, T) -> T``. It should be ``False`` if
        ``fun`` is ``(T, T) -> bool`` for instance.
    fun_name
        Stringy identifier of ``fun`` used in warnings.
    """

    def wrapped(a: Var, b: Var) -> Var:
        dtype_in = a.unwrap_tensor().dtype
        if via_dtype := _detour_type(dtype_in, mapping):
            if isinstance(via_dtype, Warn):
                fname = fun_name or fun.__name__
                _warn_lossy(fname, dtype_in, via_dtype.ty)
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
            a = op.unsqueeze(a, op.const([-1], dtype=np.int64))
            b = op.unsqueeze(b, op.const([-1], dtype=np.int64))
            res = fun(a, b)
            return op.squeeze(res, op.const([-1], dtype=np.int64))
        return fun(a, b)

    return wrapped


def _warn_lossy(fun_name: str, unsupported: np.dtype, via: type[np.generic]):
    warn(
        f"'{fun_name}' is not implemented for '{unsupported}' in onnxruntime. A lossy cast to '{np.dtype(via)}' is used instead"
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

# div does not suffer of the segfault issues.
div = _wrap_binary(op.div, _common_mapping)

_mapping_float_only: _MappingDictType = {(np.float64,): Warn(np.float32)}
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
# T: tensor(double), tensor(float), tensor(int32), tensor(int64), tensor(int8)
neg = _wrap_unary(
    op.neg,
    {
        (np.uint8, np.int16, np.uint16): np.int32,
        (np.uint32,): np.int64,
        (np.uint64,): Warn(np.int64),
    },
)
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
    (np.uint32,): np.int64,
    (np.uint64,): Warn(np.float64),
}
reduce_prod = partial(reduce_op, spox_op=op.reduce_prod, mapping=_mapping_reduce_prod)

# tensor(double), tensor(float), tensor(int32), tensor(int64)
_mapping_reduce_sum: _MappingDictType = {
    (np.bool_, np.int8, np.int16, np.uint8, np.uint16): np.int32,
    (np.uint32,): np.int64,
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
    # T = tensor(double), tensor(float), tensor(int32), tensor(int64)
    # T2 = tensor(int32), tensor(int64)
    mapping: _MappingDictType = {
        (np.uint8, np.int8, np.int16, np.uint16, np.uint32): np.int32,
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
    # tensor(uint32), tensor(uint64), tensor(uint8)
    mapping: _MappingDictType = {
        (np.int8,): np.uint8,
        (np.int16, np.uint16, np.int32): np.uint32,
        (np.int64,): np.uint64,
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


def arg_min(
    data: Var,
    *,
    axis: int = 0,
    keepdims: int = 1,
    select_last_index: int = 0,
) -> Var:
    fun = _wrap_unary(
        partial(
            op.arg_min,
            axis=axis,
            keepdims=keepdims,
            select_last_index=select_last_index,
        ),
        cast_output=False,
        mapping={
            (
                np.int16,
                np.uint16,
            ): np.int32,
            (np.uint32,): np.int64,
            (np.uint64,): Warn(np.float64),
        },
        fun_name="argmin",
    )
    return fun(data)


def arg_max(
    data: Var,
    *,
    axis: int = 0,
    keepdims: int = 1,
    select_last_index: int = 0,
) -> Var:
    fun = _wrap_unary(
        partial(
            op.arg_max,
            axis=axis,
            keepdims=keepdims,
            select_last_index=select_last_index,
        ),
        cast_output=False,
        mapping={
            (
                np.int16,
                np.uint16,
            ): np.int32,
            (np.uint32,): np.int64,
            (np.uint64,): Warn(np.float64),
        },
        fun_name="argmax",
    )
    return fun(data)


# tensor(bool), tensor(float), tensor(int32), tensor(int64), tensor(uint8)
def non_zero(
    X: Var,
) -> Var:
    if X.unwrap_tensor().dtype.kind == "U":
        X = op.not_(op.equal(X, const("")))
    elif X.unwrap_tensor().dtype not in (
        np.bool_,
        np.float32,
        np.int32,
        np.int64,
        np.uint8,
    ):
        X = op.cast(X, to=np.bool_)
    return op.non_zero(X)


# tensor(double), tensor(float), tensor(int32), tensor(int64), tensor(string), tensor(uint8)
def where(
    condition: Var,
    X: Var,
    Y: Var,
) -> Var:
    fun = _wrap_binary(
        lambda x, y: op.where(condition, x, y),
        mapping={
            (np.bool_,): np.uint8,
            (np.int8, np.int16, np.uint16): np.int32,
            (np.uint32,): np.int64,
            (np.uint64,): Warn(np.float64),
        },
        fun_name="where",
    )
    return fun(X, Y)


def unique(
    X: Var,
    *,
    axis: int | None = None,
    sorted: int = 1,
) -> tuple[Var, Var, Var, Var]:
    # tensor(double), tensor(float), tensor(int64), tensor(int8), tensor(string)
    dtype = X.unwrap_tensor().dtype
    via_dtype = _detour_type(
        dtype,
        {
            (np.bool_,): np.int8,
            (np.int16, np.int32, np.uint8, np.uint16, np.uint32): np.int64,
            (np.uint64,): Warn(np.float64),
        },
    )
    x = X
    if via_dtype is not None:
        if isinstance(via_dtype, Warn):
            via_dtype = via_dtype.ty
            _warn_lossy("unique", dtype, via_dtype)
        x = op.cast(x, to=via_dtype)

    values, indices, inverse_indices, counts = op.unique(x, axis=axis, sorted=sorted)

    if via_dtype is not None:
        values = op.cast(values, to=dtype)
    return (values, indices, inverse_indices, counts)


def label_encoder(
    X: Var,
    *,
    default_tensor: np.ndarray,
    keys_tensor: np.ndarray,
    values_tensor: np.ndarray,
) -> Var:
    # T1 = tensor(double), tensor(float), tensor(int64),
    # tensor(string)
    # T2 = tensor(double), tensor(float), tensor(int16),
    # tensor(int64), tensor(string)
    dtype_in = X.unwrap_tensor().dtype
    t1 = _detour_type(
        dtype_in,
        {
            (
                np.bool_,
                np.int8,
                np.int16,
                np.int32,
                np.uint8,
                np.uint16,
                np.uint32,
            ): np.int64,
            (np.float32,): np.float64,
            (
                np.uint64,
            ): np.int64,  # We also cast the key/values, so we don't lose precision.
        },
    )
    t1 = t1.ty if isinstance(t1, Warn) else t1

    t2 = _detour_type(
        values_tensor.dtype,
        {
            (np.bool_, np.int8, np.int32, np.uint8, np.uint16, np.uint32): np.int64,
            (np.float32,): np.float64,
            (
                np.uint64,
            ): np.int64,  # We also cast the key/values, so we don't lose precision.
        },
    )
    t2 = t2.ty if isinstance(t2, Warn) else t2

    res = ml.label_encoder(
        op.cast(X, to=t1) if t1 else X,
        keys_tensor=keys_tensor.astype(t1) if t1 else keys_tensor,
        values_tensor=values_tensor.astype(t2) if t2 else values_tensor,
        default_tensor=default_tensor.astype(t2) if t2 else default_tensor,
    )

    if t2:
        return op.cast(res, to=values_tensor.dtype)
    return res


def trilu(
    input: Var,
    k: Var | None = None,
    *,
    upper: int = 1,
) -> Var:
    # tensor(bool), tensor(double), tensor(float), tensor(int64)
    fun = _wrap_unary(
        lambda x: op.trilu(x, k, upper=upper),
        mapping={
            (
                np.int8,
                np.int16,
                np.int32,
                np.uint8,
                np.uint16,
                np.uint32,
            ): np.int64,
            (np.uint64,): np.int64,  # No warning since casting should be lossless
        },
        fun_name="trilu",
    )

    return fun(input)


def range(
    start: Var,
    limit: Var,
    delta: Var,
) -> Var:
    # tensor(double), tensor(float), tensor(int16), tensor(int32),
    # tensor(int64)
    in_dtype = start.unwrap_tensor().dtype
    via_dtype = _detour_type(
        in_dtype,
        mapping={
            (np.int8, np.uint8): np.int16,
            (np.uint16,): np.int32,
            (np.uint32,): np.int64,
            (np.uint64,): Warn(np.int64),
        },
    )
    if via_dtype is not None:
        if isinstance(via_dtype, Warn):
            via_dtype = via_dtype.ty
            _warn_lossy("range", in_dtype, via_dtype)
        start, limit, delta = (
            op.cast(el, to=via_dtype) for el in (start, limit, delta)
        )

    res = op.range(start, limit, delta)
    if via_dtype is None:
        return res
    return op.cast(res, to=in_dtype)


def matmul(
    A: Var,
    B: Var,
) -> Var:
    # This "compat" function actually extends the ONNX standard to also allow for smaller integer data types.

    # ORT support: tensor(double), tensor(float), tensor(int32), tensor(int64), tensor(uint32), tensor(uint64)

    a_dtype = A.unwrap_tensor().dtype
    b_dtype = B.unwrap_tensor().dtype
    if a_dtype == b_dtype and a_dtype in [np.int8, np.uint8]:
        # Special case for 8 bit data types

        res = op.matmul_integer(A, B)
        return op.cast(res, to=a_dtype)

    return _wrap_binary(
        op.matmul,
        mapping={
            (np.int16,): np.int32,
            (np.uint16,): np.uint32,
        },
        fun_name="matmul",
    )(A, B)


def if_(
    cond: Var,
    *,
    else_branch: Callable[[], Iterable[Var]],
    then_branch: Callable[[], Iterable[Var]],
) -> Sequence[Var]:
    """If operator with explicit value propagation."""
    # TODO: Upstream to Spox
    from spox import Tensor, build
    from spox._value_prop import PropValue

    # execute here to have access to the results of each branch
    then_results = then_branch()
    else_results = else_branch()

    results = op.if_(
        cond,
        then_branch=lambda: then_results,
        else_branch=lambda: else_results,
    )

    if not (
        cond._value is not None
        and all(el._value is not None for el in then_results)
        and all(el._value is not None for el in else_results)
    ):
        return results
    try:
        import onnxruntime as ort
    except ImportError:
        return results

    folded_res = op.if_(
        const(cond._get_value()),  # type: ignore
        then_branch=lambda: [const(el._get_value()) for el in then_results],  # type: ignore
        else_branch=lambda: [const(el._get_value()) for el in else_results],  # type: ignore
    )

    model = build({}, {f"res{i}": res for i, res in enumerate(folded_res)})
    prop_values = ort.InferenceSession(model.SerializeToString()).run(None, {})

    for res, prop_val in zip(results, prop_values):
        res._value = PropValue(Tensor(dtype=prop_val.dtype), value=prop_val)

    return results


def einsum(
    Inputs: Sequence[Var],
    *,
    equation: str,
) -> Var:
    mapping: _MappingDictType = {
        (np.uint8, np.int8, np.int16, np.uint16): np.int32,
        (np.uint32,): np.int64,
        (np.uint64,): Warn(np.int64),
    }

    # Only support binary input since that is all we use
    x1, x2 = Inputs
    return _wrap_binary(
        lambda a, b: op.einsum([a, b], equation=equation),
        mapping=mapping,
        fun_name="einsum",
    )(x1, x2)
