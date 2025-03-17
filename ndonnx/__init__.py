# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from warnings import warn

from ._dtypes import DType
from ._from_numpy_dtype import from_numpy_dtype
from ._array import Array, asarray, array


from ._deprecated import (
    from_spox_var,
    Nullable,
    Floating,
    Integral,
    Numerical,
    CoreType,
    NullableFloating,
    NullableIntegral,
)
from ._typed_array.onnx import (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    utf8,
    bool_ as bool,
)
from ._typed_array.masked_onnx import (
    nint8,
    nint16,
    nint32,
    nint64,
    nuint8,
    nuint16,
    nuint32,
    nuint64,
    nfloat16,
    nfloat32,
    nfloat64,
    nutf8,
    nbool,
    to_nullable_dtype,
)
from ._typed_array.datetime import DateTime64DType, TimeDelta64DType
from ._funcs import (
    expm1,
    log1p,
    atan2,
    conj,
    imag,
    from_dlpack,
    meshgrid,
    moveaxis,
    repeat,
    tile,
    unstack,
    tensordot,
    vecdot,
    all,
    any,
    arange,
    astype,
    argmax,
    argmin,
    argsort,
    nonzero,
    broadcast_to,
    broadcast_arrays,
    concat,
    can_cast,
    cumulative_sum,
    empty,
    empty_like,
    expand_dims,
    eye,
    flip,
    full,
    full_like,
    isdtype,
    linspace,
    matmul,
    matrix_transpose,
    mean,
    ones,
    ones_like,
    permute_dims,
    reshape,
    result_type,
    prod,
    sort,
    sum,
    searchsorted,
    stack,
    squeeze,
    take,
    roll,
    tril,
    triu,
    unique_all,
    unique_counts,
    unique_inverse,
    unique_values,
    where,
    zeros,
    zeros_like,
    max,
    min,
    std,
    var,
)
from ._elementwise import (
    abs,
    acos,
    acosh,
    add,
    asin,
    asinh,
    atan,
    atanh,
    bitwise_and,
    bitwise_left_shift,
    bitwise_invert,
    bitwise_or,
    bitwise_right_shift,
    bitwise_xor,
    ceil,
    clip,
    copysign,
    cos,
    cosh,
    divide,
    equal,
    exp,
    floor,
    floor_divide,
    greater,
    greater_equal,
    hypot,
    isfinite,
    isinf,
    isnan,
    less,
    less_equal,
    log,
    log2,
    log10,
    logaddexp,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    maximum,
    minimum,
    multiply,
    negative,
    not_equal,
    positive,
    pow,
    real,
    remainder,
    round,
    sign,
    signbit,
    sin,
    sinh,
    square,
    sqrt,
    subtract,
    tan,
    tanh,
    trunc,
)
from ._infos import finfo, iinfo
from ._namespace_info import __array_namespace_info__
from ._constants import e, inf, nan, pi, newaxis
from ._build import build
from . import extensions as extensions

_default_int = int64
_default_float = float64

__all__ = [
    "__array_namespace_info__",
    "DType",
    # ONNX data types
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "utf8",
    "bool",
    # masked ONNX data types
    "nint8",
    "nint16",
    "nint32",
    "nint64",
    "nuint8",
    "nuint16",
    "nuint32",
    "nuint64",
    "nfloat16",
    "nfloat32",
    "nfloat64",
    "nbool",
    "nutf8",
    # Standard functions
    "abs",
    "acos",
    "acosh",
    "add",
    "all",
    "any",
    "arange",
    "argmax",
    "argmin",
    "argsort",
    "asarray",
    "asin",
    "asinh",
    "astype",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "broadcast_arrays",
    "broadcast_to",
    "can_cast",
    "ceil",
    "clip",
    "concat",
    "conj",
    "copysign",
    "cos",
    "cosh",
    "cumulative_sum",
    "divide",
    "e",
    "empty",
    "empty_like",
    "equal",
    "exp",
    "expand_dims",
    "expm1",
    "eye",
    "finfo",
    "flip",
    "floor",
    "floor_divide",
    "from_dlpack",
    "full",
    "full_like",
    "greater",
    "greater_equal",
    "hypot",
    "iinfo",
    "imag",
    "inf",
    "isdtype",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "linspace",
    "log",
    "log10",
    "log1p",
    "log2",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "matmul",
    "matrix_transpose",
    "max",
    "maximum",
    "mean",
    "meshgrid",
    "min",
    "minimum",
    "moveaxis",
    "multiply",
    "nan",
    "negative",
    "newaxis",
    "nonzero",
    "not_equal",
    "ones",
    "ones_like",
    "permute_dims",
    "pi",
    "positive",
    "pow",
    "prod",
    "real",
    "remainder",
    "repeat",
    "reshape",
    "result_type",
    "roll",
    "round",
    "searchsorted",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "sort",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "std",
    "subtract",
    "sum",
    "take",
    "tan",
    "tanh",
    "tensordot",
    "tile",
    "tril",
    "triu",
    "trunc",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "unstack",
    "var",
    "vecdot",
    "where",
    "zeros",
    "zeros_like",
    # Non-standard items
    "build",
    "DateTime64DType",
    "TimeDelta64DType",
    "Array",
    "to_nullable_dtype",
    "extensions",
    "from_numpy_dtype",
]


def __getattr__(name: str):
    def _warn_use_instead(old: str, new: str):
        warn("'{old}' is deprecated in favor of '{new}'", DeprecationWarning)

    superseded = {
        "array": ("ndonnx.Array", array),
        "from_spox_var": ("ndonnx.asarray", from_spox_var),
        "Nullable": ("ndonnx.extensions.is_nullable_dtype", Nullable),
        "NullableFloating": (
            "ndonnx.extensions.is_nullable_float_dtype",
            NullableFloating,
        ),
        "NullableIntegral": (
            "ndonnx.extensions.is_nullable_integer_dtype",
            NullableIntegral,
        ),
        "Floating": ("ndonnx.extensions.is_float_dtype", Floating),
        "Integral": ("ndonnx.extensions.is_integer_dtype", Integral),
        "Numerical": ("ndonnx.extensions.is_numeric_dtype", Numerical),
        "CoreType": ("ndonnx.extensions.is_onnx_dtype", CoreType),
        "NullableCore": ("ndonnx.extensions.is_nullable_dtype", Nullable),
        # We don't raise the following errors any more so we should not export such types
        "UnsupportedOperationError": ("TypeError", TypeError),
        "CastError": ("TypeError", TypeError),
    }

    if tpl := superseded.get(name):
        new_name, old_thing = tpl
        _warn_use_instead(name, new_name)
        return old_thing

    raise AttributeError("module 'ndonnx' has no attribute `{name}`")
