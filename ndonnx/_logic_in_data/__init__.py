# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from .array import Array, asarray
from .dtypes import DType
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
    string,
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
    nstring,
    nbool,
)
from .funcs import (
    all,
    any,
    arange,
    astype,
    concat,
    cumulative_sum,
    empty,
    empty_like,
    eye,
    flip,
    full,
    full_like,
    linspace,
    ones,
    ones_like,
    mean,
    reshape,
    sum,
    where,
    zeros,
    zeros_like,
)
from .elementwise import (
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
    expm1,
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
    log1p,
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
from .infos import finfo, iinfo
from .namespace_info import __array_namespace_info__
from .constants import e, inf, nan, pi, newaxis

_default_int = int64
_default_float = float64

utf8 = string
nutf8 = nstring

__all__ = [
    "__array_namespace_info__",
    "Array",
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
    "string",
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
    "nstring",
    "all",
    "any",
    "arange",
    "asarray",
    "astype",
    "concat",
    "cumulative_sum",
    "e",
    "empty",
    "empty_like",
    "eye",
    "finfo",
    "flip",
    "full",
    "full_like",
    "iinfo",
    "inf",
    "linspace",
    "mean",
    "nan",
    "newaxis",
    "ones",
    "ones_like",
    "pi",
    "reshape",
    "sum",
    "where",
    "zeros",
    "zeros_like",
    # Element wise functions
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_invert",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "ceil",
    "clip",
    "conj",
    "copysign",
    "cos",
    "cosh",
    "divide",
    "equal",
    "exp",
    "expm1",
    "floor",
    "floor_divide",
    "greater",
    "greater_equal",
    "hypot",
    "imag",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "multiply",
    "negative",
    "not_equal",
    "positive",
    "pow",
    "real",
    "remainder",
    "round",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "square",
    "sqrt",
    "subtract",
    "tan",
    "tanh",
    "trunc",
]
