# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import importlib.metadata
import warnings

from ._array import Array, array, from_spox_var
from ._build import (
    build,
)
from ._data_types import (
    CastError,
    CoreType,
    Floating,
    Integral,
    Nullable,
    NullableFloating,
    NullableIntegral,
    NullableNumerical,
    Numerical,
    promote_nullable,
    from_numpy_dtype,
    bool,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    nbool,
    nfloat32,
    nfloat64,
    nint8,
    nint16,
    nint32,
    nint64,
    nuint8,
    nuint16,
    nuint32,
    nuint64,
    nutf8,
    uint8,
    uint16,
    uint32,
    uint64,
    utf8,
)
from ._funcs import (
    arange,
    asarray,
    empty,
    empty_like,
    eye,
    full,
    full_like,
    linspace,
    ones,
    ones_like,
    tril,
    triu,
    zeros,
    zeros_like,
    astype,
    broadcast_arrays,
    broadcast_to,
    can_cast,
    finfo,
    iinfo,
    result_type,
    abs,
    acos,
    acosh,
    add,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    bitwise_and,
    bitwise_left_shift,
    bitwise_invert,
    bitwise_or,
    bitwise_right_shift,
    bitwise_xor,
    ceil,
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
    multiply,
    negative,
    not_equal,
    positive,
    pow,
    remainder,
    round,
    sign,
    sin,
    sinh,
    square,
    sqrt,
    subtract,
    tan,
    tanh,
    trunc,
    matmul,
    matrix_transpose,
    concat,
    expand_dims,
    flip,
    permute_dims,
    reshape,
    roll,
    squeeze,
    stack,
    argmax,
    argmin,
    nonzero,
    searchsorted,
    where,
    unique_all,
    unique_counts,
    unique_inverse,
    unique_values,
    argsort,
    sort,
    cumulative_sum,
    max,
    mean,
    min,
    prod,
    clip,
    std,
    sum,
    var,
    all,
    any,
    take,
)
from ._constants import (
    e,
    inf,
    nan,
    pi,
)
from ._utility import PromotionError

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError as err:  # pragma: no cover
    warnings.warn(f"Could not determine version of {__name__}\n{err!s}", stacklevel=2)
    __version__ = "unknown"


__all__ = [
    "Array",
    "array",
    "from_spox_var",
    "e",
    "inf",
    "nan",
    "pi",
    "arange",
    "asarray",
    "empty",
    "empty_like",
    "eye",
    "full",
    "full_like",
    "linspace",
    "ones",
    "ones_like",
    "tril",
    "triu",
    "zeros",
    "zeros_like",
    "astype",
    "take",
    "broadcast_arrays",
    "broadcast_to",
    "can_cast",
    "finfo",
    "iinfo",
    "result_type",
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
    "multiply",
    "negative",
    "not_equal",
    "positive",
    "pow",
    "remainder",
    "round",
    "sign",
    "sin",
    "sinh",
    "square",
    "sqrt",
    "subtract",
    "tan",
    "tanh",
    "trunc",
    "matmul",
    "matrix_transpose",
    "concat",
    "expand_dims",
    "flip",
    "permute_dims",
    "reshape",
    "roll",
    "squeeze",
    "stack",
    "argmax",
    "argmin",
    "nonzero",
    "searchsorted",
    "where",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "argsort",
    "sort",
    "cumulative_sum",
    "max",
    "mean",
    "min",
    "prod",
    "clip",
    "std",
    "sum",
    "var",
    "all",
    "any",
    "build",
    "bool",
    "utf8",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "nbool",
    "nutf8",
    "nfloat32",
    "nfloat64",
    "nint8",
    "nint16",
    "nint32",
    "nint64",
    "nuint8",
    "nuint16",
    "nuint32",
    "nuint64",
    "NullableNumerical",
    "Numerical",
    "NullableFloating",
    "Floating",
    "NullableIntegral",
    "Nullable",
    "Integral",
    "CoreType",
    "CastError",
    "PromotionError",
    "promote_nullable",
    "from_numpy_dtype",
]
