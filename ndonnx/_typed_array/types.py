# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

from typing import TypeVar

import numpy as np

NpUnsignedInteger = np.uint8 | np.uint16 | np.uint32 | np.uint64
NpSignedInteger = np.int8 | np.int16 | np.int32 | np.int64
NpInteger = NpUnsignedInteger | NpSignedInteger
NpFloating = np.float16 | np.float32 | np.float64
NpScalar = NpInteger | NpFloating | np.bool
NpTime = np.datetime64 | np.timedelta64

Integer = int | NpInteger
Floating = float | NpFloating
Bool = np.bool | bool
Scalar = Integer | Floating | Bool | NpTime | str

PyScalar = int | float | bool | str


ISIN_SCALAR = TypeVar(
    "ISIN_SCALAR",
    int,
    float,
    str,
    np.datetime64,
    np.timedelta64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
)

# There is no datetime/timedelta support for the mapping functions, yet
MAPPING_KEY = TypeVar(
    "MAPPING_KEY",
    int,
    float,
    str,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
)
MAPPING_VALUE = TypeVar(
    "MAPPING_VALUE",
    int,
    float,
    str,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
)
