# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause
# mypy: warn_unused_ignores=True

from typing import assert_type

import numpy as np

import ndonnx as ndx

assert_type(ndx.asarray(np.bool_(True)), ndx.Array)
assert_type(ndx.asarray(np.int8(-3)), ndx.Array)
assert_type(ndx.asarray(np.int16(-3)), ndx.Array)
assert_type(ndx.asarray(np.int32(-3)), ndx.Array)
assert_type(ndx.asarray(np.int64(-3)), ndx.Array)
assert_type(ndx.asarray(np.int_(-3)), ndx.Array)
assert_type(ndx.asarray(np.intp(-3)), ndx.Array)
assert_type(ndx.asarray(np.longlong(-3)), ndx.Array)
assert_type(ndx.asarray(np.uint8(3)), ndx.Array)
assert_type(ndx.asarray(np.uint16(3)), ndx.Array)
assert_type(ndx.asarray(np.uint32(3)), ndx.Array)
assert_type(ndx.asarray(np.uint64(3)), ndx.Array)
assert_type(ndx.asarray(np.uint(3)), ndx.Array)
assert_type(ndx.asarray(np.uintp(3)), ndx.Array)
assert_type(ndx.asarray(np.ulonglong(3)), ndx.Array)
assert_type(ndx.asarray(np.float16(3.25)), ndx.Array)
assert_type(ndx.asarray(np.float32(3.25)), ndx.Array)
assert_type(ndx.asarray(np.float64(3.25)), ndx.Array)
assert_type(ndx.asarray(np.str_("x")), ndx.Array)
assert_type(ndx.asarray([np.int8(1), np.int64(2)]), ndx.Array)

ndx.asarray(np.complex64(1))  # type: ignore[arg-type]
ndx.asarray(np.void(b"x"))  # type: ignore[arg-type]
ndx.asarray(np.datetime64("2024-01-01", "s"))  # type: ignore[arg-type]
ndx.asarray(np.timedelta64(1, "s"))  # type: ignore[arg-type]
