# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

import operator

import numpy as np
import pytest

import ndonnx as ndx

from .utils import assert_array_equal


@pytest.mark.parametrize(
    "scalar, expected_dtype",
    [
        (np.bool_(True), ndx.bool),
        (np.int8(-3), ndx.int8),
        (np.int64(-3), ndx.int64),
        (np.longlong(-3), ndx.int64),
        (np.uint8(3), ndx.uint8),
        (np.uint64(3), ndx.uint64),
        (np.ulonglong(3), ndx.uint64),
        (np.float16(3.25), ndx.float16),
        (np.float32(3.25), ndx.float32),
        (np.float64(3.25), ndx.float64),
        (np.str_("x"), ndx.utf8),
    ],
)
def test_asarray_numpy_scalar(scalar, expected_dtype):
    candidate = ndx.asarray(scalar)

    assert candidate.dtype == expected_dtype
    assert candidate.shape == ()
    np.testing.assert_array_equal(
        candidate.unwrap_numpy(), np.asarray(scalar), strict=True
    )


def test_asarray_nested_numpy_scalars():
    candidate = ndx.asarray([np.int8(1), np.int64(2)])

    assert candidate.dtype == ndx.int64
    np.testing.assert_array_equal(candidate.unwrap_numpy(), np.asarray([1, 2]))


def test_custom_dtype_receives_original_numpy_scalar():
    received = []

    class RecordingDType(type(ndx.int64)):
        def __ndx_create__(self, val):
            received.append(val)
            return ndx.int64.__ndx_create__(val)

    scalar = np.int64(3)
    ndx.asarray(scalar, dtype=RecordingDType())

    assert received[0] is scalar


def test_custom_dtype_arange_receives_original_numpy_scalars():
    received = []

    class RecordingDType(type(ndx.int64)):
        def __ndx_arange__(self, start, stop, step=1):
            received.extend((start, stop, step))
            return ndx.int64.__ndx_arange__(start, stop, step)

    args = (np.int8(0), np.int8(3), np.int8(1))
    ndx.arange(*args, dtype=RecordingDType())

    assert all(candidate is expected for candidate, expected in zip(received, args))


def test_custom_dtype_inferred_arange_numpy_scalar_dispatch():
    from ndonnx._experimental import onnx

    received = []

    class RecordingArray(onnx.TyArrayInt64):
        def __init__(self, var):
            assert var.unwrap_tensor().dtype == np.dtype(np.int64)
            self._var = var

        @property
        def dtype(self):
            return dtype

    class RecordingDType(type(ndx.int64)):
        def unwrap_numpy(self):
            return np.dtype(np.int64)

        def _build(self, var):
            return RecordingArray(var)

        def __ndx_create__(self, val):
            return self._build(ndx.int64.__ndx_create__(val).disassemble())

        def __ndx_result_type__(self, other):
            raise AssertionError(
                "range dispatch must not require common-type promotion"
            )

        def __ndx_arange__(self, start, stop, step=1):
            received.extend((start, stop, step))
            return ndx.int64.__ndx_arange__(start, stop, step)

    dtype = RecordingDType()
    start = ndx.asarray(0, dtype=dtype)
    stop = ndx.asarray(3, dtype=dtype)
    step = np.int64(1)

    candidate = ndx.arange(start, stop, step)

    assert received[0] is start._tyarray
    assert received[1] is stop._tyarray
    assert received[2] is step
    np.testing.assert_array_equal(candidate.unwrap_numpy(), [0, 1, 2])


def test_custom_dtype_result_type_numpy_scalar_order():
    received = []

    class RecordingDType(type(ndx.int64)):
        def __ndx_result_type__(self, other):
            received.append(other)
            return self

    dtype = RecordingDType()
    weak = 1.0

    assert ndx.result_type(dtype, weak, np.float32(1)) is dtype
    assert received[0] == ndx.float32
    assert received[1] is weak


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.floordiv,
        operator.ge,
        operator.gt,
        operator.le,
        operator.lshift,
        operator.lt,
        operator.mod,
        operator.mul,
        operator.pow,
        operator.rshift,
        operator.sub,
        operator.truediv,
    ],
)
@pytest.mark.parametrize(
    "np_arr, np_gen",
    [
        (np.asarray([2], np.uint32), np.int8(2)),
        (np.asarray([2], np.uint32), np.int16(2)),
        (np.asarray([2], np.uint32), np.int32(2)),
        (np.asarray([2], np.uint32), np.int64(2)),
        (np.asarray([2], np.uint32), np.uint8(2)),
        (np.asarray([2], np.uint32), np.uint16(2)),
        (np.asarray([2], np.uint32), np.uint32(2)),
        (np.asarray([2], np.uint32), np.uint64(2)),
    ],
)
def test_dunders_numpy_generic(op, np_arr, np_gen):
    # The first operand is multiplied by two to better test the
    # correct application of non-commutative functions.

    # Forward
    def do(npx):
        return op(npx.asarray(np_arr) * 2, np_gen)

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np))

    # Backward
    def do(npx):  # type: ignore[no-redef]
        return op(np_gen * 2, npx.asarray(np_arr))

    assert_array_equal(do(ndx).unwrap_numpy(), do(np))


def test_temporal_numpy_scalar_dunders():
    datetime_array = np.asarray([100], dtype="datetime64[s]")
    timedelta_array = np.asarray([10], dtype="timedelta64[s]")

    cases = [
        (operator.add, datetime_array, np.timedelta64(1, "s")),
        (operator.add, np.timedelta64(1, "s"), datetime_array),
        (operator.sub, datetime_array, np.datetime64(0, "s")),
        (operator.add, timedelta_array, np.timedelta64(1, "s")),
    ]
    for op, lhs, rhs in cases:
        candidate = op(
            ndx.asarray(lhs) if isinstance(lhs, np.ndarray) else lhs,
            ndx.asarray(rhs) if isinstance(rhs, np.ndarray) else rhs,
        )
        expected = op(lhs, rhs)

        np.testing.assert_array_equal(candidate.unwrap_numpy(), expected, strict=True)


def test_numpy_array_ndx_array_reverse_dunder_called_correctly():
    np_arr = np.asarray([1, 2], dtype=np.int32)
    np_arr_2 = np.asarray([3, 4], dtype=np.int32)

    candidate = np_arr + ndx.asarray(np_arr_2)
    expected = np_arr + np_arr_2

    assert_array_equal(candidate.unwrap_numpy(), expected)
