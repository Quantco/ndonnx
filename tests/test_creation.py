# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import ndonnx as ndx

from .utils import assert_array_equal, run


@pytest.mark.parametrize(
    "start, stop, step, dtype",
    [
        (0, 10, 2, ndx.int64),
        (-10, 0, 2, ndx.int64),
        (-10, 0, 3, ndx.int64),
        (1, 10, 1, ndx.int64),
        (0.0, None, -1, ndx.int64),
        # Hypothesis test cases
        (
            -9_223_371_349_660_010_402,
            -9_223_370_112_709_427_707,
            45_812_983_809,
            ndx.float64,
        ),
        (-9_223_371_349_660_010_402, -9_223_371_349_660_010_401, 1, ndx.float64),
    ],
)
def test_arange_pyscalar(start, stop, step, dtype: ndx.DType | None):
    def do(npx):
        dtype_: np.dtype | ndx.DType | None = dtype
        if dtype is not None and npx == np:
            dtype_ = dtype.unwrap_numpy()
        return npx.arange(start, stop, step, dtype=dtype_)

    np_res, ndx_res = do(np), do(ndx).unwrap_numpy()

    np.testing.assert_array_equal(np_res, ndx_res, strict=True)


def test_arange_numpy_float64_cast_after_construction_execution():
    candidate = ndx.arange(
        np.float64(0),
        np.float64(2),
        np.float64(0.1),
        dtype=ndx.int64,
    )
    expected = np.asarray([0] * 10 + [1] * 10, dtype=np.int64)

    actual = run(ndx.build({}, {"candidate": candidate}), {})["candidate"]

    assert candidate.dtype == ndx.int64
    assert_array_equal(candidate.unwrap_numpy(), expected)
    assert_array_equal(actual, expected)


def test_arange_numpy_mixed_int8_float32_wide_span_execution():
    candidate = ndx.arange(
        np.int8(-120),
        np.int8(120),
        np.float32(100),
    )
    expected = np.asarray([-120, -20, 80], dtype=np.float32)

    eager = candidate.unwrap_numpy()
    actual = run(ndx.build({}, {"candidate": candidate}), {})["candidate"]

    assert candidate.dtype == ndx.float32
    assert eager.dtype == np.dtype(np.float32)
    assert actual.dtype == np.dtype(np.float32)
    assert len(eager) == len(actual) == len(expected) == 3
    assert_array_equal(eager, expected)
    assert_array_equal(actual, expected)


def test_arange_numpy_mixed_float16_float32_wide_span_execution():
    candidate = ndx.arange(
        np.float16(-60000),
        np.float16(60000),
        np.float32(30000),
    )
    expected = np.asarray([-60000, -30000, 0, 30000], dtype=np.float32)

    eager = candidate.unwrap_numpy()
    actual = run(ndx.build({}, {"candidate": candidate}), {})["candidate"]

    assert candidate.dtype == ndx.float32
    assert eager.dtype == np.dtype(np.float32)
    assert actual.dtype == np.dtype(np.float32)
    assert len(eager) == len(actual) == len(expected) == 4
    assert_array_equal(eager, expected)
    assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "args, expected, expected_dtype",
    [
        (
            (np.float16(-1), np.float16(65504), np.float16(32752)),
            np.asarray([-1, 32752], dtype=np.float16),
            ndx.float16,
        ),
        (
            (
                np.float32(-(2**102)),
                np.float32(np.finfo(np.float32).max),
                np.float32(np.finfo(np.float32).max / 2),
            ),
            np.asarray(
                [-5.070602400912918e30, 1.7014116317805963e38],
                dtype=np.float32,
            ),
            ndx.float32,
        ),
    ],
)
def test_arange_numpy_floating_boundary_rounding_execution(
    args, expected, expected_dtype
):
    candidate = ndx.arange(*args)
    eager = candidate.unwrap_numpy()
    actual = run(ndx.build({}, {"candidate": candidate}), {})["candidate"]

    assert candidate.dtype == expected_dtype
    assert eager.dtype == expected.dtype
    assert actual.dtype == expected.dtype
    assert len(eager) == len(actual) == len(expected) == 2
    assert_array_equal(eager, expected)
    assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "args, expected",
    [
        (
            (np.int16(-30000), np.int16(30000), np.float32(20000)),
            np.asarray([-30000, -10000, 10000], dtype=np.float32),
        ),
        (
            (
                np.int32(-2_000_000_000),
                np.int32(2_000_000_000),
                np.float32(1_000_000_000),
            ),
            np.asarray(
                [-2_000_000_000, -1_000_000_000, 0, 1_000_000_000],
                dtype=np.float64,
            ),
        ),
        (
            (
                np.int32(2_000_000_000),
                np.int32(-2_000_000_000),
                np.float64(-1_000_000_000),
            ),
            np.asarray(
                [2_000_000_000, 1_000_000_000, 0, -1_000_000_000],
                dtype=np.float64,
            ),
        ),
        (
            (np.uint8(250), np.uint8(0), np.float32(-100)),
            np.asarray([250, 150, 50], dtype=np.float32),
        ),
        (
            (
                np.int64(-(2**63) + 1),
                np.int64(2**63 - 1),
                np.float64(2**62),
            ),
            np.asarray([-(2**63), -(2**62), 0, 2**62], dtype=np.float64),
        ),
        (
            (np.uint32(2**32 - 1), np.uint32(0), np.float32(-(2**30))),
            np.asarray(
                [2**32 - 1, 3 * 2**30 - 1, 2**31 - 1, 2**30 - 1],
                dtype=np.float64,
            ),
        ),
        (
            (np.uint64(2**64 - 1), np.uint64(0), np.float32(-(2**62))),
            np.asarray([2**64, 3 * 2**62, 2**63, 2**62], dtype=np.float64),
        ),
    ],
)
def test_arange_numpy_mixed_integer_floating_wide_span(args, expected):
    candidate = ndx.arange(*args)
    eager = candidate.unwrap_numpy()

    assert candidate.dtype == ndx.from_numpy_dtype(expected.dtype)
    assert eager.dtype == expected.dtype
    assert_array_equal(eager, expected)


def test_arange_numpy_float16_empty_execution():
    candidate = ndx.arange(
        np.float16(1),
        np.float16(0),
        np.float16(1e-5),
    )
    expected = np.asarray([], dtype=np.float16)

    eager = candidate.unwrap_numpy()
    actual = run(ndx.build({}, {"candidate": candidate}), {})["candidate"]

    assert candidate.dtype == ndx.float16
    assert eager.dtype == np.dtype(np.float16)
    assert actual.dtype == np.dtype(np.float16)
    assert len(eager) == len(actual) == len(expected) == 0
    assert_array_equal(eager, expected)
    assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "args, dtype, expected",
    [
        (
            (np.float16(0), np.float16(1), np.float16(0.1)),
            None,
            np.asarray(
                [
                    0.0,
                    0.0999755859375,
                    0.199951171875,
                    0.2998046875,
                    0.39990234375,
                    0.5,
                    0.599609375,
                    0.69970703125,
                    0.7998046875,
                    0.89990234375,
                ],
                dtype=np.float16,
            ),
        ),
        (
            (np.float32(0), 1.0, 0.1),
            None,
            np.asarray(
                [
                    0.0,
                    0.10000000149011612,
                    0.20000000298023224,
                    0.30000001192092896,
                    0.4000000059604645,
                    0.5,
                    0.6000000238418579,
                    0.699999988079071,
                    0.800000011920929,
                    0.9000000357627869,
                ],
                dtype=np.float32,
            ),
        ),
        (
            (np.float64(0), np.float64(1), np.float64(0.1)),
            None,
            np.asarray(
                [
                    0.0,
                    0.1,
                    0.2,
                    0.30000000000000004,
                    0.4,
                    0.5,
                    0.6000000000000001,
                    0.7000000000000001,
                    0.8,
                    0.9,
                ],
                dtype=np.float64,
            ),
        ),
        (
            (np.float32(1), np.float32(-0.25), np.float32(-0.25)),
            None,
            np.asarray([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float32),
        ),
        (
            (np.float16(1), np.float16(0), np.float16(0.25)),
            None,
            np.asarray([], dtype=np.float16),
        ),
        (
            (np.float16(0), np.float16(1), np.float16(0.1)),
            ndx.float64,
            np.asarray(
                [
                    0.0,
                    0.0999755859375,
                    0.199951171875,
                    0.2998046875,
                    0.39990234375,
                    0.5,
                    0.599609375,
                    0.69970703125,
                    0.7998046875,
                    0.89990234375,
                ],
                dtype=np.float64,
            ),
        ),
    ],
)
def test_arange_numpy_floating_scalars(args, dtype, expected):
    candidate = ndx.arange(*args, dtype=dtype)

    assert candidate.dtype == ndx.from_numpy_dtype(expected.dtype)
    assert_array_equal(candidate.unwrap_numpy(), expected)


@pytest.mark.parametrize(
    "start, stop, step, expected_dtype",
    [
        (np.int8(0), np.int8(3), np.int8(1), ndx.int8),
        (np.float32(0), np.float32(1), np.float32(0.25), ndx.float32),
        (np.uint8(0), np.uint64(3), np.uint8(1), ndx.uint64),
    ],
)
def test_arange_numpy_scalars(start, stop, step, expected_dtype):
    candidate = ndx.arange(start, stop, step)
    expected = np.arange(start, stop, step, dtype=expected_dtype.unwrap_numpy())

    assert candidate.dtype == expected_dtype
    np.testing.assert_array_equal(candidate.unwrap_numpy(), expected, strict=True)


@pytest.mark.parametrize("explicit_dtype", [False, True])
@pytest.mark.parametrize(
    "args, expected",
    [
        (
            (np.int8(-120), np.int8(120), np.int8(1)),
            np.asarray(range(-120, 120), dtype=np.int8),
        ),
        (
            (np.uint64(2**53), np.uint64(2**53 + 3), np.uint64(1)),
            np.asarray([2**53, 2**53 + 1, 2**53 + 2], dtype=np.uint64),
        ),
        (
            (np.int8(0), np.int8(3), np.int8(1)),
            np.asarray([0, 1, 2], dtype=np.int8),
        ),
        (
            (np.float32(0), np.float32(1), np.float32(0.25)),
            np.asarray([0, 0.25, 0.5, 0.75], dtype=np.float32),
        ),
        (
            (np.float16(0), np.float16(1), np.float16(0.25)),
            np.asarray([0, 0.25, 0.5, 0.75], dtype=np.float16),
        ),
    ],
)
def test_arange_numpy_scalars_match_rank_zero(args, expected, explicit_dtype):
    dtype = ndx.from_numpy_dtype(expected.dtype) if explicit_dtype else None
    candidate = ndx.arange(*args, dtype=dtype)
    rank_zero = ndx.arange(*(ndx.asarray(arg) for arg in args), dtype=dtype)

    assert candidate.dtype == rank_zero.dtype
    np.testing.assert_array_equal(candidate.unwrap_numpy(), expected, strict=True)
    np.testing.assert_array_equal(
        candidate.unwrap_numpy(), rank_zero.unwrap_numpy(), strict=True
    )


@pytest.mark.parametrize(
    "args, dtype, expected",
    [
        ((0, np.int8(3), 1), None, np.asarray([0, 1, 2], dtype=np.int8)),
        (
            (np.float32(0), 1.0, 0.25),
            None,
            np.asarray([0, 0.25, 0.5, 0.75], dtype=np.float32),
        ),
        (
            (np.int8(0), np.int64(3), 1),
            None,
            np.asarray([0, 1, 2], dtype=np.int64),
        ),
        (
            (np.int8(-120), np.int8(120), 1),
            ndx.int64,
            np.asarray(range(-120, 120), dtype=np.int64),
        ),
    ],
)
def test_arange_mixed_numpy_python_scalars(args, dtype, expected):
    candidate = ndx.arange(*args, dtype=dtype)

    np.testing.assert_array_equal(candidate.unwrap_numpy(), expected, strict=True)


@pytest.mark.parametrize(
    "args, dtype, expected",
    [
        (
            (np.float32(0), np.float32(1), np.float32(0.25)),
            ndx.int64,
            np.asarray([0, 0, 0, 0], dtype=np.int64),
        ),
        (
            (np.int64(100_000_001), np.int64(100_000_004), np.int64(1)),
            ndx.float32,
            np.asarray([100_000_001, 100_000_002, 100_000_003], dtype=np.float32),
        ),
    ],
)
def test_arange_numpy_scalars_cast_after_construction(args, dtype, expected):
    candidate = ndx.arange(*args, dtype=dtype)
    rank_zero = ndx.arange(*(ndx.asarray(arg) for arg in args)).astype(dtype)

    np.testing.assert_array_equal(candidate.unwrap_numpy(), expected, strict=True)
    np.testing.assert_array_equal(
        candidate.unwrap_numpy(), rank_zero.unwrap_numpy(), strict=True
    )


@pytest.mark.parametrize("dtype", [None, ndx.int64])
@pytest.mark.parametrize(
    "args, expected_values",
    [
        (
            (np.int32(-2147483647), np.int32(2147483647), np.int32(1073741824)),
            [-2147483647, -1073741823, 1, 1073741825],
        ),
        (
            (np.int64(-(2**63) + 1), np.int64(2**63 - 1), np.int64(2**62)),
            [-(2**63) + 1, -(2**62) + 1, 1, 2**62 + 1],
        ),
        (
            (np.uint64(0), np.uint64(2**64 - 1), np.uint64(2**62)),
            [0, 2**62, 2**63, 3 * 2**62],
        ),
        (
            (np.uint64(0), np.uint64(2**64 - 1), np.uint64(2**63)),
            [0, 2**63],
        ),
        (
            (np.int64(2**63 - 1), np.int64(-(2**63) + 1), np.int64(-(2**62))),
            [2**63 - 1, 2**62 - 1, -1, -(2**62) - 1],
        ),
        ((np.int32(10), np.int32(-10), np.int32(1)), []),
    ],
)
def test_arange_numpy_integer_wide_span(args, expected_values, dtype):
    expected = np.asarray(expected_values, dtype=args[0].dtype)
    if dtype is not None:
        expected = expected.astype(dtype.unwrap_numpy())

    candidate = ndx.arange(*args, dtype=dtype)

    np.testing.assert_array_equal(candidate.unwrap_numpy(), expected, strict=True)


@pytest.mark.parametrize("cls", [ndx.DateTime64DType, ndx.TimeDelta64DType])
@pytest.mark.parametrize("unit", ["s", "ms"])
@pytest.mark.parametrize("scalar_type", [np.int8, np.int64, np.uint64])
@pytest.mark.parametrize("explicit_dtype", [False, True])
def test_arange_temporal_numpy_integer_step(cls, unit, scalar_type, explicit_dtype):
    dtype = cls(unit)
    start = ndx.asarray(0, dtype=dtype)
    stop = ndx.asarray(3, dtype=dtype)

    candidate = ndx.arange(
        start, stop, scalar_type(1), dtype=dtype if explicit_dtype else None
    )

    assert candidate.dtype == dtype
    np.testing.assert_array_equal(
        candidate.unwrap_numpy(),
        np.asarray([0, 1, 2], dtype=dtype.unwrap_numpy()),
        strict=True,
    )


@pytest.mark.parametrize("cls", [ndx.DateTime64DType, ndx.TimeDelta64DType])
@pytest.mark.parametrize("scalar_type", [float, np.float16, np.float32, np.float64])
@pytest.mark.parametrize("explicit_dtype", [False, True])
def test_arange_temporal_floating_scalar_step_rejected(
    cls, scalar_type, explicit_dtype
):
    dtype = cls("s")
    start = ndx.asarray(0, dtype=dtype)
    stop = ndx.asarray(3, dtype=dtype)

    with pytest.raises(ValueError, match="'arange' is not implemented"):
        ndx.arange(start, stop, scalar_type(1), dtype=dtype if explicit_dtype else None)


# A bare-integer step against datetime/timedelta bounds triggers NumPy's
# deprecation of the implicit 'generic' timedelta unit. That bare-integer
# behavior is exactly what we assert works identically in numpy and ndonnx.
@pytest.mark.filterwarnings(
    "ignore:The 'generic' unit for NumPy timedelta:DeprecationWarning"
)
@pytest.mark.parametrize(
    "start, stop, step",
    [
        (np.asarray(0.0), np.asarray(10.0), 1),
        (np.asarray(0.0), 10.0, 1),
        (np.asarray(10.0), None, 1),
        (np.asarray(0, "datetime64[s]"), np.asarray(10, "datetime64[s]"), 1),
        (np.asarray(0, "datetime64[s]"), np.asarray(10_000, "datetime64[ms]"), 1),
        (
            np.asarray(0, "datetime64[s]"),
            np.asarray(10_000, "datetime64[ms]"),
            np.asarray(1_000, "timedelta64[ms]"),
        ),
        (np.asarray(0, "timedelta64[s]"), np.asarray(10, "timedelta64[s]"), 1),
        (np.asarray(0, "timedelta64[s]"), np.asarray(10_000, "timedelta64[ms]"), 1),
        (
            np.asarray(0, "timedelta64[s]"),
            np.asarray(10_000, "timedelta64[ms]"),
            np.asarray(1_000, "timedelta64[ms]"),
        ),
    ],
)
def test_arange_array_arg(start, stop, step):
    def do(npx):
        sss = [
            el if isinstance(el, int | float | None) else npx.asarray(el)
            for el in [start, stop, step]
        ]
        return npx.arange(*sss)

    np_res, ndx_res = do(np), do(ndx).unwrap_numpy()

    np.testing.assert_array_equal(np_res[0], ndx_res[0], strict=True)


@pytest.mark.parametrize(
    "time_dtype_np, time_dtype_ndx",
    [("timedelta64", ndx.TimeDelta64DType), ("datetime64", ndx.DateTime64DType)],
)
@pytest.mark.parametrize("initial_unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("new_unit", ["s", "ms", "us", "ns"])
def test_time_dtype_creation_from_time_dtype(
    time_dtype_np, time_dtype_ndx, initial_unit, new_unit
):
    def do(npx):
        initial_dtype = (
            time_dtype_np + f"[{initial_unit}]"
            if npx == np
            else time_dtype_ndx(initial_unit)
        )
        new_dtype = (
            time_dtype_np + f"[{new_unit}]" if npx == np else time_dtype_ndx(new_unit)
        )
        arr = npx.asarray(np.asarray([1]), dtype=initial_dtype)
        return npx.asarray(arr, dtype=new_dtype)

    np.testing.assert_array_equal(do(ndx).unwrap_numpy(), do(np))
