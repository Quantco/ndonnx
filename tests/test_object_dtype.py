# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import ndonnx._refactor as ndx
from ndonnx._refactor._typed_array.object_dtype import ObjectDtype


def _determine_variant(elem):
    if isinstance(elem, float) and np.isnan(elem):
        return 0
    elif elem is None:
        return 1
    elif isinstance(elem, str):
        return 2
    else:
        raise ValueError


determine_variant = np.vectorize(
    _determine_variant,
    otypes=[
        np.uint8,
    ],
)


def test_dtype_repr():
    dtype = ObjectDtype()

    assert repr(dtype) == "ObjectDtype()"


def test_dtype_schema():
    dtype = ObjectDtype()

    assert dtype.__ndx_infov1__.meta is None


def test_cast_from_utf8():
    dtype = ObjectDtype()

    initial = ndx.asarray(["a", "b"])
    result = initial.astype(dtype)
    np.testing.assert_equal(
        result.unwrap_numpy(),
        np.asarray(["a", "b"], dtype=object),
    )


@pytest.mark.parametrize(
    "x",
    [
        ["hello", "world"],
        [[np.nan], ["xyz"]],
        [["hello"], ["None"], [None], [np.nan]],
    ],
)
def test_creation(x):
    dtype = ObjectDtype()

    initial = ndx.asarray(x, dtype=dtype)
    expected_variant = determine_variant(np.asarray(x, dtype=object))
    np.testing.assert_equal(
        initial._tyarray.variant.unwrap_numpy(),
        expected_variant,
    )


# TODO: would add more tests to just compare with pandas series behaviour (since that is exactly what we wish to model)
# but we will not do so here in ndonnx
