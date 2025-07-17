# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import ndonnx as ndx


def test_numpy_array_ndx_array_reverse_dunder_called_correctly():
    np_arr = np.asarray([1, 2], dtype=np.int32)
    np_arr_2 = np.asarray([3, 4], dtype=np.int32)

    candidate = np_arr + ndx.asarray(np_arr_2)
    expected = np_arr + np_arr_2

    np.testing.assert_array_equal(candidate.unwrap_numpy(), expected, strict=True)
