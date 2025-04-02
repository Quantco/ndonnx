# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import ndonnx as ndx


def test_iter_for_loop():
    n = 5
    a = ndx.argument(shape=(n,), dtype=ndx.int64)

    for i, el in enumerate(a):
        assert isinstance(el, ndx.Array)
        assert i < n, "Iterated past the number of elements"


@pytest.mark.parametrize(
    "arr",
    [
        ndx.asarray([1]),
        ndx.asarray([[1], [2]]),
        ndx.argument(shape=(2,), dtype=ndx.int64),
        ndx.argument(shape=(2, 3), dtype=ndx.int64),
        ndx.argument(shape=(2, "N"), dtype=ndx.int64),
    ],
)
def test_create_iterators(arr):
    it = iter(arr)
    el = next(it)
    assert el.ndim == arr.ndim - 1
    assert el.shape == arr.shape[1:]


def test_0d_not_iterable():
    scalar = ndx.argument(shape=(), dtype=ndx.int64)
    with pytest.raises(ValueError):
        next(iter(scalar))


def test_raises_dynamic_dim():
    with pytest.raises(ValueError):
        iter(ndx.argument(shape=("N",), dtype=ndx.int64))
