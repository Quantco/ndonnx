# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.array_api as npx
import pytest

import ndonnx as ndx
import ndonnx._opset_extensions as opx


@pytest.mark.parametrize("op", ["add", "multiply", "subtract", "divide"])
def test_eager_mode(op):
    a = ndx.asarray(np.array([0.0, 1, 2, 3]), dtype=ndx.float64)
    b = ndx.asarray(np.array([3.0, 4, 5, 6]), dtype=ndx.float64)
    np.testing.assert_equal(
        getattr(ndx, op)(a, b).to_numpy(),
        getattr(npx, op)(npx.asarray(a.to_numpy()), npx.asarray(b.to_numpy())),
    )


def test_indexing():
    a = ndx.asarray(np.array([0.0, 1, 2, 3]), dtype=ndx.float64)
    b = ndx.asarray(np.array([2, 1]), dtype=ndx.int64)
    np.testing.assert_equal(a[b].to_numpy(), a.to_numpy()[b.to_numpy()])  # type: ignore
    a[1] = 3.4
    np.testing.assert_equal(
        a.to_numpy(), np.array([0.0, 3.4, 2.0, 3.0], dtype=np.float64)
    )
    a[2] = 4.2
    np.testing.assert_equal(
        a.to_numpy(), np.array([0.0, 3.4, 4.2, 3.0], dtype=np.float64)
    )
    a[b] = ndx.asarray(np.array([-1.0, np.nan]))
    np.testing.assert_equal(
        a.to_numpy(), np.array([0.0, np.nan, -1.0, 3.0], dtype=np.float64)
    )


def test_slicing():
    a = ndx.asarray(np.array([0, 1, 2, 3], np.int64))
    b = ndx.asarray(np.array([1, 2], np.int64))
    c = a[1:3] * 2 + b
    np.testing.assert_equal(c.to_numpy(), np.array([3, 6], dtype=np.int64))


def test_creation():
    a = ndx.asarray(np.array([0.0, 1, 2, 3]))
    assert a.to_numpy() is not None
    np.testing.assert_equal(a.to_numpy(), np.array([0.0, 1, 2, 3], dtype=np.float64))


def test_opset_extensions_eager_propagation():
    a = ndx.asarray(
        np.arange(0, 4, dtype=np.int64),
    )

    b = ndx.asarray(
        np.arange(10, 14, dtype=np.int64),
    )
    result = opx.add(a.data, b.data)  # type: ignore
    np.testing.assert_equal(result.to_numpy(), [10, 12, 14, 16])
    result = opx.split(opx.concat([a.data, b.data], axis=0), num_outputs=2, axis=0)  # type: ignore
    assert isinstance(result, tuple) and len(result) == 2
    np.testing.assert_equal(result[0].to_numpy(), a.to_numpy())
    np.testing.assert_equal(result[1].to_numpy(), b.to_numpy())


@pytest.mark.skip(
    reason="Opset dispatch constant propagation leads to slow build times."
)
def test_opset_extensions_constant_folding():
    a = ndx.asarray(np.array([1, 2, 3]))
    b = ndx.asarray(np.array([4, 5, 6]))
    result = opx.add(a, b)
    assert result._var._op.op_type.identifier == "Constant"
    np.testing.assert_equal(result.to_numpy(), [5, 7, 9])


def test_eager_propagate_decorator():
    import ndonnx._opset_extensions as opx
    from ndonnx._array import _from_corearray
    from ndonnx._propagation import eager_propagate

    @eager_propagate
    def sin_plus(a, b):
        return _from_corearray(opx.sin(a.data)) + b

    a = ndx.asarray(
        np.array([0, 1, 2, 3], dtype=np.float64),
    )
    b = ndx.asarray(
        np.array([10, 11, 12, 13], dtype=np.float64),
    )
    result = sin_plus(a, b)
    np.testing.assert_equal(result.to_numpy(), np.sin(a.to_numpy()) + b.to_numpy())

    @eager_propagate
    def assign_inplace(a, indices, values):
        a[indices] = values
        return a

    index = ndx.asarray(np.array([0, 3], dtype=np.int64))
    c = ndx.asarray(np.array([100, 200], dtype=np.int64))
    result = assign_inplace(a, index, c)
    np.testing.assert_equal(result.to_numpy(), [100, 1, 2, 200])
    np.testing.assert_equal(a.to_numpy(), [100, 1, 2, 200])
    assert a is result
