# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort

import ndonnx as ndx


def build_and_run(fn, *np_args):
    # Only works for ONNX data types
    ins_np = {f"in{i}": arr for i, arr in enumerate(np_args)}
    ins = {
        k: ndx.argument(shape=a.shape, dtype=ndx.from_numpy_dtype(a.dtype))
        for k, a in ins_np.items()
    }

    out = {"out": fn(*ins.values())}
    mp = ndx.build(ins, out)
    session = ort.InferenceSession(mp.SerializeToString())
    (out,) = session.run(None, {f"{k}": a for k, a in ins_np.items()})
    return out


def run(
    model_proto: onnx.ModelProto, inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    # Extract input and output schemas
    output_names = [el.name for el in model_proto.graph.output]
    # input_dtypes, output_dtypes = _get_dtypes(model_proto)
    session = _make_session(model_proto)
    # flattened_inputs = _deconstruct_inputs(inputs, input_dtypes)
    # output_names = tuple(_extract_output_names(output_dtypes))

    return dict(zip(output_names, session.run(output_names, inputs)))


def _make_session(model_proto: onnx.ModelProto) -> ort.InferenceSession:
    session_options = ort.SessionOptions()
    return ort.InferenceSession(
        model_proto.SerializeToString(), sess_options=session_options
    )


def get_numpy_array_api_namespace():
    if np.__version__ < "2":
        import numpy.array_api as npx

        return npx
    else:
        return np


def assert_array_equal(
    actual: np.ndarray,
    expected: np.ndarray,
):
    if np.asarray(actual).dtype.kind == np.asarray(expected).dtype.kind == "U":
        actual = np.asarray(actual, object)
        expected = np.asarray(expected, object)
    np.testing.assert_array_equal(
        actual, expected, strict=np.__version__.startswith("2")
    )
    assert isinstance(expected, np.ma.masked_array) == isinstance(
        actual, np.ma.masked_array
    )
    if isinstance(expected, np.ma.masked_array):
        np.testing.assert_array_equal(actual.mask, expected.mask)


def assert_equal_dtype_shape(arr, dtype, shape):
    assert arr.dtype == dtype
    assert arr._tyarray.shape == shape
    assert arr.shape == tuple(None if isinstance(el, str) else el for el in shape)
