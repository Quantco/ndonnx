# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause


def test_import():
    import ndonnx  # noqa


def test_onnxruntime():
    import onnxruntime

    # Test that the onnxruntime has InfereceSession
    assert hasattr(onnxruntime, "InferenceSession")
