# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
"""Dummy module for the array api tests which ensures that we are using onnxruntime to
propagate values."""

import spox

spox._value_prop._VALUE_PROP_BACKEND = spox._value_prop.ValuePropBackend.ONNXRUNTIME

from ndonnx import *  # noqa
