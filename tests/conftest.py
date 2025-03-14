# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from spox import _future, _value_prop


@pytest.fixture(autouse=True)
def warn_when_prop_fails():
    _future.set_type_warning_level(_future.TypeWarningLevel.OUTPUTS)
    yield


@pytest.fixture(autouse=True)
def use_spox_ort_value_prop():
    # TODO: parametrize over the reference runtime and onnxruntime once the former becomes more mature.
    _value_prop._VALUE_PROP_BACKEND = _value_prop.ValuePropBackend.ONNXRUNTIME
