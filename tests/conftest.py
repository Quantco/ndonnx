# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from spox import _future, _value_prop


@pytest.fixture(autouse=True)
def warn_when_prop_fails():
    _future.set_type_warning_level(_future.TypeWarningLevel.OUTPUTS)
    yield


@pytest.fixture(autouse=True, params=["ort", "ref"])
def use_spox_ort_value_prop(monkeypatch: pytest.MonkeyPatch, request):
    # TODO: parametrize over the reference runtime and onnxruntime once the former becomes more mature.
    if request.param == "ort":
        monkeypatch.setattr(
            _value_prop, "_VALUE_PROP_BACKEND", _value_prop.ValuePropBackend.ONNXRUNTIME
        )
    elif request.param == "ref":
        monkeypatch.setattr(
            _value_prop, "_VALUE_PROP_BACKEND", _value_prop.ValuePropBackend.REFERENCE
        )
    else:
        raise NotImplementedError


def pytest_addoption(parser):
    parser.addoption(
        "--update-schema-snapshots",
        action="store_true",
        help="Update schema snapshots for stability tests. This should only be used in extraordinary and well understood circumstances.",
    )


@pytest.fixture
def update_schema_snapshots(request):
    return request.config.getoption("--update-schema-snapshots")
