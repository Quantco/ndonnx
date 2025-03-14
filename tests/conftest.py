# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--update-schema-snapshots",
        action="store_true",
        help="Update schema snapshots for stability tests. This should only be used in extraordinary and well understood circumstances.",
    )


@pytest.fixture
def update_schema_snapshots(request):
    return request.config.getoption("--update-schema-snapshots")
