# Copyright (c) QuantCo 2023-2026
# SPDX-License-Identifier: BSD-3-Clause

import inspect

import pytest

import ndonnx as ndx

NDONNX_EXCLUDE = [
    "isdtype",
    "to_nullable_dtype",
]

PUBLIC_INTERFACE = [
    *[
        (ndx, name)  # public functions of `ndonnx`
        for name, _ in inspect.getmembers(ndx, predicate=inspect.isfunction)
        if (
            not name.startswith("_")
            and callable(getattr(ndx, name))
            and name not in NDONNX_EXCLUDE
        )
    ],
    *[
        (ndx.Array, name)  # public methods of `ndonnx.Array`
        for name in dir(ndx.Array)
        if not name.startswith("_") and callable(getattr(ndx.Array, name))
    ],
]


@pytest.mark.parametrize("module, name", PUBLIC_INTERFACE)
def test_dtype_and_aliases(module, name):
    member = getattr(module, name)
    sig = inspect.signature(member)
    if dtype_param := sig.parameters.get("dtype"):
        # If a public member has a `dtype` argument, then it must be annotated
        # both with `DType` (i.e. native `ndonnx`) and `DTypeAlias` (others
        # that map to a native `ndonnx` dtype).
        assert "DType" in dtype_param.annotation
        assert "DTypeAlias" in dtype_param.annotation
