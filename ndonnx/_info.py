# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ndonnx as ndx
from ndonnx._array import ndonnx_device
from ndonnx._data_types import canonical_name


class ArrayNamespaceInfo:
    """Namespace metadata for the Array API standard."""

    _all_array_api_types = [
        ndx.bool,
        ndx.float32,
        ndx.float64,
        ndx.int8,
        ndx.int16,
        ndx.int32,
        ndx.int64,
        ndx.uint8,
        ndx.uint16,
        ndx.uint32,
        ndx.uint64,
    ]

    def capabilities(self) -> dict:
        return {
            "boolean indexing": True,
            "data-dependent shapes": True,
        }

    def default_device(self):
        return ndonnx_device

    def devices(self) -> list:
        return [ndonnx_device]

    def dtypes(
        self, *, device=None, kind: str | tuple[str, ...] | None = None
    ) -> dict[str, ndx.CoreType]:
        # We don't care for device since we are writing ONNX graphs.
        # We would rather not give users the impression that their arrays
        # are tied to a specific device when serializing an ONNX graph as
        # such a concept does not exist in the ONNX standard.
        out: dict[str, ndx.CoreType] = {}
        for dtype in self._all_array_api_types:
            if kind is None or ndx.isdtype(dtype, kind):
                out[canonical_name(dtype)] = dtype
        return out

    def default_dtypes(
        self,
        *,
        device=None,
    ) -> dict[str, ndx.CoreType | None]:
        # See comment in `dtypes` method regarding device.
        return {
            "real floating": ndx.float64,
            "integral": ndx.int64,
            "indexing": ndx.int64,
            # We don't support complex numbers yet due to immaturity in the ONNX ecoystem, so "complex floating" is meaningless.
            # The Array API standard requires this key to be present so we set it to None.
            "complex floating": None,
        }


def __array_namespace_info__() -> ArrayNamespaceInfo:  # noqa: N807
    return ArrayNamespaceInfo()
