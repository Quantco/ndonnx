# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from .. import dtypes
from ..dtypes import DType
from ._core_types import _ArrayCoreNum, _ArrayCoreType
from ._typed_array import DTYPE, _TypedArray

if TYPE_CHECKING:
    from ..array import OnnxShape


class _ArrayPyScalar(_TypedArray[DTYPE]):
    value: int | float

    def __init__(self, value: int | float):
        self.value = value

    @classmethod
    def from_data(cls, data: _TypedArray):
        # TODO
        raise NotImplementedError

    def __getitem__(self, index) -> Self:
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> OnnxShape:
        return ()

    @classmethod
    def from_np_schema(cls, schema: dict[str, Any], /) -> Self:
        if "value" in schema and len(schema) == 1:
            (val,) = schema.values()
            return cls(val)
        raise ValueError("'schema' has unexpected layout")

    def reshape(self, shape: tuple[int, ...]) -> Self:
        # TODO: Should reshape be moved into a different base class?
        raise ValueError("cannot reshape Python scalar")

    def promote(self, *others: _TypedArray) -> Sequence[_TypedArray]:
        # TODO
        raise NotImplementedError

    def _promote(self, other: _ArrayCoreType) -> tuple[_ArrayCoreType, _ArrayCoreType]:
        result_type = self.dtype._result_type(other.dtype)

        self.astype(result_type)
        raise NotImplementedError

    def __add__(self, rhs: _TypedArray) -> _ArrayCoreType:
        if isinstance(rhs, _ArrayCoreNum):
            lhs, rhs = self._promote(rhs)
            return lhs + rhs
        return NotImplemented

    def __or__(self, rhs: _TypedArray) -> _ArrayCoreType:
        return NotImplemented

    def _astype(self, dtype: DType) -> _TypedArray:
        # TODO
        raise NotImplementedError


class _ArrayPyInt(_ArrayPyScalar[dtypes._PyInt]):
    dtype = dtypes._pyint


class _ArrayPyFloat(_ArrayPyScalar[dtypes._PyFloat]):
    dtype = dtypes._pyfloat
