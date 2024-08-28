# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import Self

from .. import dtypes
from ..dtypes import DType
from .core import _ArrayCoreNum, _ArrayCoreType
from .masked import _ArrayMaCoreType  # TODO: there should be num-subclass
from .typed_array import DTYPE, _TypedArray
from .utils import promote

if TYPE_CHECKING:
    from ..array import OnnxShape


class _ArrayPyScalar(_TypedArray[DTYPE]):
    """Array representing a Python scalar.

    This implementation is written as if it were a "custom" typed array which knows
    about other (nullable) core types. Those core types are oblivious of this typed
    array though. Thus, this implementation may serve as a blue print for custom types.
    """

    value: int | float

    def __init__(self, value: int | float):
        self.value = value

    @classmethod
    def from_typed_array(cls, tyarr: _TypedArray):
        # This class should only be created when Python scalars
        # interact with Array objects. As such it should only ever be
        # instantiated using the `__int__` method.
        raise ValueError(f"`{cls}` cannot be created from `{type(tyarr)}`")

    @classmethod
    def as_argument(cls, shape: OnnxShape):
        raise ValueError(f"`{cls}` cannot be an argument to a graph")

    def __getitem__(self, index) -> Self:
        raise IndexError(f"`{type(self)}` cannot be indexed")

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
        raise ValueError("cannot reshape Python scalar")

    def __add__(self, rhs: _TypedArray[DType]) -> _TypedArray[DType]:
        if isinstance(rhs, _ArrayCoreNum | _ArrayMaCoreType):
            lhs, rhs = promote(self, rhs)
            return lhs + rhs

        # We only know about the other (nullable) built-in types &
        # these scalars should never interact with themselves.
        return NotImplemented

    def __or__(self, rhs: _TypedArray) -> _ArrayCoreType:
        return NotImplemented

    def _astype(self, dtype: DType) -> _TypedArray:
        # We implement this class under the assumption that the other
        # built-in typed arrays do not know about it. Thus, we define
        # the mapping from this class into those classes here.
        if isinstance(dtype, dtypes.CoreNumericDTypes):
            return dtype._tyarr_class.from_np_schema(
                {"var": np.array(self.value, dtypes.as_numpy(dtype))}
            )
        raise NotImplementedError


class _ArrayPyInt(_ArrayPyScalar[dtypes._PyInt]):
    dtype = dtypes._pyint


class _ArrayPyFloat(_ArrayPyScalar[dtypes._PyFloat]):
    dtype = dtypes._pyfloat
