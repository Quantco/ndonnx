# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
"""Implementation of "custom" datetime-related data types."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._dtypes import TY_ARRAY_BASE, DType
from .._schema import DTypeInfoV1
from . import onnx
from .typed_array import TyArrayBase

if TYPE_CHECKING:
    from types import NotImplementedType

    from spox import Var
    from typing_extensions import Self

    from .._array import OnnxShape
    from .indexing import (
        GetitemIndex,
        SetitemIndex,
    )
    from .onnx import TyArrayInt64


_N_MAX_CATEGORIES = np.iinfo(np.uint16).max
_PyScalar = bool | int | float | str


class CategoricalDType(DType["CategoricalArray"]):
    """Categorical data type with pandas-like semantics.

    Contrary to Pandas, the `categories` and `ordered` attributes of
    the constructor are mandatory.
    """

    _categories: list[str]
    _ordered: bool

    def __init__(self, categories: list[str], ordered: bool):
        self.categories = categories
        self._ordered = ordered

    @property
    def categories(self) -> list[str]:
        """List of unique categories."""
        return self._categories

    @categories.setter
    def categories(self, categories: list[str]):
        """List of unique categories."""
        if len(categories) != len(set(categories)):
            raise ValueError("provided categories must be unique")
        if not all(isinstance(el, str) for el in categories):
            raise TypeError("provided categories must all be of type 'str'")
        # TODO: Use uint8 if we have fewer categories. For now, uint8
        # support may be too limited in onnxruntime to provide a
        # benefit.
        if len(categories) >= _N_MAX_CATEGORIES:
            raise ValueError(
                f"at most '{_N_MAX_CATEGORIES} may be provided, found `{len(categories)}`"
            )
        self._categories = categories

    @property
    def ordered(self) -> bool:
        return self._ordered

    def __ndx_cast_from__(self, arr: TyArrayBase) -> CategoricalArray:
        if not isinstance(arr, onnx.TyArrayUtf8):
            raise NotImplementedError

        encoding = {k: i for i, k in enumerate(self.categories)}
        # TODO: Add possibility to specify value dtype of
        # `static_map`. At the moment, this would give us no practical
        # benefit due to lacking support in onnxruntime, though.
        codes = arr.apply_mapping(encoding, default=-1).astype(onnx.uint16)

        return CategoricalArray(codes=codes, dtype=self)

    def __ndx_result_type__(
        self, other: DType | _PyScalar
    ) -> DType | NotImplementedType:
        return NotImplemented

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(categories={self.categories}, ordered={self._ordered})"

    def __ndx_argument__(self, shape: OnnxShape) -> CategoricalArray:
        codes = onnx.uint16.__ndx_argument__(shape)
        return CategoricalArray(codes, dtype=self)

    @property
    def __ndx_infov1__(self) -> DTypeInfoV1:
        return DTypeInfoV1(
            author="ndonnx",
            type_name=self.__class__.__name__,
            meta={"categories": list(self.categories), "ordered": self._ordered},
        )


class CategoricalArray(TyArrayBase):
    _dtype: CategoricalDType
    # TODO: Flexible data type?
    codes: onnx.TyArrayUInt16

    def __init__(self, codes: onnx.TyArrayUInt16, dtype: CategoricalDType):
        self.codes = codes
        self._dtype = dtype

    @property
    def dtype(self) -> CategoricalDType:
        return self._dtype

    def __ndx_value_repr__(self) -> dict[str, str]:
        mapping = dict(enumerate(self.dtype.categories))
        cats = self.codes.apply_mapping(mapping, default="<NA>")
        return {
            "categories": cats.__ndx_value_repr__()["data"],
        }

    def __getitem__(self, index: GetitemIndex) -> Self:
        codes = self.codes[index]
        return type(self)(codes=codes, dtype=self.dtype)

    def __setitem__(
        self,
        key: SetitemIndex,
        value: Self,
        /,
    ) -> None:
        if self.dtype != value.dtype:
            TypeError(f"data type of 'value' must much array's, found `{value.dtype}`")
        self.codes[key] = value.codes

    def put(
        self,
        key: TyArrayInt64,
        value: Self,
        /,
    ) -> None:
        if self.dtype != value.dtype:
            TypeError(f"data type of 'value' must much array's, found `{value.dtype}`")
        self.codes.put(key, value.codes)

    @property
    def dynamic_shape(self) -> TyArrayInt64:
        return self.codes.dynamic_shape

    @property
    def mT(self) -> Self:  # noqa: N802
        codes = self.codes.mT
        return type(self)(codes=codes, dtype=self.dtype)

    @property
    def shape(self) -> OnnxShape:
        return self.codes.shape

    def reshape(self, shape: tuple[int, ...] | TyArrayInt64) -> Self:
        codes = self.codes.reshape(shape)
        return type(self)(codes=codes, dtype=self.dtype)

    def squeeze(self, /, axis: int | tuple[int, ...]) -> Self:
        codes = self.codes.squeeze(axis=axis)
        return type(self)(codes=codes, dtype=self.dtype)

    def disassemble(self) -> dict[str, Var] | Var:
        return {"codes": self.codes.var}

    def broadcast_to(self, shape: tuple[int, ...] | TyArrayInt64) -> Self:
        codes = self.codes.broadcast_to(shape=shape)
        return type(self)(codes=codes, dtype=self.dtype)

    def concat(self, others: list[Self], axis: None | int) -> Self:
        codes = self.codes.concat([el.codes for el in others], axis)
        return type(self)(codes=codes, dtype=self.dtype)

    def copy(self) -> Self:
        codes = self.codes.copy()
        return type(self)(codes=codes, dtype=self.dtype)

    def permute_dims(self, axes: tuple[int, ...]) -> Self:
        codes = self.codes.permute_dims(axes)
        return type(self)(codes=codes, dtype=self.dtype)

    def __ndx_cast_to__(self, dtype: DType[TY_ARRAY_BASE]) -> TY_ARRAY_BASE:
        if dtype != onnx.utf8:
            return NotImplemented

        mapping = dict(enumerate(self.dtype.categories))
        cats = self.codes.apply_mapping(mapping, default="<NA>")

        return cats.astype(dtype)

    def _eqcomp(self, other: TyArrayBase | _PyScalar) -> TyArrayBase:  # type: ignore
        from .._infos import iinfo
        from .funcs import astyarray

        if isinstance(other, _PyScalar):
            if isinstance(other, bool | int | float):
                return NotImplemented
            other = astyarray(other, dtype=onnx.utf8)

        if other.dtype == onnx.utf8:
            return self == other.astype(self.dtype)

        if isinstance(other.dtype, CategoricalDType):
            if self.dtype != other.dtype:
                # We directly raise here to be able to provide a better error message
                raise TypeError(
                    "comparison between arrays of categorical type requires data type to be precisely equal."
                )
            # Unclear why mypy would not figure out the type of `other` here?!
            bools = self.codes == other.codes  # type: ignore
            not_missing = self.codes != astyarray(
                iinfo(self.codes.dtype).max, self.codes.dtype
            )
            return bools & not_missing

        return NotImplemented

    def _to_categories(self) -> onnx.TyArrayUtf8:
        mapping = dict(enumerate(self.dtype.categories))
        cats = self.codes.apply_mapping(mapping, default="<NA>")

        return cats.astype(onnx.utf8)

    def unwrap_numpy(self) -> np.ndarray:
        """Cast array to utf8 and replace missing with ``numpy.nan`` values.

        The returned array has the ``object`` data type.
        """
        objs = self._to_categories().unwrap_numpy().astype(object)
        objs[self.codes.unwrap_numpy() == _N_MAX_CATEGORIES] = np.nan

        return objs
