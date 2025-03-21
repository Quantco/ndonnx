# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
"""Implementation of a categorical data type."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np

from ndonnx import DType
from ndonnx._experimental import DTypeInfoV1, TyArrayBase, onnx
from ndonnx.types import NestedSequence, OnnxShape, PyScalar

if TYPE_CHECKING:
    from types import NotImplementedType

    from spox import Var
    from typing_extensions import Self

    from ndonnx._experimental import GetitemIndex, SetitemIndex


_N_MAX_CATEGORIES = np.iinfo(np.int16).max
TY_ARRAY_BASE_co = TypeVar("TY_ARRAY_BASE_co", bound="TyArrayBase", covariant=True)


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
        # TODO: Use int8 if we have fewer categories. For now, int8
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

    def __ndx_create__(
        self, val: PyScalar | np.ndarray | TyArrayBase | Var | NestedSequence
    ) -> CategoricalArray:
        return onnx.utf8.__ndx_create__(val).astype(self)

    def __ndx_cast_from__(self, arr: TyArrayBase) -> CategoricalArray:
        if not isinstance(arr, onnx.TyArrayUtf8):
            raise NotImplementedError

        encoding = {k: i for i, k in enumerate(self.categories)}
        codes = arr.apply_mapping(encoding, default=-1).astype(onnx.int16)

        return CategoricalArray(codes=codes, dtype=self)

    def __ndx_result_type__(
        self, other: DType | PyScalar
    ) -> DType | NotImplementedType:
        return NotImplemented

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(categories={self.categories}, ordered={self._ordered})"

    def __ndx_argument__(self, shape: OnnxShape) -> CategoricalArray:
        codes = onnx.int16.__ndx_argument__(shape)
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
    _codes: onnx.TyArrayInt16

    def __init__(self, codes: onnx.TyArrayInt16, dtype: CategoricalDType):
        self._codes = codes
        self._dtype = dtype

    @property
    def dtype(self) -> CategoricalDType:
        return self._dtype

    def __ndx_value_repr__(self) -> dict[str, str]:
        mapping = dict(enumerate(self.dtype.categories))
        cats = self._codes.apply_mapping(mapping, default="<NA>")
        return {
            "categories": cats.__ndx_value_repr__()["data"],
        }

    def __getitem__(self, index: GetitemIndex) -> Self:
        codes = self._codes[index]
        return type(self)(codes=codes, dtype=self.dtype)

    def __setitem__(
        self,
        key: SetitemIndex,
        value: Self,
        /,
    ) -> None:
        if self.dtype != value.dtype:
            TypeError(f"data type of 'value' must much array's, found `{value.dtype}`")
        self._codes[key] = value._codes

    def put(
        self,
        key: onnx.TyArrayInt64,
        value: Self,
        /,
    ) -> None:
        if self.dtype != value.dtype:
            TypeError(f"data type of 'value' must much array's, found `{value.dtype}`")
        self._codes.put(key, value._codes)

    @property
    def dynamic_shape(self) -> onnx.TyArrayInt64:
        return self._codes.dynamic_shape

    @property
    def mT(self) -> Self:  # noqa: N802
        codes = self._codes.mT
        return type(self)(codes=codes, dtype=self.dtype)

    @property
    def shape(self) -> OnnxShape:
        return self._codes.shape

    @property
    def is_constant(self) -> bool:
        return self._codes.is_constant

    def reshape(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> Self:
        codes = self._codes.reshape(shape)
        return type(self)(codes=codes, dtype=self.dtype)

    def squeeze(self, /, axis: int | tuple[int, ...]) -> Self:
        codes = self._codes.squeeze(axis=axis)
        return type(self)(codes=codes, dtype=self.dtype)

    def disassemble(self) -> dict[str, Var] | Var:
        return {"codes": self._codes.disassemble()}

    def broadcast_to(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> Self:
        codes = self._codes.broadcast_to(shape=shape)
        return type(self)(codes=codes, dtype=self.dtype)

    def concat(self, others: list[Self], axis: None | int) -> Self:
        codes = self._codes.concat([el._codes for el in others], axis)
        return type(self)(codes=codes, dtype=self.dtype)

    def copy(self) -> Self:
        codes = self._codes.copy()
        return type(self)(codes=codes, dtype=self.dtype)

    def permute_dims(self, axes: tuple[int, ...]) -> Self:
        codes = self._codes.permute_dims(axes)
        return type(self)(codes=codes, dtype=self.dtype)

    def __ndx_cast_to__(self, dtype: DType[TY_ARRAY_BASE_co]) -> TY_ARRAY_BASE_co:
        if dtype != onnx.utf8:
            return NotImplemented

        mapping = dict(enumerate(self.dtype.categories))
        cats = self._codes.apply_mapping(mapping, default="<NA>")

        return cats.astype(dtype)

    def __ndx_equal__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:  # type: ignore
        from .funcs import astyarray

        if isinstance(other, PyScalar):
            if isinstance(other, bool | int | float):
                return NotImplemented
            other = astyarray(other, dtype=onnx.utf8)

        if other.dtype == onnx.utf8:
            return self == other.astype(self.dtype)

        if isinstance(other.dtype, CategoricalDType):
            if self.dtype != other.dtype:
                # We directly raise here to be able to provide a better error message
                raise TypeError(
                    "comparison between arrays of categorical type requires data type to be precisely equal"
                )
            # Unclear why mypy would not figure out the type of `other` here?!
            bools = self._codes == other._codes  # type: ignore
            not_missing = self._codes != -1
            return bools & not_missing

        return NotImplemented

    def _to_categories(self) -> onnx.TyArrayUtf8:
        mapping = dict(enumerate(self.dtype.categories))
        cats = self._codes.apply_mapping(mapping, default="<NA>")

        return cats.astype(onnx.utf8)

    def unwrap_numpy(self) -> np.ndarray:
        """Cast array to utf8 and replace missing with ``numpy.nan`` values.

        The returned array has the ``object`` data type.
        """
        objs = self._to_categories().unwrap_numpy().astype(object)
        objs[self._codes.unwrap_numpy() == -1] = np.nan

        return objs
