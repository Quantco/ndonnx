# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
"""Implementation of a "object" dtype that implements a str | None | np.nan enum."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar, cast

import numpy as np

from .._dtypes import TY_ARRAY_BASE, DType
from .._schema import DTypeInfoV1
from . import onnx
from .funcs import astyarray
from .typed_array import TyArrayBase

if TYPE_CHECKING:
    from spox import Var
    from typing_extensions import Self

    from .._array import OnnxShape
    from .indexing import GetitemIndex, SetitemIndex

_PyScalar = bool | int | float | str | None
_NestedSequence = Sequence["bool | int | float | str | _NestedSequence"]

OBJECT_ARRAY = TypeVar("OBJECT_ARRAY", bound="TyObjectArray")
OBJECT_ARRAY_co = TypeVar("OBJECT_ARRAY_co", bound="TyObjectArray", covariant=True)


class ObjectDtype(DType[OBJECT_ARRAY_co]):
    def __ndx_create__(
        self, val: _PyScalar | np.ndarray | TyArrayBase | Var | _NestedSequence
    ) -> OBJECT_ARRAY_co:
        if val is None:
            return TyObjectArray(
                variant=onnx.uint8.__ndx_create__(
                    TyObjectArray._none_encoding,
                ),
                string_data=onnx.utf8.__ndx_create__(
                    "<NONE>",
                ),
            ).astype(self)
        elif isinstance(val, float) and np.isnan(val):
            return TyObjectArray(
                variant=onnx.uint8.__ndx_create__(
                    TyObjectArray._nan_encoding,
                ),
                string_data=onnx.utf8.__ndx_create__(
                    "<NAN>",
                ),
            ).astype(self)
        elif isinstance(val, str):
            return TyObjectArray(
                variant=onnx.uint8.__ndx_create__(
                    TyObjectArray._string_encoding,
                ),
                string_data=onnx.utf8.__ndx_create__(
                    val,
                ),
            ).astype(self)
        elif isinstance(val, np.ndarray):
            if val.dtype != object:
                raise ValueError(f"'val' has dtype {val.dtype}, required {object}")
            else:

                def _determine_variant(x):
                    if isinstance(x, float) and np.isnan(x):
                        return TyObjectArray._nan_encoding
                    elif isinstance(x, str):
                        return TyObjectArray._string_encoding
                    elif x is None:
                        return TyObjectArray._none_encoding
                    else:
                        raise ValueError(f"Cannot determine variant for {x}")

                variants = np.vectorize(
                    _determine_variant,
                    otypes=[
                        np.uint8,
                    ],
                )(val)
                string_data = val.astype(np.str_)
                return TyObjectArray(
                    variant=onnx.uint8.__ndx_create__(
                        variants,
                    ),
                    string_data=onnx.utf8.__ndx_create__(
                        string_data,
                    ),
                ).astype(self)
        elif isinstance(val, Sequence):
            return self.__ndx_create__(np.asarray(val, dtype=object))
        else:
            raise NotImplementedError

    def __ndx_result_type__(self, other: DType | _PyScalar) -> DType:
        if (
            isinstance(other, str)
            or other is None
            or (isinstance(other, float) and np.isnan(other))
        ):
            return self
        elif isinstance(other, type(self)):
            return self
        else:
            return NotImplemented

    def __repr__(self) -> str:
        return "ObjectDtype()"

    @property
    def _tyarr_class(self) -> type[TyObjectArray]:
        return TyObjectArray

    def __ndx_argument__(self, shape: OnnxShape) -> OBJECT_ARRAY_co:
        return TyObjectArray(
            variant=onnx.uint8.__ndx_argument__(shape),
            string_data=onnx.utf8.__ndx_argument__(shape),
        ).astype(self)

    # Construction functions
    def _arange(
        self,
        start: int | float,
        stop: int | float,
        step: int | float = 1,
    ) -> OBJECT_ARRAY_co:
        raise NotImplementedError

    def _eye(
        self,
        n_rows: int,
        n_cols: int | None = None,
        /,
        *,
        k: int = 0,
    ) -> OBJECT_ARRAY_co:
        raise NotImplementedError

    def _ones(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> OBJECT_ARRAY_co:
        raise NotImplementedError

    def _zeros(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> OBJECT_ARRAY_co:
        raise NotImplementedError

    @property
    def __ndx_infov1__(self) -> DTypeInfoV1:
        return DTypeInfoV1(
            author="ndonnx",
            type_name=self.__class__.__name__,
            meta=None,
        )

    def __ndx_cast_from__(self, arr: TyArrayBase) -> OBJECT_ARRAY_co:
        if isinstance(arr, onnx.TyArrayUtf8):
            return TyObjectArray(
                variant=onnx.uint8.__ndx_ones__(arr.dynamic_shape)
                * self._tyarr_class._string_encoding,  # type: ignore
                string_data=arr,
            )
        else:
            raise NotImplementedError


class TyObjectArray(TyArrayBase):
    variant: onnx.TyArrayUInt8
    string_data: onnx.TyArrayUtf8

    _nan_encoding, _none_encoding, _string_encoding = range(3)

    def __init__(
        self, variant: onnx.TyArrayUInt8, string_data: onnx.TyArrayUtf8
    ) -> None:
        self.variant = variant
        self.string_data = string_data

    @property
    def dtype(self) -> ObjectDtype:
        return object_dtype

    def copy(self) -> Self:
        return type(self)(
            variant=self.variant.copy(),
            string_data=self.string_data.copy(),
        )

    def disassemble(self) -> dict[str, Var]:
        return {
            "variant": self.variant.disassemble(),
            "string_data": self.string_data.disassemble(),
        }

    @property
    def mT(self) -> Self:  # noqa: N802
        variant = self.variant.mT
        string_data = self.string_data.mT
        return type(self)(
            variant=variant,
            string_data=string_data,
        )

    def __ndx_value_repr__(self):
        return {
            "variant": self.variant.__ndx_value_repr__()["data"],
            "string_data": self.string_data.__ndx_value_repr__()["data"],
        }

    def __getitem__(self, index: GetitemIndex) -> Self:
        variant = self.variant[index]
        string_data = self.string_data[index]
        return type(self)(
            variant=variant,
            string_data=string_data,
        )

    def __setitem__(
        self,
        key: SetitemIndex,
        value: Self,
        /,
    ) -> None:
        if self.dtype != value.dtype:
            raise TypeError(
                f"Cannot set value of type {value.dtype} on array of type {self.dtype}"
            )
        self.variant[key] = value.variant
        self.string_data[key] = value.string_data

    def put(
        self,
        key: onnx.TyArrayInt64,
        value: Self,
        /,
    ) -> None:
        self.variant.put(key, value.variant)
        self.string_data.put(key, value.string_data)

    @property
    def dynamic_shape(self) -> onnx.TyArrayInt64:
        return self.variant.dynamic_shape

    @property
    def shape(self) -> OnnxShape:
        return self.variant.shape

    def broadcast_to(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> Self:
        return type(self)(
            variant=self.variant.broadcast_to(shape),
            string_data=self.string_data.broadcast_to(shape),
        )

    def concat(self, others: list[Self], axis: None | int) -> Self:
        variant = self.variant.concat([arr.variant for arr in others], axis=axis)
        string_data = self.string_data.concat(
            [arr.string_data for arr in others], axis=axis
        )
        return type(self)(
            variant=variant,
            string_data=string_data,
        )

    def reshape(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> Self:
        variant = self.variant.reshape(shape)
        string_data = self.string_data.reshape(shape)
        return type(self)(
            variant=variant,
            string_data=string_data,
        )

    def squeeze(self, /, axis: int | tuple[int, ...]) -> Self:
        variant = self.variant.squeeze(axis)
        string_data = self.string_data.squeeze(axis)
        return type(self)(
            variant=variant,
            string_data=string_data,
        )

    def permute_dims(self, axes: tuple[int, ...]) -> Self:
        variant = self.variant.permute_dims(axes)
        string_data = self.string_data.permute_dims(axes)
        return type(self)(
            variant=variant,
            string_data=string_data,
        )

    def isnan(self) -> onnx.TyArrayBool:
        return self.variant._eqcomp(self._nan_encoding)

    def __ndx_cast_to__(self, dtype: DType[TY_ARRAY_BASE]) -> TY_ARRAY_BASE:
        return NotImplemented

    def __ndx_where__(
        self, cond: onnx.TyArrayBool, other: TyArrayBase, /
    ) -> TyArrayBase:
        from .funcs import where

        other_casted = other.astype(self.dtype)
        variant = cast(
            onnx.TyArrayUInt8, where(cond, self.variant, other_casted.variant)
        )
        string_data = cast(
            onnx.TyArrayUtf8, where(cond, self.string_data, other_casted.string_data)
        )
        return type(self)(
            variant=variant,
            string_data=string_data,
        )

    def __add__(self, rhs: TyArrayBase | _PyScalar) -> Self:
        from .funcs import where

        rhs_array = self.dtype.__ndx_create__(rhs)
        return type(self)(
            variant=where(
                cast(onnx.TyArrayBool, self.variant != rhs_array.variant),
                astyarray(self._nan_encoding).astype(onnx.uint8),
                self.variant,
            ).astype(onnx.uint8),
            string_data=self.string_data + rhs.string_data,  # type: ignore
        )

    def _eqcomp(self, other: TyArrayBase | _PyScalar) -> onnx.TyArrayBool:  # type: ignore
        from .funcs import where

        if isinstance(other, onnx.TyArrayUtf8):
            return (self.variant == self._string_encoding) and (
                self.string_data == other
            )  # type: ignore
        else:
            rhs_array = self.dtype.__ndx_create__(other)
            return where(
                cast(onnx.TyArrayBool, self.variant == self._string_encoding),
                (rhs_array.variant == self._string_encoding)
                and (self.string_data == rhs_array),
                self.variant == rhs_array.variant,
            )  # type: ignore

    def unwrap_numpy(self) -> np.ndarray:
        variant = self.variant.unwrap_numpy()
        string_data = self.string_data.unwrap_numpy()

        out = string_data.copy().astype(object)
        return np.where(
            variant == self._none_encoding,
            np.asarray(None, dtype=object),
            np.where(
                variant == self._nan_encoding, np.asarray(np.nan, dtype=object), out
            ),
        ).astype(object)


object_dtype: ObjectDtype = ObjectDtype()

__all__ = [
    "object_dtype",
]
