# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
"""Implementation of a "object" dtype that implements a str | None | np.nan enum."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np

from .._dtypes import TY_ARRAY_BASE, DType
from .._schema import DTypeInfoV1
from . import onnx
from .funcs import astyarray, where
from .typed_array import TyArrayBase
from .utils import safe_cast

if TYPE_CHECKING:
    from spox import Var
    from typing_extensions import Self

    from .._types import NestedSequence, OnnxShape, PyScalar
    from .indexing import GetitemIndex, SetitemIndex


# TODO: The name is unfortunate. We cover far less than the "object"
# data type and should thus find a better name.
class ObjectDtype(DType["TyObjectArray"]):
    def _build(
        self, variant: onnx.TyArrayUInt8, string_data: onnx.TyArrayUtf8
    ) -> TyObjectArray:
        return TyObjectArray(variant=variant, string_data=string_data)

    def __ndx_create__(
        self, val: PyScalar | np.ndarray | TyArrayBase | Var | NestedSequence
    ) -> TyObjectArray:
        if val is None:
            # TODO: `None` is not a valid value in the public
            # interface of as(ty)array. Do we really want to support
            # this additional code path? The user can already use
            # NumPy object arrays to the same effect.
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
                raise ValueError(f"'val' has dtype `{val.dtype}`, required 'object'")
            else:
                variants = determine_variant(val)
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

    def __ndx_result_type__(self, other: DType | PyScalar) -> DType:
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

    def __ndx_argument__(self, shape: OnnxShape) -> TyObjectArray:
        return TyObjectArray(
            variant=onnx.uint8.__ndx_argument__(shape),
            string_data=onnx.utf8.__ndx_argument__(shape),
        ).astype(self)

    @property
    def __ndx_infov1__(self) -> DTypeInfoV1:
        return DTypeInfoV1(
            author="ndonnx",
            type_name=self.__class__.__name__,
            meta=None,
        )

    def __ndx_cast_from__(self, arr: TyArrayBase) -> TyObjectArray:
        if isinstance(arr, onnx.TyArrayUtf8):
            return TyObjectArray(
                variant=astyarray(
                    TyObjectArray._string_encoding, dtype=onnx.uint8
                ).broadcast_to(arr.dynamic_shape),
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
        # TODO: Shouldn't this also include None?
        return self.variant.__ndx_equal__(self._nan_encoding)

    def __ndx_cast_to__(self, dtype: DType[TY_ARRAY_BASE]) -> TY_ARRAY_BASE:
        return NotImplemented

    def __ndx_where__(
        self, cond: onnx.TyArrayBool, other: TyArrayBase, /
    ) -> TyArrayBase:
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

    def __add__(self, rhs: TyArrayBase | PyScalar) -> Self:
        # TODO: I don't think we do ourselves a favor by defining
        # __add__ on this type. The semantics are opinionated by
        # nature.
        from .funcs import astyarray, where

        rhs_array = astyarray(rhs, dtype=object_dtype)
        return type(self)(
            variant=where(
                ~self.variant.__ndx_equal__(rhs_array.variant),
                astyarray(self._nan_encoding).astype(onnx.uint8),
                self.variant,
            ).astype(onnx.uint8),
            string_data=self.string_data + rhs_array.string_data,
        )

    def __ndx_equal__(self, other: TyArrayBase | PyScalar) -> onnx.TyArrayBool:
        # TODO: Document semantics of NaN == None etc somewhere.
        if isinstance(other, str):
            other = astyarray(other, dtype=onnx.utf8)
        if isinstance(other, onnx.TyArrayUtf8):
            return safe_cast(
                onnx.TyArrayBool,
                (self.variant == self._string_encoding) & (self.string_data == other),
            )
        else:
            if isinstance(other, onnx.TyArrayUtf8):
                rhs_array = astyarray(other, dtype=object_dtype)
            elif isinstance(other, type(self)):
                rhs_array = other
            else:
                raise TypeError(f"Cannot compare equal {type(self)} with {type(other)}")

            return safe_cast(
                onnx.TyArrayBool,
                where(
                    self.variant == self._string_encoding,
                    (rhs_array.variant == self._string_encoding)
                    and (self.string_data == rhs_array),
                    self.variant == rhs_array.variant,
                ),
            )

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


def _determine_variant(x):
    if isinstance(x, float) and np.isnan(x):
        return TyObjectArray._nan_encoding
    elif isinstance(x, str):
        return TyObjectArray._string_encoding
    elif x is None:
        return TyObjectArray._none_encoding
    else:
        raise ValueError(f"Cannot determine variant for {x}")


determine_variant = np.vectorize(_determine_variant, otypes=[np.uint8])

object_dtype: ObjectDtype = ObjectDtype()

__all__ = [
    "object_dtype",
]
