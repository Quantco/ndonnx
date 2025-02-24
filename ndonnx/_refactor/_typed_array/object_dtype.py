"""Implementation of a "object" dtype that implements a str | None | np.nan enum"""


from __future__ import annotations
import operator

import numpy as np

from .._dtypes import TY_ARRAY_BASE, DType
from .._schema import DTypeInfoV1
from . import onnx

from .funcs import astyarray
from .typed_array import TyArrayBase

from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from .indexing import GetitemIndex, SetitemIndex
    from .._array import OnnxShape
    from typing_extensions import Self

    from spox import Var

_PyScalar = bool | int | float | str | None

OBJECT_ARRAY = TypeVar("OBJECT_ARRAY", bound="TyObjectArray")

class ObjectDtype(DType[OBJECT_ARRAY]):
    def __ndx_result_type__(self, other: DType | _PyScalar) -> DType:
        if isinstance(other, str) or other is None or (isinstance(other, float) and np.isnan(other)):
            return self
        elif isinstance(other, type(self)):
            return self
        else:
            return NotImplemented

    def __repr__(self) -> str:
        return "ObjectDtype"

    @property
    def _tyarr_class(self) -> type[TyObjectArray]:
        return TyObjectArray

    def _argument(self, shape: OnnxShape) -> OBJECT_ARRAY:
        raise TyObjectArray(
            variant=onnx.uint8._argument(shape),
            string_data=onnx.utf8._argument(shape),
        )


    # Construction functions
    def _arange(
        self,
        start: int | float,
        stop: int | float,
        step: int | float = 1,
    ) -> TyObjectArray:
        raise NotImplementedError

    def _eye(
        self,
        n_rows: int,
        n_cols: int | None = None,
        /,
        *,
        k: int = 0,
    ) -> TyObjectArray:
        raise NotImplementedError

    def _ones(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> TyObjectArray:
        raise NotImplementedError

    def _zeros(self, shape: tuple[int, ...] | onnx.TyArrayInt64) -> TyObjectArray:
        raise NotImplementedError

    @property
    def _infov1(self) -> DTypeInfoV1:
        return DTypeInfoV1(
            author="ndonnx", type_name=self.__class__.__name__, meta=None,
        )

    def __ndx_cast_from__(self, arr: TyArrayBase) -> TyObjectArray:
        if isinstance(arr, onnx.TyArrayUtf8):
            return TyObjectArray(
                variant=onnx.uint8._ones(arr.shape) * self._tyarr_class._string_encoding, # type: ignore
                string_data=arr,
            )
        else:
            raise NotImplementedError




class TyObjectArray(TyArrayBase):
    dtype: ObjectDtype
    variant: onnx.TyArrayUInt8
    string_data: onnx.TyArrayUtf8

    _nan_encoding, _none_encoding, _string_encoding = range(3)

    def __init__(self, variant: onnx.TyArrayUInt8, string_data: onnx.TyArrayUtf8) -> None:
        self.variant = variant
        self.string_data = string_data
        self.dtype = object_dtype

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

    def __ndx_value_repr__(self):
       return {
           "variant": self.variant.__ndx_value_repr__(),
           "string_data": self.string_data.__ndx_value_repr__(),
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
            raise TypeError(f"Cannot set value of type {value.dtype} on array of type {self.dtype}")
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
    def mt(self) -> Self:
        variant = self.variant.mT
        string_data = self.string_data.mT
        return type(self)(
            variant=variant,
            string_data=string_data,
        )

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
        string_data = self.string_data.concat([arr.string_data for arr in others], axis=axis)
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

        other = other.astype(self.dtype) if other.dtype == onnx.utf8 else other

        if self.dtype != other.dtype:
           return NotImplemented

        variant = cast(onnx.TyArrayUInt8, where(cond, self.variant, other.variant)) # type: ignore
        string_data = cast(onnx.TyArrayUtf8, where(cond, self.string_data, other.string_data)) # type: ignore
        return type(self)(
            variant=variant,
            string_data=string_data,
        )

    @property
    def mT(self) -> Self:
        variant = self.variant.mT
        string_data = self.string_data.mT
        return type(self)(
            variant=variant,
            string_data=string_data,
        )

    def __add__(self, rhs: TyArrayBase | _PyScalar) -> Self:
        from .funcs import astyarray, where

        rhs_array = _asarray(rhs)
        # TODO: Where variants disagree, arguably we should really crash.
        # However, we need a custom operator for this and this lives in
        # an internal library, while we are prototyping in
        # the ndonnx repository.
        # Pandas seems to just give np.nan on disagreement at time of writing.

        # We're just forced to define some behaviour. We will make None and nan infectious and take the minimum code.
        return type(self)(
            variant=where(self.variant != rhs_array, astyarray(self._nan_encoding, dtype=onnx.uint8), self.variant), # type: ignore
            string_data=self.string_data+rhs.string_data, # type: ignore
        )

    def _eqcomp(self, other: TyArrayBase | _PyScalar) -> TyArrayBool:  # type: ignore
        from .funcs import where

        if isinstance(other, onnx.TyArrayUtf8):
            return (self.variant == self._string_encoding) and (self.string_data == other)
        else:
            rhs_array = _asarray(other)
            return where(
                cast(onnx.TyArrayBool, self.variant == self._string_encoding),
                (rhs_array.variant == self._string_encoding) and (self.string_data == rhs_array),
                self.variant == rhs_array.variant,
            )

    def unwrap_numpy(self) -> np.ndarray:
        variant = self.variant.unwrap_numpy()
        string_data = self.string_data.unwrap_numpy()

        out = string_data.copy().astype(object)
        return np.where(variant == self._none_encoding, np.asarray(None, dtype=object), np.where(variant == self._nan_encoding, np.asarray(np.nan, dtype=object), out)).astype(object)

# TODO: why can I not define "asarray" for my dtype? If I can, how do I do this?
def _asarray(item: _PyScalar | TyArrayBase) -> TyObjectArray:
    if item is None:
        return TyObjectArray(
            variant=astyarray(TyObjectArray._none_encoding, dtype=onnx.uint8),
            string_data=astyarray("<NONE>", dtype=onnx.utf8),
        )
    elif isinstance(item, float) and np.isnan(item):
        return TyObjectArray(
            variant=astyarray(TyObjectArray._nan_encoding, dtype=onnx.uint8),
            string_data=astyarray("<NAN>", dtype=onnx.utf8),
        )
    elif isinstance(item, str):
        return TyObjectArray(
            variant=astyarray(TyObjectArray._string_encoding, dtype=onnx.uint8),
            string_data=astyarray(item, dtype=onnx.utf8),
        )
    elif isinstance(item, onnx.TyArrayUtf8):
        return TyObjectArray(
            variant=astyarray(TyObjectArray._string_encoding, dtype=onnx.uint8)._broadcast_to(item.dynamic_shape),
            string_data=item,
        )
    elif isinstance(item, TyObjectArray):
        return item.copy()
    else:
        raise TypeError(f"Cannot turn {item} into a {TyObjectArray}")

object_dtype = ObjectDtype()

__all__ = [
    "object_dtype",
]
