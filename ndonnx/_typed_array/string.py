# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, Literal, TypeVar

import numpy as np
from spox import Var
from typing_extensions import Self

from ndonnx import DType
from ndonnx._experimental import TyArrayBase, onnx
from ndonnx._schema import DTypeInfoV1
from ndonnx._typed_array._utils import safe_cast
from ndonnx._typed_array.onnx import SetitemIndex, TyArrayUtf8
from ndonnx.types import NestedSequence, OnnxShape, PyScalar

# StringDType[TStringBase] -> StringDTypeNoMissing
#                          -> StringDTypeSentinel
#                          -> StringDTypeNan
# StringBase -> StringNoMissing
#            -> StringSentinel
#            -> StringNan


class _NotSet: ...


_not_set = _NotSet()


class _Nan:
    def __init__(self, value: float):
        if value == value:
            raise ValueError(f"expected 'nan' value, got `{value}`")


T_STRING_ARRAY = TypeVar("T_STRING_ARRAY", bound="StringArrayBase")


class StringDType(DType["StringArrayBase"]):
    """String data type reflecting the semantics of [numpy.dtypes.StringDType][].

    If 'na_object' is provided it must be a 'nan' value or a sentinel string.
    """

    _na_object: _Nan | str | _NotSet
    _coerce: Literal[False]

    def __init__(
        self, na_object: float | str | _NotSet = _not_set, *, coerce: Literal[False]
    ):
        if coerce:
            # Safety first...
            raise ValueError(f"'coerce' must be 'False', got `{coerce}`")
        if isinstance(na_object, float | int):
            self._na_object = _Nan(na_object)
        else:
            self._na_object = na_object

        self._coerce = coerce

    def __repr__(self) -> str:
        if self._na_is_not_set:
            return f"{type(self).__name__}(coerce={self._coerce})"
        if self._na_is_nan:
            na_object: Any = np.nan
        else:
            na_object = repr(self._na_object)
        return f"{type(self).__name__}(na_object={na_object}, coerce={self._coerce})"

    def __ndx_create__(
        self,
        val: PyScalar | np.ndarray | TyArrayBase | Var | NestedSequence,
    ) -> StringArrayBase:
        if isinstance(val, TyArrayUtf8 | Var):
            mask = None
            data = TyArrayUtf8(val) if isinstance(val, Var) else val
            return self._try_build(data, mask)

        elif isinstance(val, np.ma.MaskedArray) or isinstance(val, TyArrayBase):
            pass
        else:
            arr, mask = _np_with_ndx_dtype(val, self)
            data_ = safe_cast(TyArrayUtf8, onnx.const(arr.astype(object)))
            mask_ = (
                None
                if mask is None
                else onnx.safe_cast(onnx.TyArrayBool, onnx.const(mask))
            )
            return self._try_build(data_, mask_)

        raise NotImplementedError

    def _try_build(
        self, data: onnx.TyArrayUtf8, mask: onnx.TyArrayBool | None
    ) -> StringArrayBase:
        """Try to build the array for the current data type.

        Raises a 'ValueError' if the the value of 'mask' and 'self' disagree.
        """
        if self._na_object == _not_set:
            if mask is None:
                return StringArrayNoMissing(data, dtype=self)
            raise ValueError(
                f"`{self}` signals that there is no missing data but 'mask' is not 'None'"
            )
        elif isinstance(self._na_object, str):
            if mask is None:
                return StringArraySentinel(data, dtype=self)
            raise ValueError(
                f"`{self}` signals that a sentinel is used for missing data but 'mask' is not 'None'"
            )
        elif isinstance(self._na_object, _Nan):
            return StringArrayNan(data, mask, coerce=self._coerce)
        raise ValueError(f"unexpected 'na_object' `{self._na_object}`")

    def __ndx_result_type__(self, other: DType | PyScalar) -> DType:
        return NotImplemented

    def __ndx_cast_from__(self, arr: TyArrayBase) -> StringArrayBase:
        """Convert the given array to this data type.

        This function is used to implement ``TyArrayBase.astype`` and
        should not be called directly.
        """
        if arr.dtype == self:
            return arr.copy()  # type: ignore
        if isinstance(arr, StringArrayNoMissing):
            return self._try_build(arr._data, None)
        if isinstance(arr, StringArrayNan):
            if self._na_is_not_set:
                return NotImplemented
            if isinstance(self._na_object, str):
                # _na_is_sentinel-branch, but in a mypy-compatible way
                data = arr._data.copy()
                mask = arr._mask
                sentinel = self._na_object
                if mask is not None:
                    data[mask] = onnx.const(sentinel)
                return self._try_build(data, None)
            # _na_is_nan
            return self._try_build(arr._data, arr._mask)
        if isinstance(arr, StringArraySentinel):
            if self._na_is_not_set:
                return NotImplemented
            if self._na_is_nan:
                mask = safe_cast(onnx.TyArrayBool, arr == arr.dtype._na_object)
                return self._try_build(arr._data, mask)
            if isinstance(self._na_object, str):
                mask = safe_cast(onnx.TyArrayBool, arr == arr.dtype._na_object)
                data = arr._data.copy()
                # update sentinels
                data[mask] = onnx.const(self._na_object)
                return self._try_build(
                    data, mask=None
                )  # mask=None since sentinels are inlined
        return NotImplemented

    def __ndx_argument__(self, shape: OnnxShape) -> StringArrayBase:
        raise NotImplementedError

    @property
    def __ndx_infov1__(self) -> DTypeInfoV1:
        raise NotImplementedError
        return DTypeInfoV1(author="ndonnx", type_name=self.__class__.__name__, meta=...)

    def unwrap_numpy(self) -> np.dtype:
        if isinstance(self._na_object, _Nan):
            return np.dtypes.StringDType(na_object=np.nan, coerce=self._coerce)
        if isinstance(self._na_object, _NotSet):
            return np.dtypes.StringDType(coerce=self._coerce)
        return np.dtypes.StringDType(na_object=self._na_object, coerce=self._coerce)

    @property
    def _na_is_nan(self) -> bool:
        return isinstance(self._na_object, _Nan)

    @property
    def _na_is_sentinel(self) -> bool:
        return isinstance(self._na_object, str)

    @property
    def _na_is_not_set(self) -> bool:
        return self._na_object == _not_set

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, StringDType):
            return False
        if self._na_is_nan and other._na_is_nan:
            return True
        return self._coerce == other._coerce and self._na_object == other._na_object


DTYPE = TypeVar("DTYPE", bound=StringDType)


class StringArrayBase(TyArrayBase, Generic[DTYPE]):
    _dtype: DTYPE
    _data: onnx.TyArrayUtf8

    @property
    def dtype(self) -> StringDType:
        return self._dtype

    @property
    def dynamic_shape(self) -> onnx.TyArrayInt64:
        return self._data.dynamic_shape

    @property
    def dynamic_size(self) -> onnx.TyArrayInt64:
        return self._data.dynamic_size

    @property
    def shape(self) -> OnnxShape:
        return self._data.shape


class StringArrayDataOnly(StringArrayBase[StringDType]):
    """Base class for String arrays that do not have a dedicated missing-mask.

    This may either be due to there being no missing data at all or because it simply
    uses a sentinel value.
    """

    @abstractmethod
    def __init__(self, data: onnx.TyArrayUtf8, *, dtype: StringDType): ...

    def __ndx_value_repr__(self) -> dict[str, str]:
        """A string representation of the fields to be used in ``Array.__repr__```."""
        reps = {}
        reps["data"] = self._data.__ndx_value_repr__()["data"]
        return reps

    @property
    def is_constant(self) -> bool:
        return self._data.is_constant

    def copy(self) -> Self:
        """Copy ``self`` including all component arrays."""
        return type(self)(data=self._data.copy(), dtype=self.dtype)

    def __getitem__(self, index: onnx.GetitemIndex) -> Self:
        data = self._data[index]
        return type(self)(data, dtype=self.dtype)

    def __ndx_cast_to__(
        self, dtype: DType[onnx.TY_ARRAY_BASE_co]
    ) -> onnx.TY_ARRAY_BASE_co:
        return NotImplemented

    def unwrap_numpy(self) -> np.ndarray:
        np_dtype = self.dtype.unwrap_numpy()
        data = self._data.unwrap_numpy().astype(np_dtype)
        return data

    def __eq__(self, other: TyArrayBase | PyScalar) -> onnx.TyArrayBool:  # type: ignore[override]
        if isinstance(other, str):
            return self._data == other
        if isinstance(other, StringArrayDataOnly):
            return self._data == other._data
        return NotImplemented

    def __setitem__(self, key: SetitemIndex, value: Self, /) -> None:
        self._data[key] = value._data

    def isnan(self) -> onnx.TyArrayBool:
        # NumPy does not take sentinel value into account!
        return onnx.const(False, onnx.bool_).broadcast_to(self.dynamic_shape)


class StringArrayNoMissing(StringArrayDataOnly):
    def __init__(self, data: onnx.TyArrayUtf8, *, dtype: StringDType):
        if not dtype._na_is_not_set:
            raise TypeError(
                f"expected unset 'na_object', got `{self.dtype._na_object}`"
            )
        self._dtype = dtype
        self._data = data.copy()


class StringArraySentinel(StringArrayDataOnly):
    def __init__(self, data: onnx.TyArrayUtf8, *, dtype: StringDType):
        if not isinstance(dtype._na_object, str):
            raise TypeError(
                f"expected string 'na_object', got `{self.dtype._na_object}`"
            )

        self._dtype = dtype
        self._data = data.copy()


class StringArrayNan(StringArrayBase[StringDType]):
    _mask: onnx.TyArrayBool | None

    def __init__(
        self,
        data: onnx.TyArrayUtf8,
        mask: onnx.TyArrayBool | None,
        *,
        coerce: Literal[False],
    ):
        self._dtype = StringDType(np.nan, coerce=coerce)
        self._data = data.copy()
        self._mask = mask if mask is None else mask.copy()

    def copy(self) -> Self:
        """Copy ``self`` including all component arrays."""
        mask = None if self._mask is None else self._mask.copy()
        return type(self)(data=self._data.copy(), mask=mask, coerce=self.dtype._coerce)

    def __ndx_value_repr__(self) -> dict[str, str]:
        """A string representation of the fields to be used in ``Array.__repr__```."""
        reps = {}
        reps["data"] = self._data.__ndx_value_repr__()["data"]
        reps["mask"] = (
            "None" if self._mask is None else self._mask.__ndx_value_repr__()["data"]
        )
        return reps

    @property
    def is_constant(self) -> bool:
        return self._data.is_constant and (
            True if self._mask is None else self._mask.is_constant
        )

    def __getitem__(self, index: onnx.GetitemIndex) -> Self:
        data = self._data[index]
        mask = None if self._mask is None else self._mask[index]

        return type(self)(data, mask, coerce=self.dtype._coerce)

    def __ndx_cast_to__(
        self, dtype: DType[onnx.TY_ARRAY_BASE_co]
    ) -> onnx.TY_ARRAY_BASE_co:
        return NotImplemented

    def unwrap_numpy(self) -> np.ndarray:
        np_dtype = self.dtype.unwrap_numpy()
        data = self._data.unwrap_numpy().astype(np_dtype)
        if self._mask is not None and self.dtype._na_is_nan:
            mask = self._mask.unwrap_numpy()
            data[mask] = np.nan
        return data

    def __eq__(self, other: TyArrayBase | PyScalar) -> onnx.TyArrayBool:  # type: ignore[override]
        mask = self._mask
        if isinstance(other, str):
            res = self._data == other
        elif isinstance(other, StringArrayNan):
            res = self._data == other._data
            mask = _merge_masks(mask, other._mask)
        else:
            return NotImplemented

        if mask is None:
            return res
        return res & ~mask

    def __setitem__(self, key: SetitemIndex, value: Self, /) -> None:
        self._data[key] = value._data
        if value._mask is not None:
            if self._mask is None:
                self._mask = onnx.const(False, onnx.bool_).broadcast_to(
                    self.dynamic_shape
                )
            self._mask[key] = value._mask

    def isnan(self) -> onnx.TyArrayBool:
        if self._mask is None:
            return onnx.const(False, onnx.bool_).broadcast_to(self.dynamic_shape)
        return self._mask.copy()


def _np_with_ndx_dtype(
    obj: PyScalar | NestedSequence | np.ndarray, dtype: StringDType
) -> tuple[np.ndarray, np.ndarray | None]:
    """Create a NumPy array from the given object.

    The returned data array either uses sentinels or non na-object at all.  I.e., the
    data is directly convertible into a utf8 array without further coercion.
    """
    arr = np.asarray(obj, dtype.unwrap_numpy(), copy=True)

    # Create mask for the nan-case only
    mask = None
    if isinstance(dtype._na_object, _Nan):
        mask = np.isnan(arr)
        if not np.any(mask):
            mask = None
        arr = arr.astype(np.dtypes.StringDType(na_object="NaN", coerce=False))

    return arr, mask


def from_numpy_dtype(dtype: np.dtype) -> StringDType:
    if not isinstance(dtype, np.dtypes.StringDType):
        # Use value error because that is what we also use in onnx.from_numpy
        raise ValueError(f"`{dtype}` does not have a corresponding ndonnx data type")
    if not hasattr(dtype, "na_object"):
        na_object = _not_set
    else:
        na_object = dtype.na_object
    if dtype.coerce:
        raise ValueError("'coerce=True' is not supported by ndonnx's 'StringDType'")
    return StringDType(na_object, coerce=dtype.coerce)


def _merge_masks(
    a: onnx.TyArrayBool | None, b: onnx.TyArrayBool | None
) -> onnx.TyArrayBool | None:
    """Merge two masks."""
    if a is None:
        return b
    if b is None:
        return a
    out = a | b
    if isinstance(out, onnx.TyArrayBool):
        return out
    # Should never happen
    raise TypeError("Unexpected array type")
