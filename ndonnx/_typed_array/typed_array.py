# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from types import NotImplementedType
from typing import TYPE_CHECKING, Literal, TypeVar, overload

import numpy as np
from typing_extensions import Self

from ndonnx import DType
from ndonnx.types import OnnxShape, PyScalar

from .ort_compat import const, if_

if TYPE_CHECKING:
    from spox import Var

    from .onnx import (
        KEY,
        VALUE,
        GetitemIndex,
        SetitemIndex,
        TyArrayBool,
        TyArrayInt64,
        TyArrayInteger,
    )

_Self_co = TypeVar("_Self_co", bound="TyArrayBase", covariant=True)
TY_ARRAY_BASE_co = TypeVar("TY_ARRAY_BASE_co", bound="TyArrayBase", covariant=True)


class TyArrayBase(ABC):
    @abstractmethod
    def __init__(self): ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.__dict__})"

    def __bool__(self, /) -> bool:
        return bool(self.unwrap_numpy())

    def __int__(self, /) -> int:
        return int(self.unwrap_numpy())

    def __float__(self, /) -> float:
        return float(self.unwrap_numpy())

    @abstractmethod
    def copy(self) -> Self:
        """Copy ``self`` including all component arrays."""

    @abstractmethod
    def __ndx_value_repr__(self) -> dict[str, str]:
        """A string representation of the fields to be used in ``Array.__repr__```."""
        # Note: It is unfortunate that this part of the API relies on
        # the rather useless `dict[str, str]` type hint. `TypedDict`
        # is not a viable solution (?) since it does not play nicely
        # with the subtyping.

    @property
    @abstractmethod
    def dtype(self: _Self_co) -> DType[_Self_co]: ...

    @property
    @abstractmethod
    def dynamic_shape(self) -> TyArrayInt64: ...

    @property
    @abstractmethod
    def shape(self) -> OnnxShape: ...

    @property
    @abstractmethod
    def is_constant(self) -> bool: ...

    @abstractmethod
    def __getitem__(self, index: GetitemIndex) -> Self: ...

    def __setitem__(
        self,
        key: SetitemIndex,
        value: Self,
        /,
    ) -> None:
        raise _make_type_error("__setitem__", self.dtype)

    def put(
        self,
        key: TyArrayInt64,
        value: Self,
        /,
    ) -> None:
        """Set elements with semantics identical to `numpy.put` with `mode="raise"."""
        raise _make_type_error("put", self.dtype)

    @property
    def mT(self) -> Self:  # noqa: N802
        raise _make_type_error("mT", self.dtype)

    def reshape(self, shape: tuple[int, ...] | TyArrayInt64) -> Self:
        raise _make_type_error("reshape", self.dtype)

    def squeeze(self, /, axis: int | tuple[int, ...]) -> Self:
        raise _make_type_error("squeeze", self.dtype)

    def disassemble(self) -> dict[str, Var] | Var:
        """Disassemble ``self`` into a flat mapping of its constituents."""
        raise _make_type_error("disassemble", self.dtype)

    def broadcast_to(self, shape: tuple[int, ...] | TyArrayInt64) -> Self:
        raise _make_type_error("broadcast_to", self.dtype)

    def concat(self, others: list[Self], axis: None | int) -> Self:
        raise _make_type_error("concat", self.dtype)

    def permute_dims(self, axes: tuple[int, ...]) -> Self:
        raise _make_type_error("permute_dims", self.dtype)

    def __ndx_cast_to__(self, dtype: DType[TY_ARRAY_BASE_co]) -> TY_ARRAY_BASE_co:
        """Reflective sibling method for `DType.__ndx_cast_from__` which must thus not
        call the latter.

        Use this function to implement the conversion from a custom type into a built-
        in one. This function is called by `TyArrayBase.astype`.
        """
        return NotImplemented

    def __ndx_equal__(
        self, other: TyArrayBase | PyScalar
    ) -> TyArrayBase | NotImplementedType:
        """Implementation of equal-comparison.

        '__eq__' has special semantics compared to other dunder methods.
        https://docs.python.org/3/reference/datamodel.html#object.__eq__
        """
        return NotImplemented

    @property
    def T(self) -> Self:  # noqa: N802
        if self.ndim != 2:
            raise ValueError(f"array must have two dimensions, found `{self.ndim}`")

        return self.mT

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def unwrap_numpy(self) -> np.ndarray:
        """Converter 'self' into a NumPy array.

        The conversion requires 'self' to be a constant that can be represented by a
        NumPy data type.

        Otherwise, a 'ValueError' is raised
        """
        raise ValueError(f"cannot convert '{self.__class__}' to NumPy array")

    def astype(
        self, dtype: DType[TY_ARRAY_BASE_co], /, *, copy=True
    ) -> TY_ARRAY_BASE_co:
        """Convert `self` to the `_TypedArray` associated with `dtype`."""
        if self.dtype == dtype:
            if copy:
                return self.copy()  # type: ignore
            else:
                return self  # type: ignore
        res = self.__ndx_cast_to__(dtype)
        if res is NotImplemented:
            # `type(self._data)` does not know about the target `dtype`
            res = dtype.__ndx_cast_from__(self)
        if res is not NotImplemented:
            return res
        raise ValueError(f"casting between `{self.dtype}` and `{dtype}` is undefined")

    def __iter__(self):
        try:
            n, *_ = self.shape
        except IndexError:
            raise ValueError("iteration over 0-d array")
        if isinstance(n, int):
            return (self[i, ...] for i in range(n))
        raise ValueError(
            "iteration requires dimension of static length, but dimension 0 is dynamic"
        )

    def moveaxis(
        self, source: int | tuple[int, ...], destination: int | tuple[int, ...], /
    ) -> Self:
        source = _normalize_axes_tuple(source, self.ndim)
        destination = _normalize_axes_tuple(destination, self.ndim)

        if source == destination:
            return self.copy()

        axes = [n for n in range(self.ndim) if n not in source]

        for dest, src in sorted(zip(destination, source)):
            axes.insert(dest, src)

        return self.permute_dims(axes=tuple(axes))

    def roll(
        self,
        shift: int | tuple[int, ...],
        *,
        axis: int | tuple[int, ...] | None = None,
    ) -> Self:
        x = self
        axis_ = axis
        if isinstance(shift, int):
            shift = (shift,)
        if axis_ is None:
            x = x.reshape((-1,))
            axis_ = 0
        axis_ = _normalize_axes_tuple(axis_, x.ndim)

        if len(shift) != len(axis_):
            raise ValueError("'shift' and 'axis' must be tuples of equal length")

        def _roll_axis(x: Self, shift: int, axis: int, /) -> Self:
            from .funcs import astyarray

            if shift == 0:
                return x

            indices_a = [slice(None) for i in range(x.ndim)]
            indices_b = [slice(None) for i in range(x.ndim)]

            dim = x.dynamic_shape[axis]
            (shift_,) = map(
                astyarray,
                if_(
                    (dim == 0).disassemble(),
                    then_branch=lambda: (const(0, dtype=np.int64),),
                    else_branch=lambda: ((astyarray(shift) % dim).disassemble(),),
                ),
            )
            # pre roll: |----a------|---b---|
            # postroll: |---b---|----a------|
            #           |-shift-|

            indices_a[axis] = slice(None, -shift_, 1)
            indices_b[axis] = slice(-shift_, None, 1)
            return x[tuple(indices_b)].concat([x[tuple(indices_a)]], axis=axis)

        for sh, ax in zip(shift, axis_):
            x = _roll_axis(x, sh, ax)

        if axis is None:
            return x.reshape(self.dynamic_shape)
        return x

    ################################################################
    # Member functions that reflect free functions of the standard #
    ################################################################

    def all(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> TyArrayBase:
        raise _make_type_error("all", self.dtype)

    def any(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> TyArrayBase:
        raise _make_type_error("any", self.dtype)

    def argmax(
        self, /, *, axis: int | None = None, keepdims: bool = False
    ) -> TyArrayInt64:
        raise _make_type_error("argmax", self.dtype)

    def argmin(
        self, /, *, axis: int | None = None, keepdims: bool = False
    ) -> TyArrayInt64:
        raise _make_type_error("argmin", self.dtype)

    def count_nonzero(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> TyArrayInt64:
        raise _make_type_error("count_nonzero", self.dtype)

    def argsort(
        self, /, *, axis: int = -1, descending: bool = False, stable: bool = True
    ) -> TyArrayInt64:
        raise _make_type_error("argsort", self.dtype)

    def cumulative_prod(
        self,
        /,
        *,
        axis: int | None = None,
        dtype: DType | None = None,
        include_initial: bool = False,
    ) -> TyArrayBase:
        raise _make_type_error("cumulative_prod", self.dtype)

    def cumulative_sum(
        self,
        /,
        *,
        axis: int | None = None,
        dtype: DType | None = None,
        include_initial: bool = False,
    ) -> TyArrayBase:
        raise _make_type_error("cumulative_sum", self.dtype)

    def clip(
        self, /, min: TyArrayBase | None = None, max: TyArrayBase | None = None
    ) -> Self:
        raise _make_type_error("clip", self.dtype)

    def diff(
        self,
        /,
        *,
        axis: int = -1,
        n: int = 1,
        prepend: Self | None = None,
        append: Self | None = None,
    ) -> Self:
        raise _make_type_error("diff", self.dtype)

    @overload
    def prod(
        self,
        /,
        *,
        dtype: DType[TY_ARRAY_BASE_co],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> TY_ARRAY_BASE_co: ...

    @overload
    def prod(
        self,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
    ) -> TyArrayBase: ...

    def prod(
        self,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
    ) -> TyArrayBase:
        raise _make_type_error("prod", self.dtype)

    def sort(
        self, /, *, axis: int = -1, descending: bool = False, stable: bool = True
    ) -> Self:
        raise _make_type_error("sort", self.dtype)

    def searchsorted(
        self,
        x2: Self,
        /,
        *,
        side: Literal["left", "right"] = "left",
        sorter: TyArrayInteger | None = None,
    ) -> TyArrayInt64:
        raise _make_type_error("searchsorted", self.dtype)

    @overload
    def sum(
        self,
        /,
        *,
        dtype: DType[TY_ARRAY_BASE_co],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> TY_ARRAY_BASE_co: ...

    @overload
    def sum(
        self,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
    ) -> TyArrayBase: ...

    def sum(
        self,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
    ) -> TyArrayBase:
        raise _make_type_error("sum", self.dtype)

    def unique_all(self) -> tuple[Self, TyArrayInt64, TyArrayInt64, TyArrayInt64]:
        raise _make_type_error("unique_all", self.dtype)

    def max(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Self:
        raise _make_type_error("max", self.dtype)

    def min(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Self:
        raise _make_type_error("min", self.dtype)

    def mean(
        self, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Self:
        raise _make_type_error("mean", self.dtype)

    def nonzero(self) -> tuple[TyArrayInt64, ...]:
        raise _make_type_error("nonzero", self.dtype)

    def reciprocal(self) -> TyArrayBase:
        raise _make_type_error("reciprocal", self.dtype)

    def take(self, indices: TyArrayInt64, /, *, axis: int | None = None) -> Self:
        raise _make_type_error("take", self.dtype)

    def take_along_axis(self, indices: TyArrayInt64, /, *, axis: int = -1) -> Self:
        raise _make_type_error("take_along_axis", self.dtype)

    def std(
        self,
        *,
        axis: int | tuple[int, ...] | None = None,
        correction: int | float = 0.0,
        keepdims: bool = False,
    ) -> Self:
        raise _make_type_error("std", self.dtype)

    def variance(
        self,
        *,
        axis: int | tuple[int, ...] | None = None,
        correction: int | float = 0.0,
        keepdims: bool = False,
    ) -> Self:
        raise _make_type_error("var", self.dtype)

    def repeat(
        self, repeats: int | TyArrayInt64, /, *, axis: int | None = None
    ) -> Self:
        raise _make_type_error("repeat", self.dtype)

    def tile(self, repetitions: tuple[int, ...], /) -> Self:
        raise _make_type_error("tile", self.dtype)

    # Element-wise functions without additional arguments

    def isnan(self) -> TyArrayBase:
        raise _make_type_error("isnan", self.dtype)

    def isfinite(self) -> TyArrayBase:
        raise _make_type_error("isfinite", self.dtype)

    def isinf(self) -> TyArrayBase:
        raise _make_type_error("isinf", self.dtype)

    def acos(self) -> Self:
        raise _make_type_error("acos", self.dtype)

    def acosh(self) -> Self:
        raise _make_type_error("acosh", self.dtype)

    def asin(self) -> Self:
        raise _make_type_error("asin", self.dtype)

    def asinh(self) -> Self:
        raise _make_type_error("asinh", self.dtype)

    def atan(self) -> Self:
        raise _make_type_error("atan", self.dtype)

    def atanh(self) -> Self:
        raise _make_type_error("atanh", self.dtype)

    def cos(self) -> Self:
        raise _make_type_error("cos", self.dtype)

    def cosh(self) -> Self:
        raise _make_type_error("cosh", self.dtype)

    def ceil(self) -> Self:
        raise _make_type_error("ceil", self.dtype)

    def exp(self) -> Self:
        raise _make_type_error("exp", self.dtype)

    def expm1(self) -> Self:
        raise _make_type_error("expm1", self.dtype)

    def floor(self) -> Self:
        raise _make_type_error("floor", self.dtype)

    def log(self) -> Self:
        raise _make_type_error("log", self.dtype)

    def log1p(self) -> Self:
        raise _make_type_error("log1p", self.dtype)

    def log2(self) -> Self:
        raise _make_type_error("log2", self.dtype)

    def log10(self) -> Self:
        raise _make_type_error("log10", self.dtype)

    def logical_not(self) -> Self:
        raise _make_type_error("logical_not", self.dtype)

    def round(self) -> Self:
        raise _make_type_error("round", self.dtype)

    def sign(self) -> Self:
        raise _make_type_error("sign", self.dtype)

    def signbit(self) -> Self:
        raise _make_type_error("signbit", self.dtype)

    def sin(self) -> Self:
        raise _make_type_error("sin", self.dtype)

    def sinh(self) -> Self:
        raise _make_type_error("sinh", self.dtype)

    def sqrt(self) -> Self:
        raise _make_type_error("sqrt", self.dtype)

    def square(self) -> TyArrayBase:
        raise _make_type_error("sqrt", self.dtype)

    def tan(self) -> Self:
        raise _make_type_error("tan", self.dtype)

    def tanh(self) -> Self:
        raise _make_type_error("tanh", self.dtype)

    def trunc(self) -> Self:
        raise _make_type_error("trunc", self.dtype)

    def tril(self, /, *, k: int = 0) -> Self:
        raise _make_type_error("trunc", self.dtype)

    def triu(self, /, *, k: int = 0) -> Self:
        raise _make_type_error("trunc", self.dtype)

    def __abs__(self) -> Self:
        raise _make_type_error("__abs__", self.dtype)

    def __neg__(self) -> TyArrayBase:
        raise _make_type_error("__neg__", self.dtype)

    def __pos__(self) -> TyArrayBase:
        raise _make_type_error("__pos__", self.dtype)

    def __invert__(self) -> Self:
        raise _make_type_error("__invert__", self.dtype)

    def __add__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __radd__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __and__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __floordiv__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __rfloordiv__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __ge__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __gt__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __le__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __lshift__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __lt__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __mod__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __rmod__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __mul__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __rmul__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __or__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __pow__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __rpow__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __rshift__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __sub__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __rsub__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __truediv__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __rtruediv__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __xor__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:
        return NotImplemented

    def __ne__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:  # type: ignore
        return ~(self == other)

    # mypy believes that __eq__ should return a `bool` but the docs say we can return whatever:
    # https://docs.python.org/3/reference/datamodel.html#object.__eq__
    def __eq__(self, other: TyArrayBase | PyScalar) -> TyArrayBase:  # type: ignore
        if not isinstance(other, TyArrayBase | PyScalar):
            return False

        res = self.__ndx_equal__(other)
        if res is NotImplemented and isinstance(other, TyArrayBase):
            res = other.__ndx_equal__(self)
        if res is NotImplemented:
            raise ValueError(
                f"comparison between `{type(self).__name__}` and `{type(other).__name__}` is not implemented"
            )
        return res

    # Functions which may return `NotImplemented`
    # Note: Prefixed with `__ndx_` to avoid naming collisions with
    # possible future Python dunder methods
    def __ndx_logaddexp__(self, rhs: TyArrayBase | int | float, /) -> TyArrayBase:
        return NotImplemented

    def __ndx_rlogaddexp__(self, lhs: TyArrayBase | int | float, /) -> TyArrayBase:
        return NotImplemented

    def __ndx_logical_and__(self, rhs: TyArrayBase | bool, /) -> TyArrayBase:
        return NotImplemented

    def __ndx_rlogical_and__(self, lhs: TyArrayBase | bool, /) -> TyArrayBase:
        return NotImplemented

    def __ndx_logical_or__(self, rhs: TyArrayBase | bool, /) -> TyArrayBase:
        return NotImplemented

    def __ndx_rlogical_or__(self, lhs: TyArrayBase | bool, /) -> TyArrayBase:
        return NotImplemented

    def __ndx_logical_xor__(self, rhs: TyArrayBase | bool, /) -> TyArrayBase:
        return NotImplemented

    def __ndx_rlogical_xor__(self, lhs: TyArrayBase | bool, /) -> TyArrayBase:
        return NotImplemented

    def __ndx_maximum__(
        self, other: TyArrayBase | PyScalar, /
    ) -> TyArrayBase | NotImplementedType:
        return NotImplemented

    def __ndx_minimum__(
        self, other: TyArrayBase | PyScalar, /
    ) -> TyArrayBase | NotImplementedType:
        return NotImplemented

    def __ndx_tensordot__(
        self,
        other: TyArrayBase,
        /,
        *,
        axes: int | tuple[Sequence[int], Sequence[int]] = 2,
    ) -> TyArrayBase | NotImplementedType:
        return NotImplemented

    def __ndx_where__(
        self, cond: TyArrayBool, y: TyArrayBase | PyScalar, /
    ) -> TyArrayBase | NotImplementedType:
        return NotImplemented

    def __ndx_rwhere__(
        self, cond: TyArrayBool, y: TyArrayBase | PyScalar, /
    ) -> TyArrayBase | NotImplementedType:
        return NotImplemented

    ######################################################
    # Non-standard functions that reflect free functions #
    ######################################################

    def isin(self, items: Sequence[VALUE], /) -> TyArrayBool:
        """Return for each element in ``self`` that is found in ``items``.

        ``NaN`` values do **not** compare equal.

        Parameters
        ----------
        x
            The input Array to check for the presence of items.
        items
            Scalar items to check for in the input Array.

        Returns
        -------
        out: Array
            Array of booleans indicating whether each element of ``x`` is in ``items``.
        """
        raise _make_type_error("isin", self.dtype)

    def apply_mapping(
        self, mapping: Mapping[KEY, VALUE], default: VALUE
    ) -> TyArrayBase:
        """Map values in ``self`` based on the static ``mapping``.

        Parameters
        ----------
        mapping
            A mapping from keys to values. The keys must be of the
            same type as the values in ``x``. NaN-keys compare true to
            values in ``self``.
        default
            The default value to use when a key is not found in the mapping.

        Returns
        -------
        A new Array with the values mapped according to the mapping.
        """
        raise _make_type_error("apply_mapping", self.dtype)


def _make_type_error(fn_name, dtype: DType) -> TypeError:
    return TypeError(f"'{fn_name}' is not implemented for data type `{dtype}`")


def _normalize_axes_tuple(axes: int | tuple[int, ...], rank: int) -> tuple[int, ...]:
    if isinstance(axes, int):
        axes = (axes,)

    return tuple(el if el >= 0 else rank + el for el in axes)
