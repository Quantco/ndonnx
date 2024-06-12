# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import typing
from collections.abc import Callable
from typing import Union

import numpy as np
import spox.opset.ai.onnx.v19 as op
from spox import Tensor, Var, argument
from typing_extensions import Self

import ndonnx as ndx
import ndonnx._data_types as dtypes
from ndonnx.additional import shape

from ._corearray import _CoreArray
from ._index import ScalarIndexType
from ._utility import get_shape

if typing.TYPE_CHECKING:
    from ndonnx._data_types.coretype import CoreType
    from ndonnx._data_types.structtype import StructType

IndexType = Union[ScalarIndexType, tuple[ScalarIndexType, ...], "Array"]


def array(
    *,
    shape: tuple[int | str | None, ...],
    dtype: CoreType | StructType,
) -> Array:
    """Creates a new lazy ndonnx array. This is used to define inputs to an ONNX model.

    Parameters
    ----------
    shape : tuple[int | str | None, ...]
        The shape of the array. str dimensions denote symbolic dimensions and must be globally consistent. None dimensions denote unknown dimensions.
    dtype : CoreType | StructType
        The data type of the array.

    Returns
    -------
    out : Array
        The new array. This represents an ONNX model input.
    """
    return Array._construct(shape=shape, dtype=dtype)


def from_spox_var(
    var: Var,
) -> Array:
    """Construct an ``Array`` from a ``spox.Var``. This function is useful when
    interoperating between ``ndonnx`` and ``spox``.

    Parameters
    ----------
    var : spox.Var
        The variable to construct the array from.

    Returns
    -------
    out : Array
        The constructed array.
    """
    corearray = _CoreArray(var)
    return _from_corearray(corearray)


def _from_corearray(
    corearray: _CoreArray,
) -> Array:
    return Array._from_fields(corearray.dtype, data=corearray)


class Array:
    dtype: dtypes.StructType | dtypes.CoreType
    _fields: dict[str, Array | _CoreArray]

    def __init__(self, *args, **kwargs) -> None:
        raise TypeError("Array cannot be instantiated directly.")

    @classmethod
    def _from_fields(
        cls, dtype: StructType | CoreType, **fields: Array | _CoreArray
    ) -> Array:
        instance = cls.__new__(cls)
        instance.dtype = dtype
        if isinstance(dtype, dtypes.CoreType) and tuple(fields.keys()) != ("data",):
            raise ValueError(f"Expected a single field 'data' for CoreType {dtype}")
        if isinstance(dtype, dtypes.StructType) and set(fields.keys()) != set(
            dtype._fields()
        ):
            raise ValueError(
                f"Expected fields {dtype._fields()} for StructType {dtype}, received {fields}"
            )
        instance._fields = fields
        for name, field in fields.items():
            setattr(instance, name, field)
        return instance

    def __getattr__(self, name: str) -> Array:
        field = self._fields.get(name, None)
        if isinstance(field, Array):
            return field
        else:
            raise AttributeError(f"Field {name} not found")

    def _set(self, other: Array) -> Array:
        self.dtype = other.dtype
        self._fields = other._fields
        for name, field in other._fields.items():
            setattr(self, name, self._fields[name]._set(field))  # type: ignore
        return self

    def copy(self) -> Array:
        """Clones the array, copying all the array fields.

        Returns
        -------
        out : Array
            The cloned array.
        """
        return Array._from_fields(
            self.dtype,
            **{name: field.copy() for name, field in self._fields.items()},
        )

    def to_numpy(self) -> np.ndarray | None:
        """Returns the value of the Array as a NumPy array if available, otherwise
        ``None``."""
        field_values = {}
        for name, field in self._fields.items():
            if (eager_value := field.to_numpy()) is None:
                return None
            field_values[name] = eager_value
        return self.dtype._assemble_output(field_values)

    def astype(self, to: CoreType | StructType) -> Array:
        return ndx.astype(self, to)

    def __getitem__(self, index: IndexType) -> Array:
        if isinstance(index, Array) and not (
            isinstance(index.dtype, dtypes.Integral) or index.dtype == dtypes.bool
        ):
            raise TypeError(
                f"Index must be an integral or boolean 'Array', not `{index.dtype}`"
            )

        if isinstance(index, Array):
            index = index._core()

        return self._transmute(lambda corearray: corearray[index])

    def __setitem__(
        self, index: IndexType | Self, updates: int | bool | float | Array
    ) -> Array:
        if (
            isinstance(index, Array)
            and not isinstance(index.dtype, dtypes.Integral)
            and index.dtype != dtypes.bool
        ):
            raise TypeError(
                f"Index must be an integral or boolean 'Array', not `{index.dtype}`"
            )

        updates = ndx.asarray(updates).astype(self.dtype)

        for name, field in self._fields.items():
            if isinstance(field, _CoreArray):
                field[index._core() if isinstance(index, Array) else index] = (
                    updates._core()
                )
            else:
                field[index] = getattr(updates, name)

        return self

    def __repr__(self) -> str:
        eager_value = self.to_numpy()
        prefix = (
            f"Array({eager_value}, dtype={self.dtype})"
            if eager_value is not None
            else f"Array(dtype={self.dtype})"
        )
        return f"{prefix}"

    @staticmethod
    def _construct(
        shape: tuple[int | str | None, ...],
        dtype: CoreType | StructType,
        eager_values: dict | None = None,
    ) -> Array:
        if isinstance(dtype, dtypes.CoreType):
            var = (
                op.const(eager_values["data"], dtype=dtype.to_numpy_dtype())
                if eager_values is not None
                else argument(Tensor(dtype.to_numpy_dtype(), shape))
            )
            return Array._from_fields(
                dtype,
                data=_CoreArray(
                    eager_values["data"] if eager_values is not None else var
                ),
            )
        else:
            fields = {}
            for name, field_dtype in dtype._fields().items():
                fields[name] = Array._construct(
                    shape,
                    field_dtype,
                    eager_values[name] if eager_values is not None else None,
                )
            return Array._from_fields(
                dtype,
                **fields,
            )

    @property
    def ndim(self) -> int:
        """The rank of the array.

        This can be statically determined even for lazy arrays.
        """
        return len(self._static_shape)

    @property
    def shape(self) -> Array | tuple[int, ...]:
        """The shape of the array.

        Returns:
            Array | tuple[int, ...]: The shape of the array.

        Note that the shape of the array is a tuple of integers when the "eager value"
        of the array can be determined. This is in strict compliance with the Array API standard which
        presupposes tuples in compliance tests.

        When lazy arrays are involved, the shape is data-dependent (or runtime inputs dependent) and so we
        cannot reliably determine the shape. In such cases, an integer ``ndx.Array`` is returned instead.
        """
        # The Array API standard expects that the shape has type tuple[int | None, ...], as do compliance tests.
        # However, when the array is completely lazy, a concrete shape may not be determinable.
        # We therefore provide an int Array which is provisioned for in the standard.
        # See https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.shape.html#shape for more context.
        eager_value = self.to_numpy()
        if eager_value is not None:
            return eager_value.shape
        else:
            return shape(self)

    @property
    def values(self) -> Array:
        """Accessor for data in a ``Array`` with nullable datatype."""
        if isinstance(self.dtype, dtypes.Nullable):
            return self.__getattr__("values")
        else:
            raise AttributeError("Field 'values' does not exist")

    @values.setter
    def values(self, value: Array) -> None:
        if (
            isinstance(self.dtype, dtypes.Nullable)
            and value.dtype == self.dtype._fields()["values"]
        ):
            self._fields["values"] = value
        else:
            raise ValueError("Field 'values' does not exist")

    @property
    def null(self) -> Array:
        """Accessor for missing null mask in a ``Array`` with nullable datatype."""
        if isinstance(self.dtype, dtypes.Nullable):
            return self.__getattr__("null")
        else:
            raise AttributeError("Field 'null' does not exist")

    @null.setter
    def null(self, value: Array) -> None:
        if isinstance(self.dtype, dtypes.Nullable) and value.dtype == dtypes.bool:
            self._fields["null"] = value
        else:
            raise ValueError("Field 'null' does not exist")

    @property
    def _static_shape(self) -> tuple[int | str | None, ...]:
        """Inferred shape of the array based on static ONNX shape propagation."""
        value = self.to_numpy()
        if value is not None:
            return value.shape
        else:
            current: _CoreArray | Array = self
            while not isinstance(current, _CoreArray):
                current = next(iter(current._fields.values()))
            return get_shape(current)

    def _transmute(self, op: Callable[[_CoreArray], _CoreArray]) -> Array:
        """Apply a layout transformation to the array (transmute the array).

        Data types must be preserved. This function will validate this and throw a ValueError if `op` breaks this contract.
        """
        fields: dict[str, _CoreArray | Array] = {}
        for name, field in self._fields.items():
            if isinstance(field, _CoreArray):
                fields[name] = op(field)
            else:
                fields[name] = field._transmute(op)
            if fields[name].dtype != field.dtype:
                raise ValueError(
                    f"Transmute operation changed the dtype of field {name} from {field.dtype} to {fields[name].dtype}. Only layout transformations are permitted."
                )
        return Array._from_fields(self.dtype, **fields)

    def _core(self) -> _CoreArray:
        if isinstance(self.dtype, dtypes.CoreType):
            return typing.cast(_CoreArray, self._fields["data"])
        raise ValueError("Only CoreType arrays have a core array")

    def spox_var(self) -> Var:
        """Return the underlying Spox ``Var``. This is useful when interoperating
        between ``Spox`` and ``ndonnx``.

        Raises
        ------
        TypeError
            If the data type of the ``Array`` is not a ``CoreType``, there is not a singular ``Var`` associated with it.
        """
        return self._core().var

    @staticmethod
    def __array_namespace__():
        return ndx

    def __float__(self) -> float:
        eager_value = self.to_numpy()
        if eager_value is not None and eager_value.size == 1:
            return float(eager_value)
        else:
            raise ValueError(f"Cannot convert Array of shape {self.shape} to a float")

    def __index__(self) -> int:
        eager_value = self.to_numpy()
        if (
            self.ndim == 0
            and eager_value is not None
            and isinstance(eager_value.item(), int)
        ):
            return int(eager_value)
        else:
            raise ValueError(f"Cannot convert Array {self} to an index (int)")

    def __int__(self) -> int:
        eager_value = self.to_numpy()
        if eager_value is not None and eager_value.size == 1:
            return int(eager_value)
        else:
            raise ValueError(f"Cannot convert Array of shape {self.shape} to an int")

    def __bool__(self) -> bool:
        eager_value = self.to_numpy()
        if eager_value is not None and eager_value.size == 1:
            return bool(self.to_numpy())
        else:
            raise ValueError(f"Cannot convert Array of shape {self.shape} to a bool")

    def __len__(self) -> int:
        if isinstance(eager_value := self.shape, tuple):
            return eager_value[0]
        else:
            raise ValueError(f"Cannot convert Array of shape {self.shape} to a length")

    def __pos__(self):
        return ndx.positive(self)

    def __neg__(self):
        return ndx.negative(self)

    def __abs__(self):
        return ndx.abs(self)

    def __add__(self, other):
        return ndx.add(self, other)

    def __radd__(self, other):
        return ndx.add(other, self)

    def __iadd__(self, other):
        return self._set(ndx.add(self, other))

    def __sub__(self, other):
        return ndx.subtract(self, other)

    def __rsub__(self, other):
        return ndx.subtract(other, self)

    def __isub__(self, other):
        return self._set(ndx.subtract(self, other))

    def __mul__(self, other):
        return ndx.multiply(self, other)

    def __rmul__(self, other):
        return ndx.multiply(other, self)

    def __imul__(self, other):
        return self._set(ndx.multiply(self, other))

    def __truediv__(self, other):
        return ndx.divide(self, other)

    def __rtruediv__(self, other):
        return ndx.divide(other, self)

    def __floordiv__(self, other):
        return ndx.floor_divide(self, other)

    def __rfloordiv__(self, other):
        return ndx.floor_divide(other, self)

    def __mod__(self, other):
        return ndx.remainder(self, other)

    def __rmod__(self, other):
        return ndx.remainder(other, self)

    def __pow__(self, other):
        return ndx.pow(self, other)

    def __rpow__(self, other):
        return ndx.pow(other, self)

    def __matmul__(self, other):
        return ndx.matmul(self, other)

    def __lt__(self, other):
        return ndx.less(self, other)

    def __le__(self, other):
        return ndx.less_equal(self, other)

    def __gt__(self, other):
        return ndx.greater(self, other)

    def __ge__(self, other):
        return ndx.greater_equal(self, other)

    def __invert__(self):
        return ndx.bitwise_invert(self)

    def __and__(self, other):
        return ndx.bitwise_and(self, other)

    def __or__(self, other):
        return ndx.bitwise_or(self, other)

    def __xor__(self, other):
        return ndx.bitwise_xor(self, other)

    def __lshift__(self, other):
        return ndx.bitwise_left_shift(self, other)

    def __rshift__(self, other):
        return ndx.bitwise_right_shift(self, other)

    def __eq__(self, other):
        if isinstance(other, type(Ellipsis)):
            # FIXME: What is going on here? Deserves a comment, IMHO
            return ndx.asarray(False)
        return ndx.equal(self, other)

    def __ne__(self, other):
        return ndx.not_equal(self, other)

    @property
    def mT(self) -> ndx.Array:  # noqa: N802
        """Transpose of a matrix (or a stack of matrices).

        If an array instance has fewer than two dimensions, an error should be raised.

        Returns
        -------
        out: Array
            array whose last two dimensions (axes) are permuted in reverse order relative to original array (i.e., for an array instance having shape ``(..., M, N)``, the returned array must have shape ``(..., N, M)``). The returned array must have the same data type as the original array.
        """
        return ndx.matrix_transpose(self)

    @property
    def size(self) -> ndx.Array:
        """Number of elements in the array.

        Returns
        -------
        out: Array
            Scalar ``Array`` instance whose value is the number of elements in the original array.
        """
        return ndx.prod(self.shape)

    @property
    def T(self) -> ndx.Array:  # noqa: N802
        """Transpose of the array.

        The array instance must be two-dimensional.

        Raises
        ------
        TypeError
            If the array instance is not two-dimensional.

        Returns
        -------
        out: Array
            Two-dimensional array whose first and last dimensions (axes) are permuted in reverse order relative to original array. The returned array must have the same data type as the original array.
        """
        if self.ndim != 2:
            raise TypeError("Cannot transpose Array of rank != 2")

        return ndx.matrix_transpose(self)

    def len(self):
        """Returns the length of the array as an array.

        The array instance must be one-dimensional.

        Raises
        ------
        TypeError
            If the array instance is not one-dimensional.

        Returns
        -------
        out: Array
            Singleton array whose value is the length of the original array.
        """
        if self.ndim != 1:
            raise TypeError("Cannot call len on Array of rank != 1")
        return ndx.asarray(self.shape[0])

    def sum(self, axis: int | None = 0, keepdims: bool | None = False) -> ndx.Array:
        """See :py:func:`ndonnx.sum` for documentation."""
        return ndx.sum(self, axis=axis, keepdims=False)

    def prod(self, axis: int | None = 0, keepdims: bool | None = False) -> ndx.Array:
        """See :py:func:`ndonnx.prod` for documentation."""
        return ndx.prod(self, axis=axis, keepdims=False)

    def max(self, axis: int | None = 0, keepdims: bool | None = False) -> ndx.Array:
        """See :py:func:`ndonnx.max` for documentation."""
        return ndx.max(self, axis=axis, keepdims=False)

    def min(self, axis: int | None = 0, keepdims: bool | None = False) -> ndx.Array:
        """See :py:func:`ndonnx.min` for documentation."""
        return ndx.min(self, axis=axis, keepdims=False)

    def all(self, axis: int | None = 0, keepdims: bool | None = False) -> ndx.Array:
        """See :py:func:`ndonnx.all` for documentation."""
        return ndx.all(self, axis=axis, keepdims=False)

    def any(self, axis: int | None = 0, keepdims: bool | None = False) -> ndx.Array:
        """See :py:func:`ndonnx.any` for documentation."""
        return ndx.any(self, axis=axis, keepdims=False)


__all__ = [
    "Array",
    "array",
]
