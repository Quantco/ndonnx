# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

import ndonnx as ndx
import ndonnx._data_types as dtypes
import ndonnx._opset_extensions as opx
import ndonnx.additional as nda

from ._interface import OperationsBlock
from ._utils import assemble_output_recurse, from_corearray

if TYPE_CHECKING:
    from ndonnx._array import Array, IndexType
    from ndonnx._data_types import Dtype


class UniformShapeOperations(OperationsBlock):
    """Provides implementation for shape/indexing operations that are generic across all
    data types where the array's shape is uniform across all of its constituent
    fields."""

    def shape(self, x):
        current = x
        while isinstance(current, ndx.Array):
            current = next(iter(current._fields.values()))
        return from_corearray(opx.shape(current))

    def static_shape(self, x) -> tuple[int | None, ...]:
        current = x
        while isinstance(current, ndx.Array):
            current = next(iter(current._fields.values()))

        inferred_shape = current.var.unwrap_tensor().shape
        if inferred_shape is None:
            raise ValueError(
                f"Could not determine static shape for array {x}. Dynamic reshapes cannot be resolved without input values."
            )
        return tuple(
            None if not isinstance(dim, int) else dim for dim in inferred_shape
        )

    def take(self, x, indices, axis=None):
        if axis is None:
            axis = 0
        if isinstance(indices, ndx.Array):
            indices = indices._core()
        else:
            indices = opx.const(indices, dtype=dtypes.int64)
        return x._transmute(lambda corearray: opx.gather(corearray, indices, axis=axis))

    def broadcast_to(self, x, shape):
        shape = ndx.asarray(shape, dtype=dtypes.int64)._core()
        return x._transmute(lambda corearray: opx.expand(corearray, shape))

    def expand_dims(self, x, axis):
        return x._transmute(
            lambda corearray: opx.unsqueeze(
                corearray, axes=opx.const([axis], dtype=dtypes.int64)
            )
        )

    def flip(self, x, axis):
        if x.ndim == 0:
            return x.copy()

        if axis is None:
            axis = range(x.ndim)
        elif not isinstance(axis, Iterable):
            axis = [axis]

        axis = [x.ndim + ax if ax < 0 else ax for ax in axis]

        index = tuple(
            slice(None, None, None) if i not in axis else (slice(None, None, -1))
            for i in range(0, x.ndim)
        )

        return x[index]

    def permute_dims(self, x, axes):
        return x._transmute(lambda corearray: opx.transpose(corearray, perm=axes))

    def reshape(self, x, shape, *, copy=None):
        if (
            isinstance(shape, (list, tuple))
            and len(shape) == x.ndim == 1
            and shape[0] == -1
        ):
            return x.copy()
        else:
            x = ndx.asarray(x)
            shape = ndx.asarray(shape, dtype=dtypes.int64)._core()
            out = x._transmute(lambda corearray: opx.reshape(corearray, shape))
            if copy is False:
                return x._set(out)
            return out

    def roll(self, x, shift, axis):
        if not isinstance(shift, Sequence):
            shift = [shift]

        old_shape = nda.shape(x)

        if axis is None:
            x = ndx.reshape(x, [-1])
            axis = 0

        if not isinstance(axis, Sequence):
            axis = [axis]

        if len(shift) != len(axis):
            raise ValueError("shift and axis must have the same length")

        for sh, ax in zip(shift, axis):
            len_single = opx.gather(nda.shape(x)._core(), opx.const(ax))
            shift_single = opx.add(opx.const(-sh, dtype=dtypes.int64), len_single)
            # Find the needed element index and then gather from it
            range = opx.cast(
                opx.range(
                    opx.const(0, dtype=len_single.dtype),
                    len_single,
                    opx.const(1, dtype=len_single.dtype),
                ),
                to=dtypes.int64,
            )
            new_indices = opx.mod(opx.add(range, shift_single), len_single)
            x = ndx.take(x, from_corearray(new_indices), axis=ax)

        return ndx.reshape(x, old_shape)

    def squeeze(self, x, axis):
        axis = [axis] if not isinstance(axis, Sequence) else axis
        if len(axis) == 0:
            return x.copy()
        axes = opx.const(axis, dtype=dtypes.int64)
        return x._transmute(lambda corearray: opx.squeeze(corearray, axes=axes))

    # creation.py
    def full(self, shape, fill_value, dtype=None, device=None):
        fill_value = ndx.asarray(fill_value)
        if fill_value.ndim != 0:
            raise ValueError("fill_value must be a scalar")

        dtype = fill_value.dtype if dtype is None else dtype
        shape = ndx.asarray(shape, dtype=dtypes.int64)
        if shape.ndim == 0:
            shape = ndx.reshape(shape, [1])
        return fill_value._transmute(
            lambda corearray: opx.expand(corearray, shape._core())
        ).astype(dtype)

    def full_like(self, x, fill_value, dtype=None, device=None):
        fill_value = ndx.asarray(fill_value).astype(dtype or x.dtype)
        return ndx.broadcast_to(fill_value, nda.shape(x))

    def where(self, condition, x, y):
        if x.dtype != y.dtype:
            return NotImplemented
        if isinstance(condition.dtype, dtypes.Nullable) and not isinstance(
            x.dtype, (dtypes.Nullable, dtypes.CoreType)
        ):
            raise TypeError("where condition is nullable, but both outputs are not")
        if condition.to_numpy() is not None and condition.dtype == dtypes.bool:
            if (
                condition.to_numpy().size == 1
                and condition.to_numpy().ndim <= x.ndim
                and condition.to_numpy().item()
            ):
                return x.copy()
            elif (
                condition.to_numpy().size == 1
                and condition.to_numpy().ndim <= y.ndim
                and not condition.to_numpy().item()
            ):
                return y.copy()
        if x.dtype == y.dtype and x.to_numpy() is not None and y.to_numpy() is not None:
            if isinstance(x.to_numpy(), np.ma.MaskedArray):
                if np.ma.allequal(x.to_numpy(), y.to_numpy(), fill_value=False):
                    return ndx.asarray(
                        np.broadcast_arrays(x.to_numpy(), y.to_numpy(), subok=True)[0]
                    )
            else:
                cond = np.equal(x.to_numpy(), y.to_numpy())
                if isinstance(x.dtype, ndx.Floating):
                    cond |= np.isnan(x.to_numpy()) & np.isnan(y.to_numpy())
                if cond.all():
                    return ndx.asarray(
                        np.broadcast_arrays(x.to_numpy(), y.to_numpy())[0]
                    )
        condition_values = (
            condition.values if condition.dtype == ndx.nbool else condition
        )

        # generic layout operation, work over both arrays with same data type to apply the condition
        def where_dtype_agnostic(a: ndx.Array, b: ndx.Array):
            if a.dtype != b.dtype:
                raise ValueError("The dtype of both arrays must be the same")
            if a.dtype == dtypes.bool:
                return ndx.logical_xor(
                    ndx.logical_and(condition_values, a),
                    ndx.logical_and(~condition_values, b),
                )
            elif a.dtype in (dtypes.int8, dtypes.int16):
                return from_corearray(
                    opx.where(
                        condition_values._core(),
                        a.astype(dtypes.int32)._core(),
                        b.astype(dtypes.int32)._core(),
                    )
                ).astype(a.dtype)
            elif a.dtype in (dtypes.uint16, dtypes.uint32, dtypes.uint64):
                return from_corearray(
                    opx.where(
                        condition_values._core(),
                        a.astype(dtypes.int64)._core(),
                        b.astype(dtypes.int64)._core(),
                    )
                ).astype(a.dtype)
            elif isinstance(a.dtype, dtypes.CoreType):
                return from_corearray(
                    opx.where(
                        condition_values._core(),
                        a._core(),
                        b._core(),
                    )
                )
            else:
                fields = {}
                for name in a.dtype._fields():
                    fields[name] = where_dtype_agnostic(
                        getattr(a, name), getattr(b, name)
                    )
                return ndx.Array._from_fields(a.dtype, **fields)

        output = where_dtype_agnostic(x, y)
        if condition.dtype == ndx.nbool:
            # propagate null if present
            if not isinstance(output.dtype, dtypes.Nullable):
                output = ndx.additional.make_nullable(output, condition.null)
            else:
                output.null = output.null | condition.null
        return output

    def zeros_like(self, x, dtype=None, device=None):
        return ndx.zeros(nda.shape(x), dtype=dtype or x.dtype, device=device)

    def ones_like(self, x, dtype=None, device=None):
        return ndx.ones(nda.shape(x), dtype=dtype or x.dtype, device=device)

    def make_array(
        self,
        shape: tuple[int | None | str, ...],
        dtype: Dtype,
        eager_value: np.ndarray | None = None,
    ) -> Array:
        if isinstance(dtype, dtypes.CoreType):
            return NotImplemented

        fields: dict[str, ndx.Array] = {}

        eager_values = None if eager_value is None else dtype._parse_input(eager_value)
        for name, field_dtype in dtype._fields().items():
            if eager_values is None:
                field_value = None
            else:
                field_value = assemble_output_recurse(field_dtype, eager_values[name])
            fields[name] = field_dtype._ops.make_array(
                shape,
                field_dtype,
                field_value,
            )
        return ndx.Array._from_fields(
            dtype,
            **fields,
        )

    def getitem(self, x: Array, index: IndexType) -> Array:
        if isinstance(index, ndx.Array) and not (
            isinstance(index.dtype, dtypes.Integral) or index.dtype == dtypes.bool
        ):
            raise TypeError(
                f"Index must be an integral or boolean 'Array', not `{index.dtype}`"
            )

        if isinstance(index, ndx.Array):
            index = index._core()

        return x._transmute(lambda corearray: corearray[index])
