# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from types import EllipsisType
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from . import onnx, ort_compat, safe_cast

if TYPE_CHECKING:
    from .onnx import TyArrayBool, TyArrayInt64, TyArrayInteger

    ScalarInt = TyArrayInteger
    """Alias signaling that this must be a rank-0 integer tensor."""
    BoolMask = TyArrayBool
    """Alias signaling that this must be a rank-1 boolean tensor."""
    SetitemItem: TypeAlias = int | slice | EllipsisType | ScalarInt
    """A single item; i.e. not a tuple nor a boolean mask.

    This does not include `None`.
    """
    GetitemItem: TypeAlias = int | slice | EllipsisType | ScalarInt | None
    """A single item (; i.e. not a tuple nor a boolean mask) for __getitem__.

    This includes `None`.
    """
    SetitemIndex: TypeAlias = SetitemItem | tuple[SetitemItem, ...] | BoolMask
    GetitemIndex: TypeAlias = GetitemItem | tuple[GetitemItem, ...] | BoolMask


class FancySlice:
    """Normalized "slice" object which may also represent a single-item index.

    In the latter case, the result needs to be squeezed after the indexing operation.
    """

    # 1D arrays with a single element so that concatenating them later is cheaper
    start: TyArrayInt64
    stop: TyArrayInt64
    step: TyArrayInt64
    squeeze: bool

    def __init__(self, obj: slice | int | TyArrayInt64):
        if isinstance(obj, onnx.TyArrayInt64 | int):
            self.start = _asidx(obj)
            self.stop = _asidx(_compute_end_single_idx(obj))
            self.step = _asidx(1)
            self.squeeze = True
            return

        # `slice` case
        def validate(obj: Any) -> TyArrayInt64 | None:
            if obj is None:
                return None
            if isinstance(obj, int | onnx.TyArrayInt64):
                return _asidx(obj)
            raise TypeError(f"unsupported type as slice parameter `{type(obj)}`")

        step = validate(obj.step)
        start = _compute_start_slice(validate(obj.start), step)
        stop = _compute_stop_slice(validate(obj.stop), step)
        self.step = _asidx(1) if step is None else step
        self.start = _asidx(start)
        self.stop = _asidx(stop)
        self.squeeze = False

    def is_noop(self) -> bool:
        try:
            start = self.start.unwrap_numpy()
            stop = self.stop.unwrap_numpy()
            step = self.step.unwrap_numpy()
        except ValueError:
            return False
        return (start, stop, step) in [(0, _MAX, 1), (0, _MIN, -1)]


def _asidx(obj: int | TyArrayInt64) -> TyArrayInt64:
    arr = onnx.const(obj, onnx.int64) if isinstance(obj, int) else obj

    if arr.ndim != 0:
        raise IndexError(f"expected scalar array but found array of rank `{arr.ndim}`")

    return arr


_MAX = np.iinfo(np.int64).max
_MIN = np.iinfo(np.int64).min


def _compute_end_single_idx(start: TyArrayInt64 | int) -> TyArrayInt64 | int:
    if isinstance(start, int):
        start = start
        end = start + 1
        end_is_zero = end == 0
        return _MAX if end_is_zero else end
    end_ = start + 1
    end_is_zero_ = end_ == 0
    return onnx.where(end_is_zero_, _asidx(_MAX), end_)


def _compute_start_slice(
    start: int | TyArrayInt64 | None, step: int | TyArrayInt64 | None
) -> TyArrayInt64 | int:
    if start is not None:
        return start

    if step is None:
        step = 1

    if isinstance(step, int):
        return 0 if step > 0 else _MAX

    step_gt_zero = step > 0
    return onnx.where(
        step_gt_zero, onnx.const(0, onnx.int64), onnx.const(_MAX, onnx.int64)
    )


def _compute_stop_slice(
    stop: int | TyArrayInt64 | None, step: int | TyArrayInt64 | None
) -> TyArrayInt64 | int:
    if stop is not None:
        return stop
    if step is None:
        step = 1

    is_reverse = step < 1
    if isinstance(is_reverse, bool):
        return _MIN if is_reverse else _MAX
    return onnx.where(is_reverse, _asidx(_MIN), _asidx(_MAX))


def normalize_getitem_key(
    key: GetitemIndex,
) -> tuple[GetitemItem, ...] | BoolMask:
    from . import onnx

    if isinstance(key, bool):
        return onnx.const(key, dtype=onnx.bool_)
    if (
        isinstance(key, tuple)
        and len(key) > 0
        and all(isinstance(el, bool) for el in key)
    ):
        key = onnx.const(np.array(key), dtype=onnx.bool_)

    # Fish out the boolean masks before normalizing to tuples
    if isinstance(key, onnx.TyArrayBool):
        return key

    if isinstance(key, int | slice | EllipsisType | None | onnx.TyArrayInt64):
        return (_normalize_getitem_key_item(key),)

    if isinstance(key, tuple):
        # Cannot use `count` since that will trigger an __eq__ call
        # with type promotion against `...`
        if sum(isinstance(el, type(...)) for el in key) > 1:
            raise IndexError("more than one ellipsis (`...`) in index tuple")

        return tuple(_normalize_getitem_key_item(el) for el in key)

    raise IndexError(f"unexpected key `{key}`")


def _normalize_getitem_key_item(key: GetitemItem) -> GetitemItem:
    if isinstance(key, slice):
        slice_kwargs = {"start": key.start, "stop": key.stop, "step": key.step}
        out: dict[str, int | None | onnx.TyArrayInt64] = {}
        for k, arg in slice_kwargs.items():
            if not isinstance(arg, None | int | onnx.TyArrayInt64):
                raise IndexError(f"slice argument `{k}` has invalid type `{type(arg)}`")
            if isinstance(arg, onnx.TyArrayBase):
                if arg.dtype not in (onnx.int32, onnx.int64) or arg.ndim != 0:
                    raise IndexError(
                        f"array-slice argument must be be an int64 scalar. Found `{arg}`"
                    )
                out[k] = arg.astype(onnx.int64)
            else:
                out[k] = arg
        return slice(*out.values())  # Beware that we use the dict order here!

    if isinstance(key, int | None | EllipsisType):
        return key

    if key.ndim != 0:
        raise IndexError(
            f"index arrays must be rank-0 tensors; found rank `{key.ndim}`"
        )
    return key.astype(onnx.int64)


def key_to_indices(key: tuple[slice | int, ...], shape: TyArrayInt64) -> TyArrayInt64:
    """Compute expanded indices for ``key`` for an array of shape ``shape``."""
    # TODO: Allow dynamic inputs to DType.__ndx_arange__?
    import ndonnx as ndx

    indices_per_dim = []
    for s, length in zip(key, shape):
        if isinstance(s, int):
            stop = None if s == -1 else s + 1
            s = slice(s, stop, 1)
        indices_per_dim.append(
            ndx.asarray(ort_compat.range(*[el._var for el in _get_indices(s, length)]))
        )

    grid = ndx.meshgrid(*indices_per_dim, indexing="ij")
    indices = ndx.concat([ndx.reshape(el, (-1, 1)) for el in grid], axis=-1)

    return safe_cast(onnx.TyArrayInt64, indices._tyarray)


def _get_indices(
    s: slice, length: int | TyArrayInt64
) -> tuple[TyArrayInt64, TyArrayInt64, TyArrayInt64]:
    """Array-compatible implementation of ``slice.indices``.

    Slice members must be of type ``int`` or ``None`` while `length` may be an array.
    """
    from .funcs import maximum, minimum

    length_ = (
        length
        if isinstance(length, onnx.TyArrayInt64)
        else onnx.const(length, onnx.int64)
    )

    step = 1 if s.step is None else s.step
    if not isinstance(step, int):
        raise IndexError(f"'step' must be of type 'int' found `{type(step)}`")

    if step == 0:
        raise IndexError("'step' must not be zero")

    step_is_negative = step < 0

    if step_is_negative:
        lower = onnx.const(-1, onnx.int64)
        upper = length_ + lower
    else:
        lower = onnx.const(0, onnx.int64)
        upper = length_

    if s.start is None:
        start_arr = upper if step_is_negative else lower
    elif isinstance(s.start, int):
        start_is_neg = s.start < 0
        slice_start = onnx.const(s.start, dtype=onnx.int64)
        if start_is_neg:
            start_arr = maximum(slice_start + length_, lower)
        else:
            start_arr = minimum(slice_start, upper)
    else:
        raise IndexError(f"'start' must be 'int' or 'None', found `{s.start}`")

    if s.stop is None:
        stop = lower if step_is_negative else upper
    elif isinstance(s.stop, int):
        stop = onnx.const(s.stop, dtype=onnx.int64)
        stop_is_neg = stop < 0
        if stop_is_neg:
            stop = maximum(stop + length_, lower)
        else:
            stop = minimum(stop, upper)
    else:
        raise IndexError(f"'stop' must be 'int' or 'None', found `{s.stop}`")

    return start_arr, stop, onnx.const(step, onnx.int64)
