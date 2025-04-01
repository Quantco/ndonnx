# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from types import EllipsisType
from typing import Any

import numpy as np

from . import onnx


class FancySlice:
    """Normalized "slice" object which may also represent a single-item index.

    In the latter case, the result needs to be squeezed after the indexing operation.
    """

    # 1D arrays with a single element so that concatenating them later is cheaper
    start: onnx.TyArrayInt64
    stop: onnx.TyArrayInt64
    step: onnx.TyArrayInt64
    squeeze: bool

    def __init__(self, obj: slice | int | onnx.TyArrayInt64):
        if isinstance(obj, onnx.TyArrayInt64 | int):
            self.start = _asidx(obj)
            self.stop = _asidx(_compute_end_single_idx(obj))
            self.step = _asidx(1)
            self.squeeze = True
            return

        # `slice` case
        def validate(obj: Any) -> onnx.TyArrayInt64 | None:
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


def _asidx(obj: int | onnx.TyArrayInt64) -> onnx.TyArrayInt64:
    arr = onnx.const(obj, onnx.int64) if isinstance(obj, int) else obj

    if arr.ndim != 0:
        raise IndexError(f"expected scalar array but found array of rank `{arr.ndim}`")

    return arr


_MAX = np.iinfo(np.int64).max
_MIN = np.iinfo(np.int64).min


def _compute_end_single_idx(start: onnx.TyArrayInt64 | int) -> onnx.TyArrayInt64 | int:
    if isinstance(start, int):
        start = start
        end = start + 1
        end_is_zero = end == 0
        return _MAX if end_is_zero else end
    end_ = start + 1
    end_is_zero_ = end_ == 0
    return onnx.where(end_is_zero_, _asidx(_MAX), end_)


def _compute_start_slice(
    start: int | onnx.TyArrayInt64 | None, step: int | onnx.TyArrayInt64 | None
) -> onnx.TyArrayInt64 | int:
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
    stop: int | onnx.TyArrayInt64 | None, step: int | onnx.TyArrayInt64 | None
) -> onnx.TyArrayInt64 | int:
    if stop is not None:
        return stop
    if step is None:
        step = 1

    is_reverse = step < 1
    if isinstance(is_reverse, bool):
        return _MIN if is_reverse else _MAX
    return onnx.where(is_reverse, _asidx(_MIN), _asidx(_MAX))


def normalize_getitem_key(
    key: onnx.GetitemIndex,
) -> tuple[onnx.GetitemItem, ...] | onnx._BoolMask:
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


def _normalize_getitem_key_item(key: onnx.GetitemItem) -> onnx.GetitemItem:
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
