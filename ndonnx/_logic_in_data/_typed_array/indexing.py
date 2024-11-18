# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from types import EllipsisType
from typing import TYPE_CHECKING, Any, Union

import numpy as np

if TYPE_CHECKING:
    from . import TyArrayBool, TyArrayInt64, TyArrayInteger


GetitemIndexStatic = Union[
    int | slice | EllipsisType | None, tuple[int | slice | EllipsisType | None, ...]
]
GetitemIndex = Union[GetitemIndexStatic, "TyArrayInteger", "TyArrayBool"]

SetitemIndexStatic = Union[
    int | slice | EllipsisType, tuple[int | slice | EllipsisType, ...]
]
SetitemIndex = Union[SetitemIndexStatic, "TyArrayInteger", "TyArrayBool"]


class FancySlice:
    """Normalized "slice" object which may also represent a single-item index.

    In the latter case, the result needs to be squeezed after the indexing operation.
    """

    start: TyArrayInt64
    stop: TyArrayInt64
    step: TyArrayInt64
    squeeze: bool

    def __init__(self, obj: slice | int | TyArrayInt64):
        from . import TyArrayInt64

        if isinstance(obj, TyArrayInt64 | int):
            self.start = _asidx(obj)
            self.stop = _compute_end_single_idx(obj)
            self.step = _asidx(1)
            self.squeeze = True
            return

        # `slice` case
        def validate(obj: Any) -> TyArrayInt64 | None:
            if obj is None:
                return None
            if isinstance(obj, int | TyArrayInt64):
                return _asidx(obj)
            raise TypeError(f"Unsupported type as slice parameter `{type(obj)}`")

        step = validate(obj.step)
        start = _compute_start_slice(validate(obj.start), step)
        stop = _compute_stop_slice(validate(obj.stop), step)
        self.step = _asidx(1) if step is None else step
        self.start = start
        self.stop = stop
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
    from . import TyArrayInt64
    from .funcs import astyarray, safe_cast

    arr = safe_cast(TyArrayInt64, astyarray(obj))

    if arr.ndim != 0:
        raise IndexError(f"expected scalar array but found array of rank `{arr.ndim}`")

    return arr


_MAX = np.iinfo(np.int64).max
_MIN = np.iinfo(np.int64).min


def _compute_end_single_idx(start: TyArrayInt64 | int) -> TyArrayInt64:
    from . import TyArrayBool, TyArrayInt64
    from .funcs import astyarray, safe_cast, where

    start = _asidx(start)
    end = start + _asidx(1)
    end_is_zero = safe_cast(TyArrayBool, end == astyarray(0))
    return safe_cast(TyArrayInt64, where(end_is_zero, _asidx(_MAX), end))


def _compute_start_slice(
    start: int | TyArrayInt64 | None, step: int | TyArrayInt64 | None
) -> TyArrayInt64:
    from . import TyArrayBool, TyArrayInt64
    from .funcs import safe_cast, where

    if start is not None:
        return _asidx(start)
    if step is None:
        step = 1
    step = _asidx(step)

    zero = _asidx(0)
    step_gt_zero = safe_cast(TyArrayBool, step > zero)

    return safe_cast(TyArrayInt64, where(step_gt_zero, zero, _asidx(_MAX)))


def _compute_stop_slice(
    stop: int | TyArrayInt64 | None, step: int | TyArrayInt64 | None
) -> TyArrayInt64:
    from . import TyArrayBool, TyArrayInt64
    from .funcs import safe_cast, where

    if stop is not None:
        return _asidx(stop)
    if step is None:
        step = 1

    step = _asidx(step)

    # The following abomination essentially reads as:
    # `where(step < 1, _MIN, _MAX)`

    is_reverse = safe_cast(TyArrayBool, step < _asidx(1))
    return safe_cast(TyArrayInt64, where(is_reverse, _asidx(_MIN), _asidx(_MAX)))
