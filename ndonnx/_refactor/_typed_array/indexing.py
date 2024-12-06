# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from types import EllipsisType
from typing import TYPE_CHECKING, Any, TypeAlias, Union

import numpy as np

if TYPE_CHECKING:
    from .._array import Array
    from .._array import GetitemIndex as GetitemIndexPub
    from . import TyArrayBool, TyArrayInt64, TyArrayInteger

    GetitemScalarPub: TypeAlias = int | slice | EllipsisType | None | Array

    GetitemScalar: TypeAlias = int | slice | EllipsisType | None | TyArrayInt64
    GetitemTuple: TypeAlias = tuple[GetitemScalar, ...]
    GetitemIndexStatic: TypeAlias = GetitemScalar | GetitemTuple
    GetitemIndex: TypeAlias = GetitemIndexStatic | TyArrayInteger | TyArrayBool

    SetitemIndexStatic: TypeAlias = Union[
        int | slice | EllipsisType, tuple[int | slice | EllipsisType, ...]
    ]
    SetitemIndex: TypeAlias = Union[SetitemIndexStatic, TyArrayInteger, TyArrayBool]


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

    obj_ = np.array(obj, np.int64) if isinstance(obj, int) else obj
    arr = safe_cast(TyArrayInt64, astyarray(obj_))

    if arr.ndim != 0:
        raise IndexError(f"expected scalar array but found array of rank `{arr.ndim}`")

    return arr


_MAX = np.iinfo(np.int64).max
_MIN = np.iinfo(np.int64).min


def _compute_end_single_idx(start: TyArrayInt64 | int) -> TyArrayInt64:
    from . import TyArrayBool, TyArrayInt64
    from .funcs import safe_cast, where

    start = _asidx(start)
    end = start + _asidx(1)
    end_is_zero = safe_cast(TyArrayBool, end == _asidx(0))
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


def normalize_getitem_key(
    key: GetitemIndexPub | GetitemIndex,
) -> GetitemTuple | TyArrayBool:
    from .. import Array
    from . import TyArrayBool, TyArrayInt64, onnx
    from .funcs import astyarray

    if isinstance(key, bool):
        return astyarray(key, dtype=onnx.bool_)
    if (
        isinstance(key, tuple)
        and len(key) > 0
        and all(isinstance(el, bool) for el in key)
    ):
        key = astyarray(np.array(key), dtype=onnx.bool_)  # type: ignore

    # Fish out the boolean masks before normalizing to tuples
    if isinstance(key, Array) and isinstance(key._tyarray, TyArrayBool):
        return key._tyarray
    if isinstance(key, TyArrayBool):
        return key

    if isinstance(key, int | slice | EllipsisType | None | Array | TyArrayInt64):
        return (_normalize_getitem_key_item(key),)

    if isinstance(key, tuple):
        # Cannot use `count` since that will trigger an __eq__ call
        # with type promotion against `...`
        if sum(isinstance(el, type(...)) for el in key) > 1:
            raise IndexError("more than one ellipsis (`...`) in index tuple")

        return tuple(_normalize_getitem_key_item(el) for el in key)

    raise IndexError(f"unexpected key `{key}`")


def _normalize_getitem_key_item(key: GetitemScalarPub | GetitemScalar) -> GetitemScalar:
    from .. import Array
    from .onnx import TyArrayBase, TyArrayInt64, int32, int64

    if isinstance(key, Array):
        if key.ndim != 0:
            raise IndexError(f"Index array must be a scalar but has rank `{key.ndim}`")
        if isinstance(key._tyarray, TyArrayInt64):
            return key._tyarray
        raise IndexError(
            f"indexing array must have integer or boolean data type; found `{key.dtype}`"
        )

    if isinstance(key, slice):
        slice_kwargs = {"start": key.start, "stop": key.stop, "step": key.step}
        out: dict[str, int | None | TyArrayInt64] = {}
        for k, arg in slice_kwargs.items():
            if not isinstance(arg, Array | None | int | TyArrayInt64):
                raise IndexError(f"slice argument `{k}` has invalid type `{type(arg)}`")
            if isinstance(arg, Array | TyArrayBase):
                if arg.dtype not in (int32, int64) or arg.ndim != 0:
                    raise IndexError(
                        f"array-slice argument must be be an int64 scalar. Found `{arg}`"
                    )
                ty_arr = arg._tyarray if isinstance(arg, Array) else arg
                out[k] = ty_arr.astype(int64)
            else:
                out[k] = arg
        return slice(*out.values())  # Beware that we use the dict order here!

    if isinstance(key, int | None | type(...)):
        return key
    raise IndexError(f"unexpected index element `{key}`")


def _key_to_indices(key: tuple[slice | int, ...], shape: TyArrayInt64) -> TyArrayInt64:
    """Compute expanded indices for ``key`` for an array of shape ``shape``."""
    # TODO: Allow dynamic inputs to DType._arange?
    import spox.opset.ai.onnx.v21 as op

    import ndonnx._refactor as ndx

    from . import TyArrayInt64
    from .funcs import safe_cast

    indices_per_dim = []
    for s, length in zip(key, shape):
        if isinstance(s, int):
            stop = None if s == -1 else s + 1
            s = slice(s, stop, 1)
        indices_per_dim.append(
            ndx.asarray(op.range(*[el.var for el in _get_indices(s, length)]))
        )

    grid = ndx.meshgrid(*indices_per_dim, indexing="ij")
    indices = ndx.concat([ndx.reshape(el, (-1, 1)) for el in grid], axis=-1)

    return safe_cast(TyArrayInt64, indices._tyarray)


def _get_indices(
    s: slice, length: int | TyArrayInt64
) -> tuple[TyArrayInt64, TyArrayInt64, TyArrayInt64]:
    """Array-compatible implementation of ``slice.indices``.

    Slice members must be of type ``int`` or ``None`` while `length` may be an array.
    """
    from ndonnx._refactor import Array, asarray, int64, maximum, minimum, where

    from .funcs import astyarray, safe_cast
    from .onnx import TyArrayInt64

    length_ = Array._from_tyarray(astyarray(length))

    step = 1 if s.step is None else s.step
    if not isinstance(step, int):
        raise IndexError(f"'step' must be of type 'int' found `{type(step)}`")

    if step == 0:
        raise IndexError("'step' must not be zero")

    step_is_negative = step < 0

    if step_is_negative:
        lower = -1
        upper: Array = length_ + lower
    else:
        lower = 0
        upper = length_

    if s.start is None:
        start = upper if step_is_negative else lower
    elif isinstance(s.start, int):
        start = asarray(s.start, dtype=int64)
        start = where(
            start < 0, maximum(start + length_, asarray(lower)), minimum(start, upper)
        )
    else:
        raise IndexError(f"'start' must be 'int' or 'None', found `{s.start}`")

    if s.stop is None:
        stop = lower if step_is_negative else upper
    elif isinstance(s.stop, int):
        stop = asarray(s.stop, dtype=int64)
        stop = where(
            stop < 0, maximum(stop + length_, asarray(lower)), minimum(stop, upper)
        )
    else:
        raise IndexError(f"'stop' must be 'int' or 'None', found `{s.stop}`")

    (start_, stop_, step_) = (
        safe_cast(TyArrayInt64, astyarray(el)) for el in [start, stop, step]
    )
    return (start_, stop_, step_)


def _move_ellipsis_back(
    ndim: int, key: tuple[int | slice | EllipsisType, ...]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int | slice | EllipsisType, ...]]:
    """Permute axes such that the ellipsis-axes are at the end."""

    if ... not in key:
        raise ValueError("No ellipsis found in 'key'")
    ellipsis_pos = key.index(...)
    current_dims = list(range(ndim))
    ellipsis_len = ndim - (len(key) - 1)
    new_perm = tuple(
        current_dims[:ellipsis_pos]
        + current_dims[ellipsis_pos + ellipsis_len :]
        + current_dims[ellipsis_pos:][:ellipsis_len]
    )
    inverse_map = tuple(int(el) for el in np.argsort(new_perm))

    keys_no_ellipsis = tuple(el for el in key if el != ...)
    return new_perm, inverse_map, keys_no_ellipsis
