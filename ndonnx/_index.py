# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from typing import TypeVar, Union

import numpy as np

from ._utility import get_rank

ellipsis = TypeVar("ellipsis")
ScalarIndexType = Union[int, bool, slice, ellipsis, None]


def index_normalise(a: Iterable[ScalarIndexType]) -> tuple[ScalarIndexType, ...]:
    ret: list[ScalarIndexType] = []
    for x in a:
        # x: Union[int, bool, slice, None]
        if isinstance(x, (int, bool)):
            ret.append(x)
        elif isinstance(x, slice):
            INDEX_MIN: int = np.iinfo(np.int64).min
            INDEX_MAX: int = np.iinfo(np.int64).max
            start = (
                x.start
                if x.start is not None
                else (0 if x.step is None or x.step > 0 else INDEX_MAX)
            )
            stop = (
                x.stop
                if x.stop is not None
                else (INDEX_MAX if x.step is None or x.step > 0 else INDEX_MIN)
            )
            step = x.step if x.step is not None else 1

            if start == 0 and stop == INDEX_MAX and step == 1:
                ret.append(slice(None, None, None))
            else:
                ret.append(slice(start, stop, step))
        elif x is None:
            ret.append(x)
        else:
            # ellipses are expected to be handled prior to this function call
            raise TypeError(f"Index {x} for type {type(x)} not supported")

    return tuple(ret)


def construct_index(
    arr, index: ScalarIndexType | tuple[ScalarIndexType, ...]
) -> tuple[ScalarIndexType, ...]:
    index_ = index if isinstance(index, tuple) else (index,)
    if any(i is Ellipsis for i in index_):
        rank = get_rank(arr)
        ellipsis_position = index_.index(Ellipsis)
        count_some = len(
            [x for x in index_ if not isinstance(x, (type(None), type(Ellipsis)))]
        )
        index_ = (
            index_[:ellipsis_position]
            + tuple([slice(None, None, None)] * (rank - count_some))
            + index_[ellipsis_position + 1 :]
        )
    return index_normalise(index_)
