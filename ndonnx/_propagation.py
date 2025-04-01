# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast
from warnings import warn

warn(
    "the 'ndonnx._propagation' module is deprecated and can be simply removed.",
    DeprecationWarning,
)

F = TypeVar("F", bound=Callable[..., Any])


def eager_propagate(fn: F) -> F:
    @wraps(fn)
    def inner(*args, **kwargs):
        return fn(*args, **kwargs)

    return cast(F, inner)
