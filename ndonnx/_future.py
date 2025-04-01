# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

import importlib
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial, wraps
from pathlib import Path
from typing import TypeVar

import onnx
import spox._future
import spox._value_prop
from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


def _run_onnxruntime(
    model: onnx.ModelProto,
    input_feed: dict[str, spox._value_prop.ORTValue],
    shared_library: str | Path | None,
) -> dict[str, spox._value_prop.ORTValue]:
    # `None` input is used for testing only!
    import onnxruntime

    # Silence possible warnings during execution (especially constant folding)
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    if shared_library is not None:
        # We pass `None` for testing
        options.register_custom_ops_library(str(shared_library))

    session = onnxruntime.InferenceSession(model.SerializeToString(), options)
    output_names = [output.name for output in session.get_outputs()]
    return dict(zip(output_names, session.run(None, input_feed)))


@contextmanager
def _spox_set_custom_op(shared_library: Path | str | None):
    """Make spox use the specified custom operator library when creating a session.

    This context manager properly unsets that library even if an exception is raised.
    """
    # `None` input is used for testing only!
    old_run_fun = spox._value_prop._run_onnxruntime

    spox._value_prop._run_onnxruntime = partial(
        _run_onnxruntime, shared_library=shared_library
    )
    try:
        yield
    finally:
        spox._value_prop._run_onnxruntime = old_run_fun


def propagate_with_custom_operators(
    shared_library: str | Path, /
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Enable value propagation with custom operators defined in ``shared_library``.

    The shared library must be compatible with onnxruntime and the latter must be
    installed.
    """
    if not importlib.util.find_spec("onnxruntime"):
        raise ImportError(
            "onnxruntime must be installed to use value propagation across custom operators"
        )

    def prop(fn: Callable[P, T]) -> Callable[P, T]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with _spox_set_custom_op(shared_library):
                with spox._future.value_prop_backend(
                    spox._value_prop.ValuePropBackend.ONNXRUNTIME
                ):
                    return fn(*args, **kwargs)

        return wrapper

    return prop
