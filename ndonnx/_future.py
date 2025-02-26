# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

from collections.abc import Callable
from functools import partial, wraps
from pathlib import Path
from typing import Any, TypeVar, cast

import onnx
import spox._future
import spox._value_prop

F = TypeVar("F", bound=Callable[..., Any])


def _run_onnxruntime(
    model: onnx.ModelProto,
    input_feed: dict[str, spox._value_prop.ORTValue],
    shared_library: str | Path,
) -> dict[str, spox._value_prop.ORTValue]:
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


def propagate_with_custom_operators(shared_library: str | Path, /) -> Callable[[F], F]:
    """Enable value propagation with custom operators defined in ``shared_library``.

    The shared library must be compatible with onnxruntime and the latter must be
    installed.
    """

    def prop(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            old_run_fun = spox._value_prop._run_onnxruntime

            # TODO: Use context manager to clean up state properly in case of errors
            spox._value_prop._run_onnxruntime = partial(
                _run_onnxruntime, shared_library=shared_library
            )
            with spox._future.value_prop_backend(
                spox._value_prop.ValuePropBackend.ONNXRUNTIME
            ):
                res = fn(*args, **kwargs)

            spox._value_prop._run_onnxruntime = old_run_fun

            return res

        return cast(F, wrapper)

    return prop
