# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from collections import namedtuple
from pathlib import Path

import numpy as np
import spox
import spox.opset.ai.onnx.v19 as op

import ndonnx as ndx

from ._corearray import _CoreArray

Arguments = namedtuple("Arguments", ["args", "kwargs"])
LazyEagerPair = namedtuple("LazyEagerPair", ["lazy", "value"])

try:
    import onnxruntime as ort

    ORT_PRESENT = True
except ImportError:
    ORT_PRESENT = False
    warnings.warn(
        "onnxruntime not found. Eager propagation and constant folding optimizations will be disabled."
    )


def eager_propagate(fn):
    def wrapper(*args, **kwargs):
        return _eager_propagate_with_custom_operators(None)(fn)(*args, **kwargs)

    return wrapper


def _eager_propagate_with_custom_operators(custom_operators_path: Path | None):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            constant_inputs, lazy_args, flattened_inference_inputs = (
                _aggregate_arguments(Arguments(args, kwargs))
            )
            outputs = fn(*args, **kwargs)
            if constant_inputs and ORT_PRESENT:
                is_sequence = isinstance(outputs, (list, tuple))
                prediction = _propagate_helper(
                    fn, lazy_args, flattened_inference_inputs, custom_operators_path
                )
                outputs = [outputs] if not is_sequence else outputs
                corearrays: list[_CoreArray] = []
                for output in outputs:
                    if isinstance(output, _CoreArray):
                        corearrays.append(output)
                    else:
                        _get_corearrays(output, corearrays)
                for corearray_out, value in zip(corearrays, prediction):
                    if corearray_out.dtype == ndx.utf8:
                        value = value.astype(np.str_)
                    corearray_out._eager_value = value
                    corearray_out.var = op.const(value)

                return outputs[0] if not is_sequence else outputs
            else:
                return outputs

        return wrapper

    return decorator


def _aggregate_arguments(
    args: Arguments,
) -> tuple[bool, Arguments, list[LazyEagerPair]]:
    """Takes a function as input (which may only consist of _CoreArrays and various
    structures of it) and also the arguments to the function.

    Inputs:
        args: Arguments
            The positional and keyword arguments to the function.
    Returns:
        constant_inputs: bool
            Whether all the inputs are constant.
        lazy_args: Arguments
            The arguments to the function, but with all the _CoreArrays replaced with lazy variants.
        flattened_inference_inputs: list[LazyEagerPair]
            The constituent input _CoreArrays alongside their eager values.
    """

    constant_inputs = True
    lazy_args: Arguments = Arguments([], {})
    flattened_inference_inputs: list[LazyEagerPair] = []

    def collect_lazy_arguments(obj):
        nonlocal constant_inputs
        if isinstance(obj, _CoreArray):
            if obj.to_numpy() is None:
                constant_inputs = False
                return obj
            else:
                constant_inputs &= obj.to_numpy() is not None
                if obj.dtype == ndx.utf8:
                    # Lazy variant due to onnxruntime bug
                    lazy = _CoreArray(
                        spox.argument(
                            spox.Tensor(
                                obj.dtype.to_numpy_dtype(),
                                obj.var.unwrap_tensor().shape,
                            )
                        ),
                    )
                    flattened_inference_inputs.append(
                        LazyEagerPair(lazy, obj.to_numpy())
                    )
                    return lazy
                else:
                    return obj
        elif isinstance(obj, ndx.Array):
            obj_value = obj.to_numpy()
            if obj_value is None:
                constant_inputs = False
                return obj
            else:
                if obj.dtype in (ndx.utf8, ndx.nutf8):
                    # Lazy variant due to onnxruntime bug
                    lazy = ndx.array(shape=obj._static_shape, dtype=obj.dtype)  # type: ignore

                    # disassemble the array into its core_arrays
                    _flatten(obj, lazy, flattened_inference_inputs)
                    return lazy
                else:
                    return ndx.asarray(obj_value, obj.dtype)
        elif isinstance(obj, (list, tuple)):
            return type(obj)(map(collect_lazy_arguments, obj))
        elif isinstance(obj, slice):
            return slice(
                collect_lazy_arguments(obj.start),
                collect_lazy_arguments(obj.stop),
                collect_lazy_arguments(obj.step),
            )
        else:
            return obj

    for arg in args.args:
        lazy_args.args.append(collect_lazy_arguments(arg))
    for key, value in args.kwargs.items():
        lazy_args.kwargs[key] = collect_lazy_arguments(value)
    return constant_inputs, lazy_args, flattened_inference_inputs


def _flatten(eager_array, lazy_array, flattened_inference_inputs):
    if isinstance(eager_array, _CoreArray):
        flattened_inference_inputs.append(
            LazyEagerPair(lazy_array, eager_array.to_numpy())
        )
    else:
        for name in eager_array._fields:
            _flatten(
                eager_array._fields[name],
                lazy_array._fields[name],
                flattened_inference_inputs,
            )


def _get_corearrays(array: ndx.Array, core_arrays: list[_CoreArray]):
    for name, field in array._fields.items():
        if isinstance(field, _CoreArray):
            core_arrays.append(field)
        else:
            _get_corearrays(field, core_arrays)


def _propagate_helper(
    fn, lazy_args, flattened_inference_inputs, custom_operators_path: Path | None = None
):
    lazy_outputs = fn(*lazy_args.args, **lazy_args.kwargs)
    is_sequence = isinstance(lazy_outputs, (list, tuple))
    if not is_sequence:
        lazy_outputs = [lazy_outputs]

    # destructure lazy_args and lazy_outputs into sequences of _CoreArrays for ModelProto creation

    model_proto = ndx.build(
        {
            f"input_{i}": input.lazy
            for i, input in enumerate(flattened_inference_inputs)
        },
        {
            f"output_{i}": output_corearray
            for i, output_corearray in enumerate(lazy_outputs)
        },
    )

    sess_options = ort.SessionOptions()
    if custom_operators_path is not None:
        sess_options.register_custom_ops_library(str(custom_operators_path))
    sess = ort.InferenceSession(
        model_proto.SerializeToString(), sess_options=sess_options
    )
    prediction = sess.run(
        None,
        {
            f"input_{i}": input.value
            for i, input in enumerate(flattened_inference_inputs)
        },
    )

    return prediction
