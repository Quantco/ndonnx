Experimental
============

This section documents features that are currently experimental. While we aim to promote these features to the public API, they may currently undergo breaking changes without prior notice and there is no strict timeline for inclusion.

.. _propagation:

Propagating values when interoperating with Spox
------------------------------------------------

As documented in the :doc:`Spox Interoperability </spox/spoxintegration>` section, you can extract a :class:`spox.Var` from an :class:`ndonnx.Array` using the :func:`ndonnx.Array.spox_var` method.
When exiting ndonnx to implement functionality using Spox' primitives, ndonnx is unable to propagate values by default.
Using the :func:`ndonnx._propagation.eager_propagate` decorator, you can make a function capable of propagating values. As with value propagation more broadly in ndonnx, this currently requires you to have onnxruntime installed.

As an example, we implement a function to one-hot-encode an input array of strings.

..  code-block:: python

    from typing import Iterable
    import ndonnx as ndx
    from ndonnx._propagation import eager_propagate
    import spox.opset.ai.onnx.ml.v3 as ml

    @eager_propagate
    def one_hot_encode(x: ndx.Array, categories: Iterable[str]) -> ndx.Array:
        return ndx.from_spox_var(ml.one_hot_encoder(x.spox_var(), cats_strings=categories))

    x = ndx.asarray(["a", "b", "a"])
    y = one_hot_encode(x, ["a", "b"])
    print(y.to_numpy())
    # [[1. 0.]
    # [0. 1.]
    # [1. 0.]]

Using a custom operator library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ONNX standard permits the definition of extensions in the form of custom operators.
If you are using your own custom operator with a onnxruntime-compatible implementation, value propagation may be extended to these operators too.

..  code-block:: python

    import ndonnx as ndx
    from ndonnx._propagation import _eager_propagate_with_custom_operators

    eager_propagate = _eager_propagate_with_custom_operators("path/to/shared_library.so")

    @eager_propagate
    def my_op(x: ndx.Array) -> ndx.Array:
        return ndx.from_spox_var(my_op(x.spox_var()))

.. warning::
    Using custom operators may limit the portability of your exported ONNX model since not all runtimes may implement a non-standard operator.


User-defined data types
-----------------------

Internally, ``ndonnx`` defines the ``StructType`` interface. This interface is used to implement the ``Nullable`` data types exported by ``ndonnx`` like ``nint64`` and ``nutf8`` but is actually designed to enable user-defined data types outside of ndonnx.

The interface is defined below:

.. autoclass:: ndonnx._experimental.StructType
   :members:
   :undoc-members:
   :private-members:


Implementing a ``datetime64`` data type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that we will use the ``_experimental`` submodule.

..  code-block:: python

    from typing import Self

    import ndonnx as ndx
    from ndonnx import CastError, CoreType, Array
    from ndonnx._experimental import OperationsBlock, CastMixin, Schema, StructType
    import numpy as np


    class DatetimeOperations(OperationsBlock):
        def less_equal(self, x: Array, y: Array) -> Array:
            if x.dtype == y.dtype == datetime64:
                return x.unix_timestamp <= y.unix_timestamp
            return NotImplemented

    class Datetime64(StructType, CastMixin):
        def _fields(self) -> dict[str, StructType | CoreType]:
            return {
                "unix_timestamp": ndx.int64,
            }

        def copy(self) -> Self:
            return self

        def _schema(self) -> Schema:
            return Schema(
                type_name="datetime64",
                author="myextensionlibrary",
            )

        def _parse_input(self, input: np.ndarray) -> dict:
            # We accept numpy arrays of dtype datetime64[s] as input only.
            if input.dtype == np.dtype("datetime64[s]"):
                unix_timestamp = input.astype(np.int64)
            else:
                raise ValueError(f"Cannot parse input of dtype {input.dtype} to {self}")

            return {
                "unix_timestamp": self._fields()["unix_timestamp"]._parse_input(unix_timestamp),
            }

        def _assemble_output(self, fields: dict[str, np.ndarray]) -> np.ndarray:
            return fields["unix_timestamp"].astype("datetime64[s]")

        def _cast_to(self, array: Array, dtype: CoreType | StructType) -> Array:
            raise CastError(f"Cannot cast {self} to {dtype}")

        def _cast_from(self, array: Array) -> Array:
            if isinstance(array.dtype, ndx.Numerical):
                unix_timestamp = array.astype(ndx.int64)
            else:
                raise CastError(f"Cannot cast {array.dtype} to {self}")

            return Array._from_fields(self, unix_timestamp=unix_timestamp)

        _ops = DatetimeOperations()


    datetime64 = Datetime64()


We can test our implementation out:

..  code-block:: python

    a = np.array(["2024-02-01", "1969-07-20", "1912-02-12"], dtype="datetime64[s]")
    b = np.array(["1924-02-01", "1969-07-21"], dtype="datetime64[s]")

    x = ndx.asarray(a, dtype=datetime64)
    y = ndx.asarray(b, dtype=datetime64)
    print(f"{x=}")
    # x=Array(['2024-02-01T00:00:00' '1969-07-20T00:00:00' '1912-02-12T00:00:00'], dtype=Datetime64)
    print(f"{y=}")
    # y=Array(['1924-02-01T00:00:00' '1969-07-21T00:00:00'], dtype=Datetime64)
    print(x[:2] <= y)
    # Array([False  True], dtype=Boolean)

We've successfully implemented a ``datetime64`` data type! We now have a much more useful abstraction for working with datetime data in ONNX.

Let's track back and explain how we just implemented this, in English rather than code:

1. Decide the appropriate layout for ``datetime64`` and encoded it in ``_fields``. Representing time using an ``ndx.int64`` unix timestamp is a reasonable and efficient approach.
2. Implement ``_parse_input`` and ``_assemble_output``. These may be used by runtime utilities (and internally by ``ndonnx`` for propagation of build time values) to disassemble input datetime NumPy arrays into the constituent ``_fields`` and reassemble into datetime NumPy array from these fields. These functions must be each inverses.
3. Implement ``_schema``, which advertises some metadata about this type like the name, version and source library. This is written into the ONNX ``ModelProto`` for inference utilities and runtimes to optionally leverage.
4. Implement some optional casting behaviour using the ``CastMixin``. We would like to be able to cast between numerical arrays and ``datetime64`` arrays.
5. Implement functions that operate on arrays of our new type. Without this, arrays with ``datetime64`` dtype cannot do much. This is where the ``OperationsBlock`` is implemented and assigned to the ``_ops`` field of the data type.
