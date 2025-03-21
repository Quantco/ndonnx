Experimental
============

This section documents features found in the :func:`ndonnx._experimental` module.
Breaking changes may occur in this module with no prior notice.


.. _propagation:

Value propagation with custom operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ndonnx strives to provide a user experience where operations between constant arrays produce a constant arrays which values that that the user may inspect using the :func:`Array.unwrap_numpy` member function.
However, values cannot be propagated across custom operators by default.
Value propagation may be enabled in scenarios where a custom operator exists for the onnxruntime in the form of a shared library:

..  code-block:: python

    import ndonnx as ndx
    from ndonnx._experimental import propagate_with_custom_operators

    @propagate_with_custom_operators("path/to/shared_library.so")
    def my_op(x: ndx.Array) -> ndx.Array:
        return ndx.asarray(my_op(x.disassemble()))

.. warning::
    Using custom operators may limit the portability of your exported ONNX model since not all runtimes may implement a non-standard operator.
