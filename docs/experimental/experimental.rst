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


User-defined data types
-----------------------

Downstream projects may defined their own data types.
This is done by subclassing the abstract classes :class:`ndonnx.DType` and :class:`ndonnx._experimental.TyArrayBase`.
Once all required ``abstract_methods`` on the new subclasses are defined one may use them in the existing ``ndonnx`` interfaces.
Built-in data types utilize exactly the same machinery and thus may serve as a starting point to users looking to implement their own.
Due to their limited scope the datetime and timedelta data types defined in `here <https://github.com/Quantco/ndonnx/blob/typed-array/ndonnx/_typed_array/datetime.py>`_ may be of particular interest.
