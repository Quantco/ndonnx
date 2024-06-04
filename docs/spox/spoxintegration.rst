Interoperating with Spox
========================

`Spox <https://github.com/quantco/spox>`_ defines a Pythonic, direct interface to the ONNX standard and is an effective way to construct ONNX graphs.
ndonnx builds on Spox and so it is able to provide excellent interoperability with the lower level interface Spox has when needed.

When is it useful to interoperate with Spox?
--------------------------------------------

1. You want to directly access an ONNX operator that describes your intent. High level operators like `LSTM <https://github.com/onnx/onnx/blob/main/docs/Operators.md#LSTM>`_ or `TreeEnsemble <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.TreeEnsemble>`_ are not directly accessible from ndonnx since they aren't natural array operations.
2. You want to reuse an existing ONNX model. Spox's `Inline <https://spox.readthedocs.io/en/latest/guides/inline.html>`_ tool is a powerful utility for this.
3. You want to dispatch to certain specialized `custom operators <https://onnxruntime.ai/docs/reference/operators/add-custom-op.html>`_ that go beyond the scope of the ONNX standard. This is also achieved via Spox's Inline tool.

Moving between Spox and ndonnx
------------------------------

ndonnx to Spox
~~~~~~~~~~~~~~

Extract a :class:`spox.Var` from an :class:`ndonnx.Array` using the :func:`ndonnx.Array.spox_var` method. This allows you to use Spox to manipulate the ONNX graph more directly.

..  code-block:: python

    import spox
    import ndonnx as ndx
    import spox.opset.ai.onnx.v20 as op

    x = ndx.array(shape=(2, 1), dtype=ndx.utf8)

    var = x.spox_var()

    # Freely use Spox
    var = op.string_split(var, delimiter=",")

While this is trivial in the case of data types with a direct correspondence in ONNX, arrays with nullable data types like :class:`ndonnx.nutf8` must first be decomposed into their constituent parts.
These can be accessed using the :func:`ndonnx.Array.values` property for the data and :func:`ndonnx.Array.null` for the boolean mask.

..  code-block:: python

    import ndonnx as ndx

    x = ndx.array(shape=(2, 1), dtype=ndx.nutf8)

    # Disassemble the nullable array into its data and null components
    data, null = x.values, x.null

    # data and null are regular ndonnx arrays
    print(data, null)
    # Array(dtype=Utf8) Array(dtype=Boolean)

    # Extract Spox Var
    data_var = data.spox_var()
    null_var = null.spox_var()

Spox to ndonnx
~~~~~~~~~~~~~~

Taking the ``var`` from the previous example, we can convert it back to an :class:`ndonnx.Array` using the :func:`ndonnx.from_spox_var` function.

..  code-block:: python

    array = ndx.from_spox_var(var)


Implementing a one-hot-encode function
-----------------------------------------------

Spox allows you to directly access the standard ONNX operators.
This means that while ndonnx does not directly expose a ``one_hot_encode`` function, since the ONNX standard has a `specialized operator <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.OneHotEncoder>`_ for this, we can implement a function that one-hot-encodes ndonnx arrays by going through Spox.

..  code-block:: python

    from typing import Iterable
    import ndonnx as ndx
    import spox.opset.ai.onnx.ml.v3 as ml

    def one_hot_encode(x: ndx.Array, categories: Iterable[str]) -> ndx.Array:
        return ndx.from_spox_var(ml.one_hot_encoder(x.spox_var(), cats_strings=categories))

We can use this as normal to export and run an ONNX model.

..  code-block:: python

    import onnxruntime as ort
    import ndonnx as ndx

    x = ndx.array(shape=("N",), dtype=ndx.utf8)
    y = one_hot_encode(x, ["a", "b", "c"])

    model = ndx.build({"x": x}, {"y": y})
    onnx.save(model, "one_hot_encode.onnx")

    sess = ort.InferenceSession("one_hot_encode.onnx")
    out, = sess.run(None, {"x": ["c", "b", "b", "a"]})
    print(out)
    # [[0. 0. 1.]
    # [0. 1. 0.]
    # [0. 1. 0.]
    # [1. 0. 0.]]

.. note::
    When working with Spox directly, you step outside of ndonnx. This means ndonnx is unable to propagate values as you might typically expect.
    :ref:`propagation` explains how you can ensure ndonnx can continue to eagerly propagate available values.
