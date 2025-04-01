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

Extract a :class:`spox.Var` from an :class:`ndonnx.Array` using the :func:`ndonnx.Array.unwrap_spox` method. This allows you to use Spox to manipulate the ONNX graph more directly.

..  code-block:: python

    import spox
    import ndonnx as ndx
    import spox.opset.ai.onnx.v20 as op

    x = ndx.argument(shape=(2, 1), dtype=ndx.utf8)

    var = x.unwrap_spox()

    # Freely use Spox
    var = op.string_split(var, delimiter=",")

While this is trivial in the case of data types with a direct correspondence in ONNX, arrays with nullable data types like :class:`ndonnx.nutf8` must first be decomposed into their constituent parts.
This is achieved via the `:meth:ndonnx.Array.disassemble` method.

..  code-block:: python

    import ndonnx as ndx

    x = ndx.argument(shape=(2, 1), dtype=ndx.nutf8)

    data_var = x.disassemble()["data"]
    null_var = x.disassemble()["null"]


Spox to ndonnx
~~~~~~~~~~~~~~

Taking the ``var`` from the previous example, we can convert it back to an :class:`ndonnx.Array` using the :func:`ndonnx.asarray` function.

..  code-block:: python

    array = ndx.asarray(var)


Implementing a one-hot-encode function
-----------------------------------------------

Spox allows you to directly access the standard ONNX operators.
This means that while ndonnx does not directly expose a ``one_hot_encode`` function, since the ONNX standard has a `specialized operator <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.OneHotEncoder>`_ for this, we can implement a function that one-hot-encodes ndonnx arrays by going through Spox.

..  code-block:: python

    from typing import Iterable
    import ndonnx as ndx
    import spox.opset.ai.onnx.ml.v3 as ml

    def one_hot_encode(x: ndx.Array, categories: Iterable[str]) -> ndx.Array:
        return ndx.asarray(ml.one_hot_encoder(x.unwrap_spox(), cats_strings=categories))

We can use this as normal to export and run an ONNX model.

..  code-block:: python

    import onnxruntime as ort
    import ndonnx as ndx

    x = ndx.argument(shape=("N",), dtype=ndx.utf8)
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
    See :ref:`propagation` on how to ensure that value propagation is functioning even when using custom ONNX operators.
