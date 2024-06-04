Why we created ndonnx
=====================

ndonnx is an implementation of the `Array API <https://data-apis.org/array-api/latest/>`_ standard backed by `ONNX <https://onnx.ai/>`_.

We believe that ONNX is a flexible, performant way to take machine learning models from research to production.
However, tooling for converting models from the framework you have trained it in to ONNX has traditionally been error-prone and challenging to maintain.

Existing tooling revolves around `ONNX converter libraries <https://onnx.ai/onnx/intro/converters.html>`_.
The fundamental problem with this approach that inference logic must be **duplicated**. It is first expressed in an API suitable for training using host libraries like NumPy or PyTorch and then reimplemented again in terms of ONNX operators using a library like `Spox <https://github.com/quantco/spox>`_. This is error-prone and hard to maintain as the host library evolves and the latest API changes need to be mirrored.

We identified that an `Array API <https://data-apis.org/array-api/latest/>`_ standard compliant interface to ONNX would solve these problems by uniting these APIs while retaining complete development flexibility. This motivated us to create ndonnx.

Practical features for expressing ONNX computation graphs
---------------------------------------------------------
Beyond solving the code duplication problem outlined above, ndonnx has several features that make it a compelling choice for expressing ONNX computation graphs:

#. **ndonnx is an array library**. Features like type promotion, advanced slicing, and other high-level functions you can come to expect from array libraries like NumPy and PyTorch are present in ndonnx. This leads to more concise, less buggy converters compared to using lower level libraries that only expose the `ONNX operator set <https://github.com/onnx/onnx/blob/main/docs/Operators.md>`_.

#. :doc:`Interoperability </spox/spoxintegration>`. Parts of the ONNX standard are unsuitable for the Array abstraction such as the LSTM operator. ndonnx can be used incrementally where useful and where not, you can drop down to the ONNX operator set since ndonnx builds on `Spox <https://github.com/quantco/spox>`_.

#. **Constant folding**. Arrays whose values can be determined before even building the ONNX graph are eagerly computed. This means that the exported ONNX computational graph is as lean as possible.

#. :doc:`User-defined data types (experimental) </experimental/experimental>`. ONNX specifies a narrow set of C-inspired data types. When defining Machine Learning pipelines, domain-specific types like datetimes and nullability are useful. We expose an experimental API for defining "struct" types as a user. We were inspired by NumPy's `structured arrays <https://numpy.org/doc/stable/user/basics.rec.html>`_ and Arrow's `StructType <https://arrow.apache.org/docs/python/generated/pyarrow.StructType.html>`_ and use this feature extensively.
