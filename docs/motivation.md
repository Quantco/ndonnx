# Why we created ndonnx

ndonnx is an implementation of the [Array
API](https://data-apis.org/array-api/latest/) standard backed by
[ONNX](https://onnx.ai/).

We believe that ONNX is a flexible, performant way to take machine
learning models from research to production. However, tooling for
converting models from the framework you have trained it in to ONNX has
traditionally been error-prone and challenging to maintain.

Existing tooling revolves around [ONNX converter
libraries](https://onnx.ai/onnx/intro/converters.html). The fundamental
problem with this approach is that inference logic must be **duplicated**.
It is first expressed in an API suitable for training using host
libraries like NumPy or PyTorch and then reimplemented again in terms of
ONNX operators using a library like
[Spox](https://github.com/quantco/spox). This is error-prone and hard to
maintain as the host library evolves and the latest API changes need to
be mirrored.

We identified that an [Array
API](https://data-apis.org/array-api/latest/) standard compliant
interface to ONNX would solve these problems by uniting these APIs while
retaining complete development flexibility.
