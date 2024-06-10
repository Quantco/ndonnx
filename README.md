# ndonnx

[![CI](https://github.com/quantco/ndonnx/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/quantco/ndonnx/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-latest-success?branch=main&style=plastic)](https://docs.dev.quantco.cloud/qc-github-artifacts/Quantco/ndonnx/latest/index.html)
[![Quetz](https://img.shields.io/badge/quetz-ndonnx-success?style=plastic)](https://conda.prod.quantco.cloud/channels/qc-internal/packages/ndonnx)

An ONNX-backed array library that is compliant with the [Array API](https://data-apis.org/array-api/) standard.

## Installation

You can install `ndonnx` using `conda`:

```bash
# using conda
conda install ndonnx
# using micromamba
micromamba install ndonnx
# using pixi
pixi add ndonnx
```

## Development

You can install the package in development mode using:

```bash
git clone https://github.com/quantco/ndonnx
cd ndonnx
git submodule update --init --recursive

pixi run pre-commit-install
pixi run postinstall
pixi run test
```

## Quick start

`ndonnx` is an ONNX based python array library.

It has a couple of key features:

- It implements the [`Array API`](https://data-apis.org/array-api/) standard. Standard compliant code can be executed without changes across numerous backends such as like `NumPy`, `JAX` and now `ndonnx`.

  ```python
  import numpy.array_api as npx
  import ndonnx as ndx
  from jax.experimental import array_api as jxp

  def mean_drop_outliers(a, low=-5, high=5):
      xp = a.__array_namespace__()
      return xp.mean(a[(low < a) & (a < high)])

  arr = [-12.12, 1.12, 2.12, 2.13, 123.,]

  np_result = mean_drop_outliers(npx.asarray(arr))
  jax_result = mean_drop_outliers(jxp.asarray(arr))
  ndx_result = mean_drop_outliers(ndx.asarray(arr))
  print(np_result)  # 1.79
  print(jax_result)  # 1.79
  print(ndx_result) # Array(1.79, dtype=ndx.Float64)
  assert np_result == ndx_result.to_numpy()
  ```

- It supports ONNX export. This allows you persist your logic into an ONNX computation graph for convenient and performant inference.

  ```python
  import onnx
  import ndonnx as ndx

  a = ndx.array(shape=("N",), dtype=ndx.float64)
  b = ndx.array(shape=("N",), dtype=ndx.float64)
  out = a[:2] + b[:2]
  model_proto = ndx.build({"a": a, "b": b}, {"c": out})
  onnx.save(model_proto, "model.onnx")

  # Having serialised your model to disk, perform
  # inference using a runtime of your choosing.
  import onnxruntime as ort
  import numpy as np
  inference_session = ort.InferenceSession("model.onnx")
  prediction, = inference_session.run(None, {
      "a": np.array([1, 2, 3], dtype=np.float64),
      "b": np.array([4, 5, 6], dtype=np.float64),
  })
  print(prediction)  # array([5., 7.])
  ```

In the future we will be enabling a stable API for an extensible data type system. This will allow users to define their own data types and operations on arrays with these data types.

## Array API coverage

Array API compatibility is tracked in the array-api coverage test suite in `api-coverage-tests`. Missing coverage is tracked in the `skips.txt` file. Contributions are welcome!

Summary(1119 total):

- 898 passed
- 210 failed
- 11 deselected

Run the tests with:

```bash
ARRAY_API_TESTS_MODULE=ndonnx pytest array_api_tests/ --json-report --json-report-file=api-coverage-tests.json -n auto
```
