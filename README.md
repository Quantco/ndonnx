# ndonnx

[![CI](https://img.shields.io/github/actions/workflow/status/quantco/ndonnx/ci.yml?style=flat-square&branch=main)](https://github.com/quantco/ndonnx/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/ndonnx/badge/?version=latest)](https://ndonnx.readthedocs.io/en/latest/?badge=latest)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/ndonnx?style=flat-square&logoColor=white&logo=conda-forge)](https://anaconda.org/conda-forge/ndonnx)
[![pypi](https://img.shields.io/pypi/v/ndonnx.svg?logo=pypi&logoColor=white)](https://pypi.org/project/ndonnx)

An ONNX-backed array library that is compliant with the [Array API](https://data-apis.org/array-api/) standard.

## Installation

Releases are available on PyPI and conda-forge.

```bash
# using pip
pip install ndonnx
# using conda
conda install ndonnx
# using pixi
pixi add ndonnx
```

## Development

You can install the package in development mode using:

```bash
git clone https://github.com/quantco/ndonnx
cd ndonnx

# For Array API tests
git submodule update --init --recursive

pixi shell
pre-commit run -a
pip install --no-build-isolation --no-deps -e .
pytest tests -n auto
```

## Quick start

`ndonnx` is an ONNX based python array library.

It has a couple of key features:

- It implements the [`Array API`](https://data-apis.org/array-api/) standard. Standard compliant code can be executed without changes across numerous backends such as like NumPy, JAX and now ndonnx.

  ```python
  import numpy as np
  import ndonnx as ndx
  import jax.numpy as jnp

  def mean_drop_outliers(a, low=-5, high=5):
      xp = a.__array_namespace__()
      return xp.mean(a[(low < a) & (a < high)])

  np_result = mean_drop_outliers(np.asarray([-10, 0.5, 1, 5]))
  jax_result = mean_drop_outliers(jnp.asarray([-10, 0.5, 1, 5]))
  onnx_result = mean_drop_outliers(ndx.asarray([-10, 0.5, 1, 5]))

  assert np_result == onnx_result.to_numpy() == jax_result == 0.75
  ```

- It supports ONNX export. This allows you persist your logic into an ONNX computation graph.

  ```python
  import ndonnx as ndx
  import onnx

  # Instantiate placeholder ndonnx array
  x = ndx.array(shape=("N",), dtype=ndx.float32)
  y = mean_drop_outliers(x)

  # Build and save ONNX model to disk
  model = ndx.build({"x": x}, {"y": y})
  onnx.save(model, "mean_drop_outliers.onnx")
  ```

  You can then make predictions using a runtime of your choice.

  ```python
  import onnxruntime as ort
  import numpy as np

  inference_session = ort.InferenceSession("mean_drop_outliers.onnx")
  prediction, = inference_session.run(None, {
      "x": np.array([-10, 0.5, 1, 5], dtype=np.float32),
  })
  assert prediction == 0.75
  ```

In the future we will be enabling a stable API for an extensible data type system. This will allow users to define their own data types and operations on arrays with these data types.

## Array API coverage

Array API compatibility tested against the official `array-api-tests` suite.
Missing coverage is tracked in the `skips.txt` file.
Contributions are welcome!

Run the tests with:

```bash
pixi run arrayapitests
```
