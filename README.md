# ndonnx

[![CI](https://img.shields.io/github/actions/workflow/status/quantco/ndonnx/ci.yml?style=flat-square&branch=main)](https://github.com/quantco/ndonnx/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/ndonnx/badge/?version=latest)](https://ndonnx.readthedocs.io/en/latest/?badge=latest)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/ndonnx?style=flat-square&logoColor=white&logo=conda-forge)](https://anaconda.org/conda-forge/ndonnx)
[![pypi](https://img.shields.io/pypi/v/ndonnx.svg?logo=pypi&logoColor=white)](https://pypi.org/project/ndonnx)

An ONNX-backed implementation of the [Array API](https://data-apis.org/array-api/) standard.

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
```

## Quick start

The Array API standard standardizes a subset of the NumPy API.
This allows for writing backend-agnostic code such as the following functions:

```python
def mean_drop_outliers(a, low=-5, high=5):
   xp = a.__array_namespace__()
   return xp.mean(a[(low < a) & (a < high)])
```

The `mean_drop_outliers` function may be called with any array object that implements the Array API such as the array objects from `numpy`, `jax.numpy` and `ndonnx`:

```python
import numpy as np
import ndonnx as ndx
import jax.numpy as jnp

np_result = mean_drop_outliers(np.asarray([-10, 0.5, 1, 5]))
jax_result = mean_drop_outliers(jnp.asarray([-10, 0.5, 1, 5]))
onnx_result = mean_drop_outliers(ndx.asarray([-10, 0.5, 1, 5]))

assert np_result == onnx_result.unwrap_numpy() == jax_result == 0.75
```

Arrays in ndonnx may have constant values (as in the above example) or be placeholders for inputs of a computational graph.
By using such placeholder arrays it is possible to export any array-api compliant code to ONNX.
Exporting `mean_drop_outliers` for instance may be achieved as follows:

```python
import ndonnx as ndx
import onnx

# Instantiate placeholder ndonnx array
x = ndx.Array(shape=("N",), dtype=ndx.float32)
y = mean_drop_outliers(x)

# Build and save ONNX model to disk
model = ndx.build({"x": x}, {"y": y})
onnx.save(model, "mean_drop_outliers.onnx")
```

The created artifact is a regular ONNX model which may be loaded and used by runtimes such as `onnxruntime`:

```python
import onnxruntime as ort
import numpy as np

inference_session = ort.InferenceSession("mean_drop_outliers.onnx")
prediction, = inference_session.run(None, {
    "x": np.array([-10, 0.5, 1, 5], dtype=np.float32),
})
assert prediction == 0.75
```

## Array API compliance

Ndonnx strives to be fully array-api compliant.
Any violation of the standard is considered a bug.

The upstream test suite may be executed as follows:

```bash
pixi run arrayapitests
```
