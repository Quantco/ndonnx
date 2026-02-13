# Welcome to ndonnx's documentation!

ndonnx is an ONNX-backed Python array library that implements the [Array
API standard](https://data-apis.org/array-api/latest/).
ndonnx can be use to export large existing NumPy code bases to ONNX with minimal effort.
It was designed from the ground up with the following priorities in mind:

- Correctness
- First class developer experience
- Seamless interoperability with [spox](https://spox.readthedocs.io/en/latest/index.html)

### Example

```python
import ndonnx as ndx
import numpy as np
import onnx

def mean_drop_outliers(a, low=-5, high=5):
    xp = a.__array_namespace__()
    return xp.mean(a[(low < a) & (a < high)])

# run with Python
print(mean_drop_outliers(np.asarray([-10, 0.5, 1, 4])))

# export to ONNX
a = ndx.argument(shape=("N",), dtype=ndx.float64)
result = mean_drop_outliers(a)

model_proto = ndx.build({"a": a}, {"result": result})
onnx.save_model(model_proto, "model.onnx")
```
