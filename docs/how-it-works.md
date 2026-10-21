# How ndonnx works

An [ndonnx.Array][] is a regular Python object that traces which operations were applied to it [^1].
For instance, adding `1` to an existing array object does not compute a value; it records the operation.
The resulting array thereby maintains the _lineage_ of the computation.
Consequently, any array object can be used to reconstruct the full computational graph that created it.

This design follows from the target format: an ONNX model is a static computational graph that is executed later by an inference runtime.
ndonnx therefore records the computation rather than performing it.

An array's computational graph contains almost all the information needed to create an ONNX model.
The only missing ingredients are the names of the inputs and outputs of the graph.
When building the ONNX model, this information is provided to [ndonnx.build][] as `dict[str, ndonnx.Array]` objects.

ndonnx furthermore distinguishes two kinds of array: constant arrays, whose value is known at build time, and arrays whose value depends on the graph inputs and is therefore known only at inference time.
Constant arrays translate to `Constant` nodes in an ONNX graph.
The lineage of an input-dependent array eventually points to an [ndonnx.argument][] array: a special array instance marking an input of the computational graph.
Only arrays created with [ndonnx.argument][] may be used as `inputs` in [ndonnx.build][].
Constant arrays are constructed using [creation functions](api/creation-functions.md) such as [ndonnx.asarray][] or [ndonnx.ones][].
Even if the concrete value of an array may only be known at inference time, any ndonnx array object always has a known data type and dimensionality.
Each individual dimension may be static (e.g. `2`), symbolic (e.g. `"N"`), or unknown (`None`).
This allows ndonnx to perform full type and shape inference at every operation which allows for precise error reporting and interactive debugging.

The following snippet puts all of the above information into action:

```python
import ndonnx as ndx
import onnx

a = ndx.argument(shape=("N",), dtype=ndx.int64)  # (1)!
b = ndx.asarray([1, 2])  # (2)!

result = (a + b) * 2  # (3)!

model_proto = ndx.build({"a": a}, {"result": result})  # (4)!

onnx.save_model(model_proto, "example.onnx")  # (5)!
```

1. Create an argument array that will later be used a s a graph input
2. Create an array with constant values. The data type is inferred according to the array-api specification.
3. Perform any operations supported by the array-api. The `result` object contains the full computational graph with all constant values.
4. Export the computational graph to an `onnx.ModelProto` object.
5. Serialize the model to a file.

[^1]: ndonnx is built on top of [spox](https://github.com/Quantco/spox) which technically does the tracing.
