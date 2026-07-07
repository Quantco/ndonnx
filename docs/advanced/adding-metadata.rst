Adding metadata to an artifact
==============================

The ONNX specification allows for adding metadata to an artifact.
Ndonnx does not offer dedicated tooling to write this data directly, but it can be achieved using the ``onnx`` package as shown below.
Metadata is stored as a string-to-string mapping.
Ndonnx stores its own metadata under the ``"ndonnx_schema"`` key, which should not be overwritten.

..  code-block:: python

    import ndonnx as ndx
    import onnx


    def update_metadata(model_proto: onnx.ModelProto, metadata: dict[str, str]):
        """Update the model's metadata."""
        # Ensure that existing items are kept
        existing = {el.key: el.value for el in model_proto.metadata_props}
        onnx.helper.set_metadata_props(model_proto, existing | metadata)

    a = ndx.argument(shape=("N",), dtype=ndx.int64)
    model_proto = ndx.build({"in": a}, {"out": a})
    update_metadata(model_proto, {"foo": "bar"})
    assert "foo" in [el.key for el in model_proto.metadata_props]
