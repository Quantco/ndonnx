import numpy as np
import spox
import spox.opset.ai.onnx.v21 as op
import onnx
import onnxruntime as ort

def test_sign():
    arg = spox.argument(spox.Tensor(shape=(None, None), dtype=np.int64, ))
    model_proto = spox.build({"input": arg}, {"result": op.sign(arg)})
    onnx.save(model_proto, "test.onnx")

    sess = ort.InferenceSession("test.onnx")
    inp = np.asarray([[2147483649, 2147483649]])
    res, = sess.run(None, {"input": inp})
    np.testing.assert_array_equal(res, np.sign(inp))
