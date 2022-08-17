

def check_onnx_model(onnx_file):
    import onnx
    print("Checking ONNX model")
    model = onnx.load(onnx_file)

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    print("ONNX graph --BEGIN--")
    print(onnx.helper.printable_graph(model.graph))
    print("ONNX graph --END--")


def load_pb_model(pb_model_path):
    import tensorflow as tf

    with open(pb_model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        return graph_def
