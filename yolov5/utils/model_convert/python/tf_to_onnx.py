from .util.util import check_onnx_model


def tf_to_onnx(tf_graph, output_file, opset=None, check_model=False, input_names=None, output_names=None):
    import tf2onnx
    import tensorflow as tf

    if isinstance(tf_graph, tf.GraphDef):
        print("Converting frozen graphdef to graph")
        tf.reset_default_graph()
        tf.import_graph_def(tf_graph)
        tf_graph = tf.get_default_graph()

    if input_names is not None:
        input_names = [f"import/{n}:0" for n in input_names]
    else:
        assert False

    if output_names is not None:
        output_names = [f"import/{n}:0" for n in output_names]
    else:
        assert False

    onnx_graph = tf2onnx.tfonnx.process_tf_graph(tf_graph, opset=opset,
                                                 input_names=input_names, output_names=output_names)

    model_proto = onnx_graph.make_model("test")
    with open(output_file, "wb") as f:
        f.write(model_proto.SerializeToString())

    if check_model:
        check_onnx_model(output_file)
