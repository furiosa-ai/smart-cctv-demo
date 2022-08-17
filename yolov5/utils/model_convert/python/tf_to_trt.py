# https://github.com/jkjung-avt/tensorrt_demos/issues/43

import os


from .tf_to_onnx import tf_to_onnx
from .onnx_to_trt import onnx_to_trt


def tf_to_trt(tf_graph, output_file, batch_stream=None, precision=None, onnx_opset=None, check_onnx_model=False,
              input_shape=None, input_names=None, output_names=None):
    import tensorflow as tf

    if isinstance(tf_graph, tf.GraphDef):
        print("Converting frozen graphdef to graph")
        tf.reset_default_graph()
        tf.import_graph_def(tf_graph)
        tf_graph = tf.get_default_graph()

    output_file_onnx = os.path.splitext(output_file)[0] + ".onnx"

    print("Converting to ONNX")
    tf_to_onnx(tf_graph, output_file_onnx, opset=onnx_opset, check_model=check_onnx_model,
               input_names=input_names, output_names=output_names)

    print("Converting to TensorRT")
    onnx_to_trt(output_file_onnx, output_file, batch_stream=batch_stream, precision=precision, input_shape=input_shape)
