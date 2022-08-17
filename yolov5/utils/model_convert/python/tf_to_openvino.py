import os

# takes frozen graph
from .model_to_openvino import model_to_openvino


def tf_to_openvino(tf_graph, output_path, **kwargs):
    import tensorflow as tf

    output_dir = os.path.dirname(output_path)
    output_pb_name = os.path.basename(output_path) + ".pb"

    tf.train.write_graph(tf_graph, output_dir, output_pb_name, as_text=False)
    model_to_openvino(os.path.join(output_dir, output_pb_name), output_path, **kwargs)
