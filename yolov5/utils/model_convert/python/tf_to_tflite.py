# https://github.com/jkjung-avt/tensorrt_demos/issues/43
import os


def tf_saved_model_to_tflite(saved_model_dir, output_file, quantize_mode="int8", representative_dataset_gen=None):
    import tensorflow as tf

    # converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(tf_graph, input_names, output_names)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) #, input_names, output_names)

    if tf.__version__ >= '2.2.0':
        converter.experimental_new_converter = False

    if quantize_mode == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
    elif quantize_mode == 'int8':
        assert representative_dataset_gen is not None

        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.allow_custom_ops = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = representative_dataset_gen
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    open(output_file, 'wb').write(tflite_model)

    print(f"Wrote TFLite {output_file}")


# needs eager execution and tf2
def tf_frozen_graph_to_tflite(tf_graph, output_file, input_names, output_names,
                              quantize_mode="int8", representative_dataset_gen=None):
    from .tf1_to_tf2 import frozen_graph_to_saved_model

    saved_model_out = os.path.join(os.path.dirname(output_file), "saved_model")

    frozen_graph_to_saved_model(
        tf_graph,
        saved_model_out,
        input_names=input_names,
        output_names=output_names
    )

    tf_saved_model_to_tflite(
        saved_model_out,
        "/home/kevin/Documents/projects/svm-benchmark/export/out.tflite",
        quantize_mode=quantize_mode,
        representative_dataset_gen=representative_dataset_gen
    )
