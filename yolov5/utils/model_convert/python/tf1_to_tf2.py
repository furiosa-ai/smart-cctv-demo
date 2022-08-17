
# provide dict for input_names or output_names if you want to rename
# graph must be frozen !!!
def tf_frozen_graph_to_saved_model(tf_graph, export_dir, input_names, output_names):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from tensorflow.python.saved_model import signature_constants
    from tensorflow.python.saved_model import tag_constants

    from .util.util import load_pb_model

    if not isinstance(input_names, dict):
        input_names = {v: v for v in input_names}

    if not isinstance(output_names, dict):
        output_names = {v: v for v in output_names}

    if isinstance(tf_graph, str):
        tf_graph = load_pb_model(tf_graph)

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    sigs = {}
    with tf.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(tf_graph, name="")
        g = tf.get_default_graph()

        input_sig = {k: g.get_tensor_by_name(v + ":0") for k, v in input_names.items()}
        output_sig = {k: g.get_tensor_by_name(v + ":0") for k, v in output_names.items()}

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
                input_sig, output_sig)

        builder.add_meta_graph_and_variables(sess,
                                             [tag_constants.SERVING],
                                             signature_def_map=sigs)
        builder.save()

    print(f"Wrote saved model {export_dir}")