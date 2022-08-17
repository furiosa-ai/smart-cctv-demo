# output.pb
import os


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True, output_file=None):
    import tensorflow as tf
    from tensorflow.python.framework.graph_util import convert_variables_to_constants

    try:
        tf = tf.compat.v1
        # tf.disable_eager_execution()
    except:
        pass

    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)

        if output_file is not None:
            tf.train.write_graph(frozen_graph, os.path.dirname(output_file), os.path.basename(output_file),
                                 as_text=False)

        print("%d ops in the final graph." % len(frozen_graph.node))

        return frozen_graph


def freeze_keras_model(model_build_func, output_file=None):
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    K.set_learning_phase(0)

    model = model_build_func()

    return freeze_session(
        K.get_session(), output_names=[out.op.name for out in model.outputs], output_file=output_file
    ), model


def freeze_checkpoint(model_folder, output_names,
                      output_file='frozen-graph.pb',
                      rename_outputs=None,
                      fix_batchnorm=False
                      ):
    import tensorflow as tf

    try:
        tf = tf.compat.v1
        tf.disable_eager_execution()
    except:
        pass

    # Load checkpoint
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # Devices should be cleared to allow Tensorflow to control placement of
    # graph when loading on different machines
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=True)

    graph = tf.get_default_graph()

    onames = output_names

    # https://stackoverflow.com/a/34399966/4190475
    if rename_outputs is not None:
        nnames = rename_outputs.split(',')
        with graph.as_default():
            for o, n in zip(output_names, nnames):
                _out = tf.identity(graph.get_tensor_by_name(o + ':0'), name=n)
            onames = nnames

    input_graph_def = graph.as_graph_def()

    if fix_batchnorm:
        # fix batch norm nodes
        for node in input_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, input_checkpoint)

        freeze_session(sess, output_names=onames, output_file=output_file)

