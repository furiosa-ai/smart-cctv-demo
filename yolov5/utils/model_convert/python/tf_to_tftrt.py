# https://github.com/jkjung-avt/tensorrt_demos/issues/43

# output_names: string array containing output node names
def tf_to_tftrt(tf_graph, output_file, outputs, batch_stream=None, precision="FP32"):
    # import tensorflow.contrib.tensorrt as trt

    print("Creating inference graph")

    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    
    converter = trt.TrtGraphConverter(
        input_graph_def=tf_graph,
        # nodes_blacklist=['logits', 'classes']
    )  # output nodes

    trt_graph = converter.convert()

    with open(output_file, 'wb') as f:
        f.write(trt_graph.SerializeToString())

    """
    trt_graph = trt.create_inference_graph(
        input_graph_def=tf_graph,
        outputs=outputs,
        max_batch_size=1,
        #max_workspace_size_bytes=1 << 25,
        precision_mode=precision,
        #minimum_segment_size=50
    )
    """

    print("Done")
