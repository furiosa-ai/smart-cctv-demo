from pathlib import Path
import numpy as np
import onnx


def _create_norm_layer_weights(mean, std):
    mean = np.array(mean)
    std = np.array(std)

    weight = 1 / std
    bias = -mean / std

    weight = weight[:, None, None, None]

    return weight, bias


def _create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:

    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())

    return initializer_tensor


def _onnx_add_dwconv(model, name, weight, bias, idx):
    graph = model.graph

    weight = weight.astype(np.float32)
    bias = bias.astype(np.float32)

    conv2_kernel_shape = (1, 1)
    conv2_pads = (0, 0, 0, 0)

    output_name = f"{graph.node[idx].input[0]}_normed"

    # Create the initializer tensor for the weights.
    conv2_W_initializer_tensor_name = f"{name}_W"
    conv2_W_initializer_tensor = _create_initializer_tensor(
        name=conv2_W_initializer_tensor_name,
        tensor_array=weight,
        data_type=onnx.TensorProto.FLOAT)
    conv2_B_initializer_tensor_name = f"{name}_B"
    conv2_B_initializer_tensor = _create_initializer_tensor(
        name=conv2_B_initializer_tensor_name,
        tensor_array=bias,
        data_type=onnx.TensorProto.FLOAT)

    conv2_node = onnx.helper.make_node(
        name=name,
        op_type="Conv",
        inputs=[
            graph.node[idx].input[0], conv2_W_initializer_tensor_name,
            conv2_B_initializer_tensor_name
        ],
        outputs=[output_name],
        kernel_shape=conv2_kernel_shape,
        pads=conv2_pads,
        group=weight.shape[0]
    )

    graph.node[idx].input[0] = output_name

    graph.initializer.insert(0, conv2_W_initializer_tensor)
    graph.initializer.insert(0, conv2_B_initializer_tensor)
    graph.node.insert(idx, conv2_node)


def onnx_add_norm_layer(model, norm_mean, norm_std):
    weight, bias = _create_norm_layer_weights(norm_mean, norm_std)
    _onnx_add_dwconv(model, "Normalize", weight, bias, 0)
    onnx.checker.check_model(model)


class OnnxModelEdit:
    def __init__(self, in_file, out_suffix) -> None:
        self.in_file = Path(in_file)
        self.out_file = self.in_file.parent / (self.in_file.stem + out_suffix + self.in_file.suffix)
        self.model = None

    def __enter__(self):
        self.model = onnx.load(str(self.in_file))
        return self.out_file, self.model

    def __exit__(self ,type, value, traceback):
        onnx.checker.check_model(self.model)
        onnx.save(self.model, str(self.out_file))
