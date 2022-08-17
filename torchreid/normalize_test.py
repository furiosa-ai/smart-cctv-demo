
import copy
import torch
import numpy as np
import onnx
import onnxruntime
from torch import nn
# from utils.reid_predictor import ReIdPredictor


def _normalize(x, mean, std):
    return (x - mean[None, :, None, None]) / std[None, :, None, None]


@torch.no_grad()
def create_norm_layer_weights(mean, std):
    n = len(mean)

    mean = np.array(mean)
    std = np.array(std)

    weight = 1 / std
    bias = -mean / std

    weight = weight[:, None, None, None]

    return weight, bias


@torch.no_grad()
def create_norm_layer(mean, std):
    n = len(mean)

    mean = torch.tensor(mean)
    std = torch.tensor(std)

    layer = nn.Conv2d(n, n, 1, groups=n)
    layer.weight = nn.parameter.Parameter(1 / std[:, None, None, None])
    layer.bias = nn.parameter.Parameter(-mean / std)

    return layer


def add_norm_layer(model, mean, std):
    return nn.Sequential(create_norm_layer(mean, std), model)


@torch.no_grad()
def main():
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # model_ori, model = [ReIdPredictor(cfg="configs/im_r50_softmax_256x128_amsgrad.yaml", weights="pretrained/resnet50_market_xent.pth.tar",
    #     output_type="np", batch_size=1, pad_batch=True).to("cpu").get_model() for _ in range(2)]

    norm_mean = torch.tensor(norm_mean)
    norm_std = torch.tensor(norm_std)

    model_ori = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
    model = nn.Sequential(create_norm_layer(norm_mean, norm_std), model_ori)

    """

    conv_layer = model
    conv_layer.bias = nn.parameter.Parameter(-torch.sum(conv_layer.weight * (norm_mean / norm_std), dim=(1,2,3)))
    conv_layer.weight /= norm_std
    """

    # norm_layer = create_norm_layer(norm_mean, norm_std)

    inp = torch.rand(1, 3, 256, 128) * 255

    y1 = model_ori(_normalize(inp, norm_mean, norm_std))
    y2 = model(inp)

    print(torch.nn.MSELoss()(y1, y2))


def create_initializer_tensor(
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


def onnx_add_dwconv(model, name, weight, bias, idx):
    graph = model.graph

    weight = weight.astype(np.float32)
    bias = bias.astype(np.float32)

    conv2_kernel_shape = (1, 1)
    conv2_pads = (0, 0, 0, 0)

    output_name = f"{graph.node[idx].input[0]}_normed"

    # Create the initializer tensor for the weights.
    conv2_W_initializer_tensor_name = f"{name}_W"
    conv2_W_initializer_tensor = create_initializer_tensor(
        name=conv2_W_initializer_tensor_name,
        tensor_array=weight,
        data_type=onnx.TensorProto.FLOAT)
    conv2_B_initializer_tensor_name = f"{name}_B"
    conv2_B_initializer_tensor = create_initializer_tensor(
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
    weight, bias = create_norm_layer_weights(norm_mean, norm_std)
    onnx_add_dwconv(model, "Normalize", weight, bias, 0)


def _infer_onnx(model, x):
    sess = onnxruntime.InferenceSession(model.SerializeToString())
    out = sess.run(None, {"input.1": x})[0]
    return out


@torch.no_grad()
def main2():
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    model_ori = onnx.load("im_r50_softmax_256x128_amsgrad.onnx")

    model = onnx.load("im_r50_softmax_256x128_amsgrad.onnx")
    onnx_add_norm_layer(model, norm_mean, norm_std)

    x = np.random.rand(1, 3, 256, 128).astype(np.float32)

    y_ori = _infer_onnx(model_ori, _normalize(x, np.array(norm_mean, dtype=np.float32),  np.array(norm_std, dtype=np.float32)))
    y = _infer_onnx(model, x)

    print(np.sum(np.power(y_ori - y, 2)))
    onnx.checker.check_model(model)


    #graph.node.insert(0, nms_node)

    onnx.save(model, "im_r50_softmax_256x128_amsgrad_norm.onnx")

    # model_def = onnx.shape_inference.infer_shapes(model)
    # onnx.checker.check_model(model)

    pass


if __name__ == "__main__":
    main2()
