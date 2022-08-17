import onnx
import os
import torch
from torch import nn
from models.backbones.efficientnet import EfficientNet

from models.yolo import Model
from utils.global_config import global_config, reset_global_cfg


class STD(nn.Module):
    # Focus wh information into c-space
    def __init__(self):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return torch.cat([x[..., ::2, ::2], x[..., ::2, 1::2], x[..., 1::2, ::2], x[..., 1::2, 1::2]], 1)
        # return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


def test_std():
    model = STD()

    x = torch.zeros(1, 3, 256, 256)

    torch.onnx.export(model, x, "std.onnx", opset_version=12)


def quantize_model_random(onnx_file, outpath=None):
    from furiosa.quantizer.frontend.onnx import post_training_quantization_with_random_calibration
    from furiosa.quantizer.frontend.onnx.quantizer.utils import QuantizationMode

    model = onnx.load_model(onnx_file)
    quant_model = post_training_quantization_with_random_calibration(model, True, True, QuantizationMode.DFG, 1)

    if outpath is None:
        outpath = f"{os.path.splitext(onnx_file)[0]}_i8.onnx"
    onnx.save_model(quant_model, f'{outpath}')

    return True


def export_yolo(cfg_name="yolov5m", size=512, batch_size=None, quant=False):
    reset_global_cfg()
    # use_relu = False

    # global_config["use_add"] = True

    # cfg_name = "yolov5m"
    # cfg_name = "yolov5m_warboy"
    out_name = cfg_name

    # if use_relu:
    #     global_config["force_act"] = nn.ReLU()
    #     out_name += "_relu"

    suffix = "_".join(
        [str(size)] +
        ([f"b{batch_size}"] if batch_size is not None else [])
    )

    if batch_size is None:
        batch_size = 1

    out_name = f"out/test/{out_name}_{suffix}.onnx"

    model = Model(f"models/{cfg_name}.yaml")
    model.model[-1].set_mode("export")
    model.eval()

    x = torch.zeros(batch_size, 3, size, size)

    torch.onnx.export(model, x, out_name, opset_version=12)
    
    if quant:
        quantize_model_random(out_name, out_name)

    # onnx_model = onnx.load(out_name)
    # onnx.checker.check_model(onnx_model)

    print("Done")


def export_efficientnet_yolo():
    EfficientNet.export = True
    # use_relu = False

    size = 512
    # global_config["export_skip_postproc"] = True
    # global_config["use_add"] = True

    for i in range(0, 7+1):
        cfg_name = f"yolov5s_efficientnet_b{i}"
        # cfg_name = "yolov5m_warboy"
        out_name = cfg_name

        # if use_relu:
        #     global_config["force_act"] = nn.ReLU()
        #     out_name += "_relu"

        out_name = f"out/yolov5m_efficientnet/{out_name}_{size}.onnx"

        model = Model(f"models/effnet/{cfg_name}.yaml")
        model.model[-1].set_mode("export")

        model.eval()

        x = torch.zeros(1, 3, size, size)
        _ = model(x)

        torch.onnx.export(model, x, out_name, opset_version=12)

        """
        onnx_model = onnx.load(out_name)
        onnx.checker.check_model(onnx_model)

        break
        """

    print("Done")


def export_convnext_yolo():
    EfficientNet.export = True
    # use_relu = False

    size = 640
    global_config["export_skip_postproc"] = True
    # global_config["use_add"] = True

    cfg_name = f"convnext/yolov5_ConvNeXt_tiny_warboy"
    # cfg_name = "yolov5m_warboy"
    out_name = os.path.basename(cfg_name)

    # if use_relu:
    #     global_config["force_act"] = nn.ReLU()
    #     out_name += "_relu"

    out_name = f"out/convnext/{out_name}_{size}.onnx"

    model = Model(f"models/{cfg_name}.yaml")
    model.eval()

    x = torch.zeros(1, 3, size, size)

    torch.onnx.export(model, x, out_name, opset_version=12)

    """
    onnx_model = onnx.load(out_name)
    onnx.checker.check_model(onnx_model)

    break
    """

    print("Done")


def main():
    # export_efficientnet_yolo()

    cfgs = [
        "yolov5m_warboy"
        # "yolov5l",
        # "yolov5x",
        # "yolov5s",
        # "yolov5s_add",
        # "yolov5s_convtrans",
        # "yolov5s_relu",
        # "yolov5m",
        # "yolov5m_add",
        # "yolov5m_convtrans",
        # "yolov5m_relu",
    ]

    for cfg in cfgs:
        export_yolo(cfg)

    # for b in range(4):
    #     export_yolo(batch_size=b+1, quant=True)
    # export_convnext_yolo()


if __name__ == "__main__":
    main()