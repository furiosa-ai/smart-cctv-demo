import argparse
import onnx
import os
import torch
from torch import nn
from models.backbones.efficientnet import EfficientNet

from models.yolo import Model
from utils.global_config import global_config, reset_global_cfg


def quantize_model_random(onnx_file, outpath=None):
    from furiosa.quantizer.frontend.onnx import post_training_quantization_with_random_calibration
    from furiosa.quantizer.frontend.onnx.quantizer.utils import QuantizationMode

    model = onnx.load_model(onnx_file)
    quant_model = post_training_quantization_with_random_calibration(model, True, True, QuantizationMode.DFG, 1)

    if outpath is None:
        outpath = f"{os.path.splitext(onnx_file)[0]}_i8.onnx"
    onnx.save_model(quant_model, f'{outpath}')

    return True


def export_yolo(cfg_name="yolov5m", size=512, batch_size=None, num_classes=80, quant=False):
    reset_global_cfg()
    # use_relu = False

    # global_config["use_add"] = True

    # cfg_name = "yolov5m"
    # cfg_name = "yolov5m_warboy"
    out_name = cfg_name

    # if use_relu:
    #     global_config["force_act"] = nn.ReLU()
    #     out_name += "_relu"

    if batch_size is None:
        batch_size = 1

    suffix = (
        [f"c{num_classes}"] +
        [f"s{size}"] +
        ([f"b{batch_size}"] if batch_size != 1 else [])
    )

    suffix = "_".join([str(s) for s in suffix])

    out_name = f"out/onnx/{out_name}_{suffix}.onnx"

    model = Model(f"models/{cfg_name}.yaml", nc=num_classes)
    model.model[-1].set_mode("export")
    model.eval()

    x = torch.zeros(batch_size, 3, size, size)

    torch.onnx.export(model, x, out_name, opset_version=12)
    
    if quant:
        out_name_i8 = out_name.replace(".onnx", "_i8.onnx")
        quantize_model_random(out_name, out_name_i8)

    # onnx_model = onnx.load(out_name)
    # onnx.checker.check_model(onnx_model)

    print("Done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg")
    parser.add_argument("-s", "--size", type=int, default=512)
    parser.add_argument("-c", "--num_classes", type=int, default=80)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    args = parser.parse_args()

    # export_efficientnet_yolo()

    # cfgs = [
        # "yolov5m_warboy"
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
    # ]

    # for cfg in cfgs:
    export_yolo(cfg_name=args.cfg, size=args.size, batch_size=args.batch_size, num_classes=args.num_classes, quant=True)

    # for b in range(4):
    #     export_yolo(batch_size=b+1, quant=True)
    # export_convnext_yolo()


if __name__ == "__main__":
    main()