import onnx
import os
from pathlib import Path
import torch
from torch import nn

from models.yolo import Model



def export_yolo(cfg_name="yolov5m", size=512, batch_size=None, quant=False):
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

    out_name = f"out/{out_name}_face_{suffix}.onnx"

    model = Model(f"models/{cfg_name}.yaml")
    model.model[-1].export_warboy = True
    model.eval()

    x = torch.zeros(batch_size, 3, size, size)

    torch.onnx.export(model, x, out_name, opset_version=12)
    
    # onnx_model = onnx.load(out_name)
    # onnx.checker.check_model(onnx_model)

    print("Done")


def main():
    cfgs = [
        # "yolov5m"
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

    cfgs = sorted([p.stem for p in Path(".").glob("models/yolov*.yaml")])

    for cfg in cfgs:
        export_yolo(cfg)

    # for b in range(4):
    #     export_yolo(batch_size=b+1, quant=True)
    # export_convnext_yolo()


if __name__ == "__main__":
    main()