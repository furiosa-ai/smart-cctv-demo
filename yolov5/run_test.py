
import torch
from models.yolo import Model


def main():
    # use_relu = False

    size = 512
    # global_config["export_skip_postproc"] = True
    # global_config["use_add"] = True

    # cfg_name = "yolov5m_efficientnet_b2"
    # cfg_name = "models/convnext/yolov5_ConvNeXt_tiny_warboy.yaml"
    cfg_name = "models/convnext/yolov5_ConvNeXt_tiny_warboy_focus.yaml"

    model = Model(cfg_name)
    model.eval()

    x = torch.zeros(1, 3, size, size)

    y = model(x)

    print(y)


if __name__ == "__main__":
    main()