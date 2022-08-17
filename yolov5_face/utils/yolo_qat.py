import torch
from torch import nn
from pathlib import Path
from utils.qat import QATModel

"""
class DetectQAT(nn.Module):
    def __init__(self, detect_layer):
        super().__init__()

        self.nc = detect_layer.nc
        self.anchor = detect_layer.
        self.m = detect_layer.m

    def forward(self, x):
        return [l(x[i]) for i, l in enumerate(self.m)]
"""


class Yolov5QATModel(QATModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.export = False

    def forward(self, x, augment=False):
        x = self.quant(x)
        x = self.model(x, augment=augment)
        if self.training or self.export:
            if isinstance(x, list):
                x = [self.dequant(t) for t in x]
            else:
                x = self.dequant(x) 
        return x



def _test():
    from models.experimental import attempt_load
    from models.yolo import Model
    # model = attempt_load("weights/yolov5m-face.pt", "cpu")
    cfg = "models/yolov5m_relu.yaml"
    model = Model(cfg).eval()
    model = Yolov5QATModel(model)

    # fuse_model(model)
    # return

    x = torch.rand(1, 3, 512, 512)
    y = model(x)
    print(y)

    model.export = True
    model.model.model[-1].export_warboy = True

    model.apply(torch.quantization.disable_observer)
    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    torch.onnx.export(model.model, x, str(Path("out") / (Path(cfg).stem + ".onnx")), opset_version=13)

    quantized_model = torch.quantization.convert(model.eval(), inplace=False)
    quantized_model.eval()
    quantized_model(x)

    torch.onnx.export(quantized_model, x, str(Path("out") / (Path(cfg).stem + "_quant.onnx")))


if __name__ == "__main__":
    _test()
