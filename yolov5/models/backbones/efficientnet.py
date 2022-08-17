from turtle import forward
from torch import nn
from models.backbones.util.efficientnet.model import EfficientNet as EN



class EfficientNet(nn.Module):
    output_channels = {
        "efficientnet-b0": [40, 112, 320],
        "efficientnet-b1": [40, 112, 320],
        "efficientnet-b2": [48, 120, 352],
        "efficientnet-b3": [48, 136, 384],
        "efficientnet-b4": [56, 160, 448],
        "efficientnet-b5": [64, 176, 512],
        "efficientnet-b6": [72, 200, 576],
        "efficientnet-b7": [80, 224, 640],
    }

    export = False

    def __init__(self, name) -> None:
        super().__init__()

        self.model = EN.from_name(name)

        if EfficientNet.export:
            self.model.set_swish(False)

    def forward(self, x):
        feat = self.model.extract_endpoints(x)
        feat = list(feat.values())
        feat = feat[-4:-1]

        return feat


def _test():
    import torch

    x = torch.zeros(1, 3, 512, 512)

    for i in range(0, 7+1):
        model = EfficientNet(f"efficientnet-b{i}")

        y = model(x)

        print(f"efficientnet-b{i}:", [t.shape[1] for t in y])


if __name__ == "__main__":
    _test()
