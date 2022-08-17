
import torch
from torch import nn

from recognition.arcface_torch.backbones.resnet import ResNet_50


def export1():
    model = ResNet_50([112, 112])
    torch.onnx.export(model, torch.zeros(1, 3, 112, 112), "onnx/resnet50.onnx")
    

def main():
    # export2()
    export1()


if __name__ == "__main__":
    main()
