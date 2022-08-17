from pathlib import Path
from typing import List, Tuple, Union
import torch
from torch import nn
from torch.quantization import QuantStub, DeQuantStub



def model_layer_map(model, layer_type, replace_func):
    for name, module in model._modules.items():
        if isinstance(module, layer_type):
            model._modules[name] = replace_func(module)
        else:
            model_layer_map(module, layer_type, replace_func)

    return model


def fuse_model(model):
    torch.quantization.fuse_modules

    for name, module in model._modules.items():
        if isinstance(module, nn.Conv2d):
            print("conv")
        else:
            fuse_model(module)

    
def prepare_qat(model):
    if not isinstance(model, (
        nn.Sequential, nn.ModuleList, nn.quantized.FloatFunctional, nn.Conv2d, 
        nn.BatchNorm2d, nn.ReLU, nn.Identity, nn.MaxPool2d, nn.Upsample, QuantStub, DeQuantStub)):
        model.prepare_qat()

    for name, module in model._modules.items():
        prepare_qat(module)


class QATConcat(nn.Module):
    def __init__(self):
        super().__init__()

        self.f_cat = None

    def prepare_qat(self):
        self.f_cat = nn.quantized.FloatFunctional()

    def forward(self, tensors: List[torch.Tensor], dim: int=0):
        if self.f_cat is None:
            return torch.cat(tensors, dim)
        else:
            return self.f_cat.cat(tensors, dim)


class QATAdd(nn.Module):
    def __init__(self):
        super().__init__()

        self.f_add = None

    def prepare_qat(self):
        self.f_add = nn.quantized.FloatFunctional()

    def forward(self, a, b):
        if self.f_add is None:
            return a + b
        else:
            return self.f_add.add(a, b)


class QATModel(nn.Module):
    def __init__(self, model, prep_qat=True):
        super().__init__()

        if prep_qat:
            prepare_qat(model)

        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

        if prep_qat:
            self.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            torch.quantization.prepare_qat(self, inplace=True)

    def prepare_qat(self):
        pass

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

