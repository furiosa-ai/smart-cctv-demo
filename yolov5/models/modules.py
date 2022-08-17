from turtle import forward
from torch import nn


class SelectOutput(nn.Module):
    def __init__(self, idx):
        super().__init__()

        self.idx = idx

    def forward(self, x):
        return x[self.idx]
