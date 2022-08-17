import cv2
import glob
import os
import torch
import numpy as np
from torchvision import transforms


class LoadImages:
    def __init__(self, input_src, output_dict=True, resize=None, as_tensor=False, img_count=None) -> None:
        img_files = sorted(glob.glob(os.path.join(input_src, "*")))

        if img_count is not None:
            img_files = img_files[:img_count]

        self.input_src = input_src
        self.img_files = img_files
        self.output_dict = output_dict

        self.resize = resize
        self.as_tensor = as_tensor

    def __len__(self):
        return len(self.img_files)

    def __call__(self, key):
        assert len(self.img_files) > 0, f"No image files found in '{self.input_src}'"

        assert os.path.isfile(self.img_files[key])
        out = cv2.imread(self.img_files[key])
        assert out is not None, f"Could not read '{self.img_files[key]}'"
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        if self.resize is not None:
            r = self.resize if isinstance(self.resize, (tuple, list)) else (self.resize, self.resize)
            out = cv2.resize(out, r)

        if self.as_tensor:
            out = transforms.ToTensor()(out)
        else:
            out = (out.astype(np.float32) / 255).transpose(2, 0, 1)

        return out
