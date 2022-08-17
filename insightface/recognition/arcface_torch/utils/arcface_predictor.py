
import os
from pathlib import Path
from typing import List, OrderedDict

import numpy as np
import torch

from backbones import get_model
from eval import verification
from bin_to_np import load_np
from ai_util.inference_framework import CalibrationDatasetImage, PredictorBase

from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed


class ArcFacePredictorSingle(PredictorBase):
    def __init__(self, cfg, weights, name=None, normalize=False, **kwargs) -> None:
        if name is None:
            name = Path(cfg).stem

        if isinstance(cfg, str):
            cfg = get_config(cfg, [])
            # global control random seed
            setup_seed(seed=cfg.seed, cuda_deterministic=False)
        
        super().__init__(name=name, norm_mean=0.5, norm_std=0.5, **kwargs)
        self.cfg = cfg
        self.weights = weights
        self.normalize = normalize

    def _normalize_feat(self, feat):
        return feat / np.linalg.norm(feat, axis=1, keepdims=True)

    def preproc(self, x, input_format, input_prec):
        if x.ndim == 4:
            assert x.shape[0] == 1
            x = x[0]

        x = self.preproc_img(x, (112, 112), input_format, input_prec)
        # x = ((x.astype(np.float32) / 255) - 0.5) / 0.5

        return x, None

    def postproc(self, pred, info):
        if isinstance(pred, (tuple, list)):
            assert len(pred) == 1
            pred = pred[0]

        if self.normalize:
            pred = self._normalize_feat(pred)

        return torch.from_numpy(pred)

    def build_model(self):
        backbone = get_model(
            self.cfg.network, dropout=0.0, fp16=False, num_features=self.cfg.embedding_size, act=self.cfg.act).eval()

        checkpoint = torch.load(self.weights, map_location="cpu")

        backbone.load_state_dict(checkpoint)
        backbone.eval()

        return backbone

    def to(self, device, *args, **kwargs):
        self.get_model()
        return super().to(device, *args, **kwargs)

    def get_calibration_dataset(self):
        path = '../data/ms1m-retinaface-t1/lfw.bin'
        dataset = verification.load_bin(path, image_size=(112, 112), limit=10 if (self.calib_mode is None or self.calib_mode == "minmax") else 100)
        dataset = dataset[0][0]
        dataset = ((dataset / 255) - 0.5) / 0.5
        dataset = dataset.numpy()  # only images, no flip
        return dataset 


class ArcFacePredictor(ArcFacePredictorSingle):
    def __init__(self, cfg, weights, name=None, normalize=False, flip_inputs=True, **kwargs) -> None:
        super().__init__(cfg, weights, name, normalize, **kwargs)

        self.single_batch_size = self.batch_size
        self.batch_size = self.batch_size // 2

        if self.batch_size == 0:
            is_single_batch = True
            self.batch_size = 1
        else:
            is_single_batch = False

        self.is_single_batch = is_single_batch
        self.flip_inputs = flip_inputs

    def _infer_single(self, inputs):
        return super().infer(inputs)

    def infer(self, inputs):
        if not self.flip_inputs:
            return self._infer_single(inputs)
        else:
            n = inputs.shape[0]
            assert self.input_format == "chw"
            inputs_flipped = np.flip(inputs, axis=self.get_dim_index("w"))
            inputs = np.concatenate([inputs, inputs_flipped])

            if self.is_single_batch:
                # need to split input
                assert n == 1
                x = np.concatenate([self._infer_single(inputs[i:i+1]) for i in range(2)])
            else:
                x = self._infer_single(inputs)

            x = x.reshape(2, self.batch_size, x.shape[1])
            # average over LR
            x = x[0] + x[1]

            return x
