
import sys
import numpy as np
from pathlib import Path
import torch
import torchreid
from torchreid.data.transforms import build_transforms
from torchreid.utils import load_pretrained_weights
from torchreid.utils.torchtools import load_checkpoint
from ai_util.inference_framework import CalibrationDatasetImage, PredictorBase
from torchvision.transforms import Compose
from PIL import Image
from scripts.default_config import get_default_config, imagedata_kwargs, lr_scheduler_kwargs, optimizer_kwargs


def _load_cfg(config_file, weights):
    cfg = get_default_config()
    cfg.merge_from_file(config_file)
    cfg.use_gpu = False
    cfg.data.root = "../data/"
    cfg.model.load_weights = weights
    cfg.test.evaluate = True
    return cfg


class ReIdPredictor(PredictorBase):
    def __init__(self, name=None, engine=None, cfg=None, weights=None, batch_size=None, 
        input_size=(128, 256), output_type="torch", dist_metric=None, pad_batch=True, **kwargs) -> None:
        assert (engine is not None or cfg is not None) and not (engine is not None and cfg is not None)

        if cfg is not None:
            if isinstance(cfg, str):
                if name is None:
                    name = Path(cfg).stem

                cfg = _load_cfg(cfg, weights)

            batch_size = cfg.test.batch_size if batch_size is None else batch_size
            dist_metric = cfg.test.dist_metric

        if batch_size is None:
            batch_size = 1

        super().__init__(name=name, batch_size=batch_size, pad_batch=pad_batch, 
            norm_mean=np.float32([0.485, 0.456, 0.406]), norm_std=np.float32([0.229, 0.224, 0.225]), **kwargs)
        self.input_size = input_size
        self.engine = engine
        self.cfg = cfg
        self.pytorch_model = None
        self.calib_dataset = None
        self.output_type = output_type
        self.dist_metric = dist_metric

    def preproc(self, x, input_format, input_prec):
        x = self.preproc_img(x, self.input_size, input_format, input_prec)

        return x, None

    def postproc(self, pred, info):
        if isinstance(pred, (tuple, list)):
            assert len(pred) == 1
            pred = pred[0]

        if self.output_type == "torch":
            pred = torch.from_numpy(pred)
        return pred

    def build_model(self):
        if self.engine is None:
            datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(self.cfg))

            model = torchreid.models.build_model(
                name=self.cfg.model.name,
                num_classes=datamanager.num_train_pids,
                loss=self.cfg.loss.name,
                pretrained=self.cfg.model.pretrained,
                use_gpu=self.cfg.use_gpu
            )

            # checkpoint = load_checkpoint(self.cfg.model.load_weights)["state_dict"]
            # model.load_state_dict(checkpoint)
            load_pretrained_weights(model, self.cfg.model.load_weights)

            self.pytorch_model = model.eval()
            self.calib_dataset = datamanager.test_dataset
        else:
            self.pytorch_model = self.engine.model.eval()
            self.calib_dataset = self.engine.datamanager.test_dataset

        return self.pytorch_model

    def get_calibration_dataset(self):
        if self.skip_preproc:
            transform = lambda x: self.preproc(x)[0]
        else:
            transform = None

        data = CalibrationDatasetImage("../data/market1501/Market-1501-v15.09.15/bounding_box_train/*", needs_preproc=True, 
            limit=32 if (self.calib_mode is None or self.calib_mode == "minmax") else 100, 
            transform=transform
        )

        return data

