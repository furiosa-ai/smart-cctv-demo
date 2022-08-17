
from contextlib import ExitStack
import sys
import os
import cv2
import numpy as np
from pathlib import Path
import torch

from models.common import DetectMultiBackend
from models.yolo import Model
from yolo_predictor.box_decode.box_decoder import BoxDecoderC
from yolo_predictor.utils import letterbox, nms
from ai_util.inference_framework import CalibrationDatasetImage, PredictorBase
from utils.mot.tracker import Tracker as _Tracker


Tracker = _Tracker


# need tracker option
class Yolov5Predictor(PredictorBase):
    def __init__(self, cfg, weights, input_size, name=None, conf_thres=0.25, iou_thres=0.5, calib_data=None, **kwargs) -> None:
        if name is None:
            name = os.path.splitext(os.path.basename(os.path.dirname((os.path.dirname(weights)))))[0]
        
        super().__init__(name=name, **kwargs)

        self.cfg = cfg
        self.weights = weights
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.box_decoder = None
        self.calib_data = calib_data

    def build_model(self):
        with ExitStack() as stack:
            if "models" not in sys.modules:
                # fix imports
                from ai_util.imp_env import ImpEnv
                stack.enter_context(ImpEnv(
                    paths=Path(__file__).parent.parent,
                    imports=["models", "utils"]))
            model_load = torch.load(self.weights)["model"]
            model = Model(self.cfg, nc=model_load.model[-1].nc)
            # model_mutli_back = DetectMultiBackend(self.weights, device="cpu", dnn=False, data=None).model
            # model_load.fuse()
            model.load_state_dict(model_load.state_dict())
            # model.warmup()
        model.eval()
        det_layer = model.model[-1]
        det_layer.set_mode("export")

        self.box_decoder = BoxDecoderC(nc=det_layer.nc, anchors=det_layer.anchors.numpy(), stride=det_layer.stride.numpy(), conf_thres=self.conf_thres)

        return model

    def to(self, device, out_file=None):
        self.get_model()  # build model to load box decoder params
        return super().to(device, out_file)

    def preproc(self, x, input_format, input_prec):
        x, (sx, sy), (padw, padh) = letterbox(x, (self.input_size[1], self.input_size[0]), auto=False, scaleup=False)

        if input_format == "chw":
            x = x.transpose(2, 0, 1)

        if input_prec == "f32":
            x = x.astype(np.float32) / 255

        assert sx == sy
        scale = sx

        return x, (scale, (padw, padh))

    def postproc(self, feats_batched, info):
        def _reshape_output(feat):
            if isinstance(feat, torch.Tensor):
                feat = feat.cpu().numpy()

            return np.ascontiguousarray(feat.reshape(
                feat.shape[0], -1, self.box_decoder.nc + 5, feat.shape[2], feat.shape[3]
            ).transpose(0, 1, 3, 4, 2))

        boxes_batched = []

        for i in range(feats_batched[0].shape[0]):
            feats = [f[i:i+1] for f in feats_batched]
            feats = [_reshape_output(f) for f in feats]
            boxes = self.box_decoder(feats)

            if self.iou_thres is not None:
                boxes = nms(boxes, self.iou_thres)[0]

            # rescale boxes
            if info is not None:
                (scale, (padw, padh)) = info[i]
                boxes[:, [0, 2]] = (1 / scale) * (boxes[:, [0, 2]] - padw)
                boxes[:, [1, 3]] = (1 / scale) * (boxes[:, [1, 3]] - padh)

            boxes_batched.append(boxes)

        return boxes_batched


    def get_calibration_dataset(self):
        if self.calib_data is not None and not isinstance(self.calib_data, str):
            data = self.calib_data
        else:
            assert self.calib_data is not None
            path = self.calib_data if self.calib_data is not None else "../data/CrowdHuman/images/val/*"

            data = CalibrationDatasetImage(path, limit=10 if (self.calib_mode is None or self.calib_mode == "minmax") else 100, needs_preproc=True)
        return data

