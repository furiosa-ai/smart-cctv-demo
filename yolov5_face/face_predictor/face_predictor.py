from pathlib import Path
import cv2
import torch
import os
import sys
import numpy as np
from models.yolo import Model
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from ai_util.inference_framework import CalibrationDatasetImage, PredictorBase
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from contextlib import ExitStack

from models.experimental import attempt_load


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


class Yolov5FacePredictor(PredictorBase):
    def __init__(self, weights, cfg=None, img_size=640, conf_thres=0.25, iou_thres=0.5, name=None, **kwargs) -> None:
        if name is None:
            name = os.path.splitext(os.path.basename(weights))[0]

        super().__init__(name=name, **kwargs)
        self.cfg = cfg
        self.weights = weights
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size

    def preproc(self, img0, input_format, input_prec):
        h0, w0 = img0.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        # if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        # imgsz = check_img_size(self.img_size, s=model.stride.max())  # check img_size
        imgsz = self.img_size

        img = letterbox(img0, new_shape=imgsz, auto=False)[0]

        if input_format == "chw":
            img = img.transpose(2, 0, 1)
            img_shape = img.shape[1:]
        else:
            img_shape = img.shape[:2]

        if input_prec == "f32":
            img = img.astype(np.float32) / 255

        return img, (img_shape, (h0, w0))

    def postproc(self, pred, info):
        assert len(info) == 1

        pred = self._box_decode(pred)

        img_shape, orgimg_shape = info[0]
        pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)

        # Process detections
        out = []
        for i, det in enumerate(pred):  # detections per image
            out_img = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img_shape, det[:, :4], orgimg_shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                det[:, 5:15] = scale_coords_landmarks(img_shape, det[:, 5:15], orgimg_shape).round()

                for j in range(det.shape[0]):
                    xyxy = det[j, :4].cpu().numpy()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = det[j, 5:15].cpu().numpy()
                    # class_num = det[j, 15].cpu().numpy()

                    out_img.append((xyxy, conf, landmarks))
            out.append(out_img)
        
        return out

    def build_model(self):
        if self.cfg is not None:
            with ExitStack() as stack:
                # if "models" not in sys.modules:
                if True:
                    # fix imports
                    from ai_util.imp_env import ImpEnv
                    stack.enter_context(ImpEnv(
                        paths=Path(__file__).parent.parent,
                        imports=["models", "utils.datasets"]))
                model = Model(self.cfg)
                model.load_state_dict(torch.load(self.weights, map_location="cpu")['model'].float().state_dict())
        else:
            model = load_model(self.weights, "cpu")
        model.eval()

        detect_layer = model.model[-1]
        detect_layer.export_warboy = True

        self.grid = detect_layer.grid
        self.stride = detect_layer.stride
        self.anchor_grid = detect_layer.anchor_grid
        self.nl = detect_layer.nl
        self.no = detect_layer.no
        self.na = detect_layer.na
        self.nc = detect_layer.nc
        self.detect_layer = detect_layer

        return model

    def to(self, device, *args, **kwargs):
        self.get_model()

        if device == "cuda":
            self.grid = [g.to(device) for g in self.grid]
            self.stride = self.stride.to(device)
            self.anchor_grid = self.anchor_grid.to(device)

        return super().to(device, *args, **kwargs)

    def get_calibration_dataset(self):
        return CalibrationDatasetImage("../data/WIDER_val/images/*/*.jpg", needs_preproc=True, limit=10 if (self.calib_mode is None or self.calib_mode == "minmax") else 100)
        # return CalibrationDatasetImage("data/images/result.jpg", limit=1)

    def _box_decode(self, x):
        z = []
        
        for i in range(self.nl):
            if isinstance(x[i], np.ndarray):
                x[i] = torch.from_numpy(x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self.detect_layer._make_grid(nx, ny).to(x[i].device)

            y = torch.full_like(x[i], 0)
            class_range = list(range(5)) + list(range(15,15+self.nc))
            y[..., class_range] = x[i][..., class_range].sigmoid()
            y[..., 5:15] = x[i][..., 5:15]
            #y = x[i].sigmoid()

            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

            #y[..., 5:15] = y[..., 5:15] * 8 - 4
            y[..., 5:7]   = y[..., 5:7] *   self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i] # landmark x1 y1
            y[..., 7:9]   = y[..., 7:9] *   self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]# landmark x2 y2
            y[..., 9:11]  = y[..., 9:11] *  self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]# landmark x3 y3
            y[..., 11:13] = y[..., 11:13] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]# landmark x4 y4
            y[..., 13:15] = y[..., 13:15] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]# landmark x5 y5

            #y[..., 5:7] = (y[..., 5:7] * 2 -1) * self.anchor_grid[i]  # landmark x1 y1
            #y[..., 7:9] = (y[..., 7:9] * 2 -1) * self.anchor_grid[i]  # landmark x2 y2
            #y[..., 9:11] = (y[..., 9:11] * 2 -1) * self.anchor_grid[i]  # landmark x3 y3
            #y[..., 11:13] = (y[..., 11:13] * 2 -1) * self.anchor_grid[i]  # landmark x4 y4
            #y[..., 13:15] = (y[..., 13:15] * 2 -1) * self.anchor_grid[i]  # landmark x5 y5

            z.append(y.view(bs, -1, self.no))
        
        return torch.cat(z, 1)