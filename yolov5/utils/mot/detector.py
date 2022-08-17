import torch
import numpy as np
import cv2

from utils.augmentations import letterbox
from utils.mot.box_decoder import BoxDecoder, BoxDecoderTorch, nms
from utils.util import PerfMeasure

from utils.logging import log_func

class Detector:
    def __init__(self, det_model, conf_thres, iou_thres, boxdec_backend="c"):
        assert boxdec_backend in ("c", "python")
        # pass model pool
        self.det_model = det_model
        self.input_size = det_model.input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        if boxdec_backend == "c":
            box_decode = BoxDecoder.from_weights(self.det_model.weights, conf_thres=conf_thres)
        elif boxdec_backend == "python":
            box_decode = BoxDecoderTorch.from_weights(self.det_model.weights, conf_thres=conf_thres)

        self.box_decode = box_decode

    @log_func
    def _resize(self, img):
        w, h = self.input_size
        return letterbox(img, (h, w), auto=False)

    @log_func
    def _cvt_color(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @log_func
    def _transpose(self, img):
        return img.transpose(2, 0, 1)

    @log_func
    def _normalize(self, img):
        img = img.astype(np.float32) / 255
        return img

    @log_func
    def _preproc(self, img):
        img, (sx, sy), (padw, padh) = self._resize(img)
        img = self._cvt_color(img)

        if self.det_model.input_format == "chw":
            img = self._transpose(img)
        else:
            assert self.det_model.input_format == "hwc"

        if self.det_model.input_type == "f32":
            img = self._normalize(img)
        else:
            assert self.det_model.input_type == "i8"

        assert sx == sy
        scale = sx

        return img, (scale, (padw, padh))

    @log_func
    def _post_proc_transpose(self, pred):
        na = self.box_decode.na
        pred = [np.ascontiguousarray(feat.reshape(feat.shape[0], na, -1, feat.shape[2], feat.shape[3]).transpose(0, 1, 3, 4, 2)) for feat in pred]
        return pred

    @log_func
    def _decode(self, outputs):
        return self.box_decode(outputs)

    @log_func
    def _nms(self, pred):
        return nms(pred, self.iou_thres)

    @log_func
    def _postproc(self, pred, prepoc_params):
        with PerfMeasure(f"Postproc transpose"):
            pred = self._post_proc_transpose(pred)

        with PerfMeasure(f"Box decode"):
            pred_decode = self._decode(pred)

        with PerfMeasure(f"NMS"):
            boxes = self._nms(pred_decode)

        scale, (padw, padh) = prepoc_params

        assert len(boxes) == 1, "No batch supported"
        boxes = boxes[0]

        boxes[:, [0, 2]] = (1 / scale) * (boxes[:, [0, 2]] - padw)
        boxes[:, [1, 3]] = (1 / scale) * (boxes[:, [1, 3]] - padh)

        return boxes
    
    @log_func
    def __call__(self, img):
        img, prepoc_params = self._preproc(img)
        with PerfMeasure("Prediction Latency"):
            pred = self.det_model(img[None])
        boxes = self._postproc(pred, prepoc_params)

        return boxes
