
import time
import torch
import torchvision
import numpy as np

from utils.general import check_version, xywh2xyxy
from utils.metrics import box_iou
from utils.mot.cmot_tools import box_decode
# from utils.mot.box_decode import box_decode
from utils.util import dump_args



def nms(prediction, iou_thres=0.45, class_agnostic=True):
    # Checks
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height

    output = []
    for x in prediction:  # image index, image inference
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        # Batched NMS
        if not class_agnostic:
            c = x[:, 5:6] * max_wh  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        else:
            boxes, scores = x[:, :4], x[:, 4]
        
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

        output.append(x[i])

    return output


class BoxDecoderBase:
    def __init__(self, nc, anchors, stride, conf_thres) -> None:
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.anchors = anchors
        self.nl = anchors.shape[0] # number of detection layers
        self.na = anchors.shape[1]  # number of anchors
        self.grid = [None for _ in range(self.nl)]
        self.anchor_grid = [None for _ in range(self.nl)]
        self.stride = stride
        self.conf_thres = conf_thres

class BoxDecoderTorch(BoxDecoderBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def _from_yolo_det_layer(det_layer, *args, **kwargs):
        return BoxDecoderTorch(det_layer.nc, det_layer.anchors, det_layer.stride, *args, **kwargs)

    @staticmethod
    def from_weights(weights, *args, **kwargs):
        model = torch.load(weights, map_location="cpu")
        det_layer = model["model"].model[-1]

        return BoxDecoderTorch._from_yolo_det_layer(det_layer, *args, **kwargs)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid

    def __call__(self, x):
        x = [torch.from_numpy(t) for t in x]

        z = []
        for i, y in enumerate(x):
            bs, _, ny, nx, _ = y.shape

            if self.grid[i] is None:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            assert self.grid[i].shape[2:4] == x[i].shape[2:4]

            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            y = y.view(bs, -1, self.no)

            z.append(y)

        out = []
        # zc = z[..., 4] > self.conf_thres  # candidates

        z = torch.cat(z, 1)
        
        for x in z:
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[x[..., 4] > self.conf_thres]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > self.conf_thres]

            # Check shape
            n = x.shape[0]  # number of boxes
            if n == 0:  # no boxes
                continue

            out.append(x)

        return out  # batch * num_boxes * 6 (xyxy, conf, cls)
    

class BoxDecoder(BoxDecoderBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def _from_yolo_det_layer(det_layer, *args, **kwargs):
        return BoxDecoder(det_layer.nc, det_layer.anchors.numpy().astype(np.float32), det_layer.stride.numpy().astype(np.float32), *args, **kwargs)

    @staticmethod
    def from_weights(weights, *args, **kwargs):
        model = torch.load(weights, map_location="cpu")
        det_layer = model["model"].model[-1]

        return BoxDecoder._from_yolo_det_layer(det_layer, *args, **kwargs)

    def __call__(self, feats):
        assert all(isinstance(feat, np.ndarray) for feat in feats)

        # dump_args("box_decode_test_args", dict(anchors=self.anchors, stride=self.stride, conf_thres=self.conf_thres, feats=feats))
        out_boxes_batched = box_decode(self.anchors, self.stride, self.conf_thres, feats)
        
        # return (torch.cat(z, 1), x)
        return out_boxes_batched